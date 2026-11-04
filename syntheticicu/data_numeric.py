"""Parquet adapter for the opt-in numerical workflow."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype, is_bool_dtype
import pyarrow.dataset as ds
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from syntheticicu.model_numeric import NumericalBins


PAD, SOS, EOS, GAP = 0, 1, 2, 3


def apply_reference_weights(meta, path):
    """Fill only the two explicitly approved missing preprocessing weights."""
    rows = json.loads(Path(path).read_text())["weights"]
    if len(rows) != 2:
        raise ValueError("Exactly two approved weight overrides are required")
    keys = [(row["centre"], int(row["stay_id"])) for row in rows]
    if len(set(keys)) != 2:
        raise ValueError("Duplicate weight-override identity")
    result = meta.copy()
    for row, key in zip(rows, keys):
        weight = float(row["weight_static"])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(
                "Recorded preprocessing weight must be finite and positive"
            )
        selected = meta.centre.eq(key[0]) & meta.stay_id.eq(key[1])
        if selected.sum() != 1:
            raise ValueError("Approved weight override has no unique metadata stay")
        if not meta.loc[selected, "weight_static"].isna().all():
            raise ValueError("Weight override cannot replace an observed weight")
        result.loc[selected, "weight_static"] = weight
    return result


def read_numerical_tables(cfg):
    """Project selected columns and explicitly join metadata by center/stay.

    Sampling selects complete patients before reading the core Parquet file.
    Hourly duplicates are averaged; negative hours and hours outside the configured
    prediction window are excluded. No outcomes are read from the core table.
    """
    features, demographics = list(cfg.features), list(cfg.demographics)
    if len(set(features + demographics)) != len(features + demographics):
        raise ValueError("Features and demographics must be distinct")
    columns = ["centre", "stay_id", "patient_uid", "death_icu", *demographics]
    meta = pd.read_parquet(cfg.meta_path, columns=columns)
    meta["centre"] = meta.centre.str.lower()
    meta = meta.loc[meta.centre.isin(cfg.centers)].copy()
    for name in ["stay_id", "patient_uid"]:
        if meta[name].isna().any() or not is_integer_dtype(meta[name]):
            raise ValueError(f"{name} must contain exact integer identities")
    if meta.duplicated(["centre", "stay_id"]).any():
        raise ValueError("Metadata must have one row per center/stay")
    if meta.death_icu.isna().any() or not meta.death_icu.isin([0, 1]).all():
        raise ValueError("death_icu must be a complete binary label")
    if cfg.get("weight_overrides") is not None:
        meta = apply_reference_weights(meta, cfg.weight_overrides)
    reference = cfg.get("split_unit", "patient") == "reference_stay"
    if reference:
        if cfg.split_manifest is None or cfg.max_patients is not None:
            raise ValueError("Reference replay requires a full, explicit stay manifest")
        selections = pd.read_parquet(cfg.split_manifest, columns=["centre", "stay_id"])
        if selections.duplicated(["centre", "stay_id"]).any():
            raise ValueError("Reference stay manifest contains duplicate identities")
        meta = selections.merge(
            meta, on=["centre", "stay_id"], how="left", validate="one_to_one"
        )
        if meta.patient_uid.isna().any():
            raise ValueError("Reference stay is absent from current metadata")
    patients = np.sort(meta.patient_uid.unique())
    if cfg.max_patients is not None:
        if cfg.max_patients < 3:
            raise ValueError("max_patients must be at least three")
        if len(patients) > cfg.max_patients:
            patients = np.random.default_rng(cfg.seed).choice(
                patients, cfg.max_patients, replace=False
            )
            meta = meta.loc[meta.patient_uid.isin(patients)].copy()
    if meta.empty:
        raise ValueError("No metadata patients selected")
    if cfg.max_hours is not None and cfg.max_hours < 1:
        raise ValueError("max_hours must be positive")
    dataset = ds.dataset(str(cfg.core_path), format="parquet")
    selected = (
        ds.field("stay_id").isin(meta.stay_id.unique().tolist())
        & ds.field("centre").isin(list(cfg.centers))
        & (ds.field("stay_time") >= 0)
    )
    if cfg.max_hours is not None:
        selected = selected & (ds.field("stay_time") < cfg.max_hours)
    core = dataset.to_table(
        columns=["centre", "stay_id", "stay_time", *features], filter=selected
    ).to_pandas()
    for name in features:
        if not is_numeric_dtype(core[name]) or is_bool_dtype(core[name]):
            raise ValueError(f"{name} is not a continuous numerical feature")
        if np.isinf(core[name].to_numpy(dtype=float)).any():
            raise ValueError(f"{name} contains infinite values")
    core = core.merge(
        meta[["centre", "stay_id"]],
        on=["centre", "stay_id"],
        how="inner",
        validate="many_to_one",
    )
    if cfg.normalize_urine_by_weight:
        if "urine_raw" not in features or "weight_static" not in demographics:
            raise ValueError("Urine normalization requires urine_raw and weight_static")
        core = core.merge(
            meta[["centre", "stay_id", "weight_static"]],
            on=["centre", "stay_id"],
            validate="many_to_one",
        )
        if ((core.weight_static <= 0) & core.urine_raw.notna()).any():
            raise ValueError(
                "Observed urine requires positive weight for normalization"
            )
        core["urine_raw"] = core.urine_raw / core.weight_static
        core = core.drop(columns="weight_static")
    core["hour"] = np.floor(core.stay_time).astype(np.int64)
    core = (
        core.groupby(["centre", "stay_id", "hour"], sort=True)[features]
        .mean()
        .reset_index()
    )
    observed = core.loc[
        core[features].notna().any(axis=1), ["centre", "stay_id"]
    ].drop_duplicates()
    if cfg.get("trim_empty_boundaries", False):
        bounds = (
            core.loc[core[features].notna().any(axis=1)]
            .groupby(["centre", "stay_id"])
            .hour.agg(["min", "max"])
        )
        core = core.join(bounds, on=["centre", "stay_id"], validate="many_to_one")
        core = core.loc[core.hour.ge(core["min"]) & core.hour.le(core["max"])].drop(
            columns=["min", "max"]
        )
    if cfg.min_hours > 0 and not reference:
        durations = core.groupby(["centre", "stay_id"]).hour.max().add(1)
        eligible = durations[durations >= cfg.min_hours].reset_index()[
            ["centre", "stay_id"]
        ]
        observed = observed.merge(
            eligible, on=["centre", "stay_id"], validate="one_to_one"
        )
    if reference and len(observed) != len(meta):
        raise ValueError(
            "Reference cohort contains stays without numerical observations"
        )
    meta = meta.merge(observed, on=["centre", "stay_id"], validate="one_to_one")
    core = core.merge(
        meta[["centre", "stay_id"]], on=["centre", "stay_id"], validate="many_to_one"
    )
    if meta.empty:
        raise ValueError("No stays contain observations for the selected features")
    return core, meta


def assign_patient_splits(meta, cfg):
    """Use patient splits by default; replay legacy stay partitions only explicitly."""
    split_unit = cfg.get("split_unit", "patient")
    if split_unit not in {"patient", "reference_stay"}:
        raise ValueError("split_unit must be patient or reference_stay")
    if split_unit == "reference_stay" and cfg.split_manifest is None:
        raise ValueError("Reference stay replay requires a split manifest")
    if cfg.split_manifest is not None:
        splits = pd.read_parquet(
            cfg.split_manifest, columns=["centre", "stay_id", "split"]
        )
        if splits.duplicated(["centre", "stay_id"]).any():
            raise ValueError("Split manifest has duplicate center/stay rows")
        meta = meta.merge(
            splits, on=["centre", "stay_id"], how="left", validate="one_to_one"
        )
        if (
            meta.split.isna().any()
            or not meta.split.isin(["train", "val", "test"]).all()
        ):
            raise ValueError("Split manifest must cover every selected stay")
    else:
        if not (
            0 < cfg.val_fraction < 1
            and 0 < cfg.test_fraction < 1
            and cfg.val_fraction + cfg.test_fraction < 1
        ):
            raise ValueError(
                "Positive validation/test fractions must sum to less than one"
            )
        patients = np.sort(meta.patient_uid.unique())
        train_val, test = train_test_split(
            patients, test_size=cfg.test_fraction, random_state=cfg.seed
        )
        train, val = train_test_split(
            train_val,
            test_size=cfg.val_fraction / (1 - cfg.test_fraction),
            random_state=cfg.seed,
        )
        mapping = {
            int(patient): split
            for split, ids in [("train", train), ("val", val), ("test", test)]
            for patient in ids
        }
        meta = meta.copy()
        meta["split"] = meta.patient_uid.map(mapping)
    if (
        split_unit == "patient"
        and (meta.groupby("patient_uid").split.nunique() != 1).any()
    ):
        raise ValueError("Split manifest places one patient in multiple partitions")
    if set(meta.split) != {"train", "val", "test"}:
        raise ValueError("All three patient partitions must be nonempty")
    return meta


def _edges(values, num_bins, feature):
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    if len(observed) == 0:
        raise ValueError(f"No training observations for {feature}")
    edges = np.unique(
        np.quantile(observed, np.linspace(0, 1, num_bins + 1)).astype(np.float32)
    )
    if len(edges) < 2:
        raise ValueError(
            f"Constant training feature {feature}; remove it from the configured features"
        )
    return edges.tolist()


def fit_numerical_schema(core, meta, cfg):
    """Fit physiology and demographic categories using training patients only."""
    if cfg.num_bins < 2:
        raise ValueError("num_bins must be at least two")
    training = core.merge(
        meta.loc[meta.split == "train", ["centre", "stay_id"]],
        on=["centre", "stay_id"],
        validate="many_to_one",
    )
    vocabulary = {"<PAD>": PAD, "<SOS>": SOS, "<EOS>": EOS, "<GAP>": GAP}
    features, demographics = {}, {}
    for name in cfg.features:
        edges = _edges(training[name], cfg.num_bins, name)
        tokens = [f"{name}_bin_{index}" for index in range(len(edges) - 1)]
        features[name] = {"edges": edges, "tokens": tokens}
        for token in tokens:
            vocabulary[token] = len(vocabulary)
    for name in cfg.demographics:
        values = meta.loc[meta.split == "train", name].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all() or len(values) == 0:
            raise ValueError(f"Need finite training demographics for {name}")
        if name == "female_static" or len(np.unique(values)) == 1:
            categories = sorted(np.unique(values).tolist())
            specification = {"categories": categories}
            count = len(categories)
        else:
            edges = _edges(values, cfg.num_bins, name)
            specification = {"edges": edges}
            count = len(edges) - 1
        tokens = [f"{name}_bin_{index}" for index in range(count)] + [f"{name}_missing"]
        specification["tokens"] = tokens
        demographics[name] = specification
        for token in tokens:
            vocabulary[token] = len(vocabulary)
    return {
        "vocabulary": vocabulary,
        "features": features,
        "demographics": demographics,
        "centers": list(cfg.centers),
        "max_hours": (
            int(training.hour.max()) + 1
            if cfg.max_hours is None
            else int(cfg.max_hours)
        ),
    }


def numerical_bins(schema):
    """Construct continuous bounds without treating demographics as continuous."""
    return NumericalBins.from_vocabulary(
        schema["vocabulary"],
        {name: spec["edges"] for name, spec in schema["features"].items()},
    )


def category_index(value, specification, out_of_range):
    """Map a value to a fitted bin or category under an explicit range policy."""
    if pd.isna(value):
        return len(specification["tokens"]) - 1
    if "categories" in specification:
        return specification["categories"].index(float(value))
    edges = specification["edges"]
    if out_of_range == "error" and not edges[0] <= value <= edges[-1]:
        raise ValueError("Observation outside training bin bounds")
    return int(np.searchsorted(edges[1:-1], value, side="right"))


def encode_numerical_stay(
    stay, schema, max_tokens, out_of_range, trim_terminal_gap=False
):
    """Chunk whole hours while preserving exact token/value/time alignment."""
    vocabulary = schema["vocabulary"]
    bins = numerical_bins(schema)
    prefix = [SOS]
    for name, specification in schema["demographics"].items():
        index = category_index(stay["demographics"][name], specification, out_of_range)
        prefix.append(vocabulary[specification["tokens"][index]])
    prefix.append(GAP)
    records, current = [], []
    clipped = 0

    def finish(events, terminal_trim=False):
        ids = prefix + [event[0] for event in events] + [EOS]
        values = (
            [float("nan")] * len(prefix)
            + [event[1] for event in events]
            + [float("nan")]
        )
        end_time = events[-1][2] + int(not terminal_trim)
        times = [0] * len(prefix) + [event[2] for event in events] + [end_time]
        ids = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": ids,
            "residuals": bins.encode(ids, torch.tensor(values)),
            "time_positions": torch.tensor(times, dtype=torch.long),
            "class_labels": int(stay["death_icu"]),
            "center_ids": schema["centers"].index(stay["centre"]),
            "patient_uid": stay["patient_uid"],
            "stay_id": stay["stay_id"],
            "centre": stay["centre"],
            "split": stay["split"],
        }

    for hour, row in zip(stay["hours"].tolist(), stay["values"].numpy()):
        events = []
        for (name, specification), raw in zip(schema["features"].items(), row):
            if np.isnan(raw):
                continue
            edges = specification["edges"]
            outside = raw < edges[0] or raw > edges[-1]
            clipped += int(outside)
            index = category_index(raw, specification, out_of_range)
            value = float(np.clip(raw, edges[0], edges[-1]))
            events.append((vocabulary[specification["tokens"][index]], value, hour + 1))
        events.append((GAP, float("nan"), hour + 1))
        if len(prefix) + len(events) + 1 > max_tokens:
            raise ValueError(
                "max_tokens cannot fit one complete hour plus demographics"
            )
        if len(prefix) + len(current) + len(events) + 1 > max_tokens:
            records.append(finish(current))
            current = []
        current.extend(events)
    if current:
        if trim_terminal_gap:
            while current and current[-1][0] == GAP:
                current.pop()
            if not current:
                raise ValueError(
                    "Final numerical chunk has no physiological observation"
                )
        records.append(finish(current, terminal_trim=trim_terminal_gap))
    return records, clipped


def prepare_numerical_data(cfg):
    """Read Parquet inputs and return a tensor-only, serializable data bundle."""
    if cfg.out_of_range not in {"clip", "error"}:
        raise ValueError("out_of_range must be clip or error")
    core, meta = read_numerical_tables(cfg)
    meta = assign_patient_splits(meta, cfg)
    schema = fit_numerical_schema(core, meta, cfg)
    metadata = {
        (row["centre"], int(row["stay_id"])): row for row in meta.to_dict("records")
    }
    stays, records = [], []
    report = {
        "clipped": {split: 0 for split in ["train", "val", "test"]},
        "out_of_range": str(cfg.out_of_range),
    }
    for key, rows in core.groupby(["centre", "stay_id"], sort=True):
        row = metadata[key]
        first = int(rows.hour.min()) if cfg.get("trim_empty_boundaries", False) else 0
        hours = np.arange(first, int(rows.hour.max()) + 1)
        values = (
            rows.set_index("hour")[list(schema["features"])]
            .reindex(hours)
            .to_numpy(dtype=np.float32)
        )
        stay = {
            "centre": str(row["centre"]),
            "stay_id": int(row["stay_id"]),
            "patient_uid": int(row["patient_uid"]),
            "split": str(row["split"]),
            "death_icu": bool(row["death_icu"]),
            "demographics": {name: float(row[name]) for name in cfg.demographics},
            "hours": torch.tensor(hours, dtype=torch.long),
            "values": torch.from_numpy(values),
        }
        chunks, clipped = encode_numerical_stay(
            stay,
            schema,
            cfg.max_tokens,
            cfg.out_of_range,
            trim_terminal_gap=cfg.get("trim_empty_boundaries", False),
        )
        records.extend(chunks)
        stays.append(stay)
        report["clipped"][stay["split"]] += clipped
    report["stays"] = len(stays)
    report["chunks"] = len(records)
    report["patients"] = len({stay["patient_uid"] for stay in stays})
    result = {"schema": schema, "stays": stays, "records": records, "report": report}
    if cfg.get("split_unit", "patient") == "reference_stay" or cfg.get(
        "trim_empty_boundaries", False
    ):
        result["preparation_contract"] = {
            "split_unit": cfg.get("split_unit", "patient"),
            "trim_empty_boundaries": bool(cfg.get("trim_empty_boundaries", False)),
            "encoding_version": 1,
        }
        report.update(result["preparation_contract"])
        report["patients_crossing_partitions"] = int(
            meta.groupby("patient_uid").split.nunique().gt(1).sum()
        )
    return result


def collate_numerical(records, shuffle=False):
    """Jointly pad and optionally shuffle physiology tokens within their hour."""
    rows = []
    for record in records:
        ids, residuals, times = [
            record[key].clone() for key in ["input_ids", "residuals", "time_positions"]
        ]
        if shuffle:
            for hour in times[times > 0].unique():
                indices = (
                    ((times == hour) & (ids != GAP) & (ids != EOS)).nonzero().flatten()
                )
                order = indices[torch.randperm(len(indices))]
                ids[indices], residuals[indices] = ids[order], residuals[order]
        rows.append((ids, residuals, times))
    return {
        "input_ids": pad_sequence(
            [row[0] for row in rows], batch_first=True, padding_value=PAD
        ),
        "residuals": pad_sequence(
            [row[1] for row in rows], batch_first=True, padding_value=0
        ),
        "time_positions": pad_sequence(
            [row[2] for row in rows], batch_first=True, padding_value=0
        ),
        "class_labels": torch.tensor(
            [row["class_labels"] for row in records], dtype=torch.long
        ),
        "center_ids": torch.tensor(
            [row["center_ids"] for row in records], dtype=torch.long
        ),
    }


def repeated_training_indices(bundle, n_canaries, duplication_factor, seed):
    """Duplicate complete training patients; never repeat validation/test rows."""
    if duplication_factor < 1 or n_canaries < 0:
        raise ValueError(
            "Positive duplication factor and nonnegative canary count required"
        )
    patients = sorted(
        {
            record["patient_uid"]
            for record in bundle["records"]
            if record["split"] == "train"
        }
    )
    if n_canaries > len(patients) or (duplication_factor > 1 and n_canaries == 0):
        raise ValueError("Choose a nonempty canary set within the training patients")
    canaries = [
        int(value)
        for value in np.random.default_rng(seed).choice(
            patients, n_canaries, replace=False
        )
    ]
    indices = []
    for index, record in enumerate(bundle["records"]):
        if record["split"] == "train":
            copies = duplication_factor if record["patient_uid"] in canaries else 1
            indices.extend([index] * copies)
    return indices, canaries
