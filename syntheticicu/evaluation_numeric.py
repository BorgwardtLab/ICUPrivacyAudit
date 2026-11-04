"""Continuous-value adapters for the shared clinical utility and privacy tasks."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from syntheticicu.data_numeric import PAD, SOS, EOS, GAP
from syntheticicu.workflow_numeric import data_signature


@dataclass
class EvaluationData:
    """Validated real partitions and a release from the same preparation run."""

    root: Path
    bundle: dict
    synthetic: list
    manifest: dict


def load_evaluation_data(root, require_release=True):
    """Load a complete release and reject mismatched signatures or identities."""
    root = Path(root).resolve()
    bundle = torch.load(root / "prepared.pt", map_location="cpu", weights_only=True)
    _validate_partitions(
        bundle["stays"],
        bundle.get("preparation_contract", {}).get("split_unit", "patient"),
    )
    if not require_release:
        return EvaluationData(root, bundle, [], {})
    manifest = json.loads((root / "generated/generation.json").read_text())
    if manifest["data_signature"] != data_signature(bundle):
        raise ValueError("Release data signature does not match prepared real data")
    synthetic = _load_synthetic(root, bundle["schema"], manifest)
    return EvaluationData(root, bundle, synthetic, manifest)


def _validate_partitions(stays, split_unit="patient"):
    if split_unit not in {"patient", "reference_stay"}:
        raise ValueError("Unknown preparation split unit")
    patients, keys = {}, set()
    for stay in stays:
        key = (stay["centre"], stay["stay_id"])
        if key in keys:
            raise ValueError("Duplicate real center/stay identity")
        keys.add(key)
        patient = stay["patient_uid"]
        if not isinstance(patient, int):
            raise ValueError("Patient identifiers must remain exact integers")
        split = stay["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown patient split")
        if split_unit == "patient" and patients.setdefault(patient, split) != split:
            raise ValueError("Patient overlaps evaluation partitions")


def _load_synthetic(root, schema, manifest):
    core = pd.read_parquet(root / "generated/core.parquet")
    meta = pd.read_parquet(root / "generated/meta.parquet")
    if len(meta) != manifest["num_samples"] or meta.stay_id.duplicated().any():
        raise ValueError("Release sample count or stay identities are invalid")
    if not pd.api.types.is_integer_dtype(meta.stay_id):
        raise ValueError("Synthetic stay identifiers must be integers")
    if not pd.api.types.is_integer_dtype(core.stay_id):
        raise ValueError("Synthetic core stay identifiers must be integers")
    if not pd.api.types.is_integer_dtype(meta.patient_uid):
        raise ValueError("Synthetic patient identifiers must be integers")
    if meta.patient_uid.duplicated().any():
        raise ValueError("Synthetic releases require unique generated patients")
    if set(core.stay_id) != set(meta.stay_id):
        raise ValueError("Synthetic core and metadata identities differ")
    if core.duplicated(["stay_id", "stay_time"]).any():
        raise ValueError("Duplicate synthetic hourly rows")
    if meta.death_icu.isna().any() or not meta.death_icu.isin([0, 1]).all():
        raise ValueError("Synthetic outcomes must be complete binary labels")
    if not np.isfinite(core.stay_time).all() or (core.stay_time < 0).any():
        raise ValueError("Synthetic hours must be finite nonnegative integers")
    if (core.stay_time % 1 != 0).any():
        raise ValueError("Synthetic hours must be integers")
    values = core[list(schema["features"])].to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("Synthetic measurements contain infinity")
    lookup = meta.set_index("stay_id")
    classifier_rows = classification_rows_from_tokens(root, schema)
    if meta.stay_id.sort_values().tolist() != list(range(len(classifier_rows))):
        raise ValueError("Synthetic token rows require contiguous generated stay IDs")
    stays = []
    for identifier, rows in core.groupby("stay_id", sort=True):
        info = lookup.loc[identifier]
        if set(rows.centre) != {info.centre}:
            raise ValueError("Synthetic core and metadata centers differ")
        rows = rows.sort_values("stay_time")
        stays.append(
            dict(
                stay_id=int(identifier),
                patient_uid=int(info.patient_uid),
                centre=str(info.centre),
                split="synthetic",
                death_icu=bool(info.death_icu),
                demographics={
                    name: float(info[name]) for name in schema["demographics"]
                },
                hours=torch.tensor(rows.stay_time.to_numpy(), dtype=torch.long),
                values=torch.tensor(
                    rows[list(schema["features"])].to_numpy(), dtype=torch.float32
                ),
                terminated=bool(info.terminated),
                classifier_values=classifier_rows[int(identifier)],
                classifier_hours=torch.arange(
                    1, len(classifier_rows[int(identifier)]) + 1
                ),
            )
        )
    return stays


def classification_rows_from_tokens(root, schema):
    """Preserve legacy nonempty token blocks, including the demographic-only row.

    Consecutive GAP tokens introduce no empty rows in the legacy classifier loader.
    Raw clinical hours remain separate from this classifier-only representation.
    """
    features = list(schema["features"])
    valid_ids = frozenset(schema["vocabulary"].values())
    mapping = {
        schema["vocabulary"][token]: column
        for column, name in enumerate(features)
        for token in schema["features"][name]["tokens"]
    }
    batches = torch.load(
        Path(root) / "generated/tokens.pt", map_location="cpu", weights_only=True
    )
    result = []
    for batch in batches:
        if batch["input_ids"].dtype != torch.long or batch["input_ids"].ndim != 2:
            raise ValueError("Synthetic token IDs must be an integer batch matrix")
        if batch["input_ids"].shape != batch["values"].shape:
            raise ValueError("Synthetic token/value rows are misaligned")
        for ids, values in zip(batch["input_ids"], batch["values"]):
            rows, block = [], []
            for token, value in zip(ids.tolist(), values.tolist()):
                if token not in valid_ids:
                    raise ValueError("Unknown generated token ID")
                if token == EOS:
                    break
                if token in {PAD, SOS}:
                    continue
                if token == GAP:
                    if block:
                        rows.append(_classification_row(block, mapping, len(features)))
                    block = []
                else:
                    block.append((token, value))
            if block:
                rows.append(_classification_row(block, mapping, len(features)))
            if not rows:
                raise ValueError("Synthetic sequence has no classifier token blocks")
            result.append(torch.tensor(np.asarray(rows), dtype=torch.float32))
    return result


def _classification_row(block, mapping, count):
    row = np.full(count, np.nan, dtype=np.float32)
    for token, value in block:
        if token in mapping:
            if not np.isfinite(value):
                raise ValueError("Numerical token has no finite generated value")
            row[mapping[token]] = value
    return row
