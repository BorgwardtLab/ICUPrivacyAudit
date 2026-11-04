"""Read existing releases for utility without rewriting their training artifacts."""

from types import SimpleNamespace

import joblib
import numpy as np
import torch

from syntheticicu.data_classes import EarlyWarningDataset, ICUDataset
from syntheticicu.evaluation_numeric import load_evaluation_data


def load_clinical_source(source, generation_mode, metadata, require_release):
    """Resolve one explicit representation; no inferred or substitute inputs."""
    if source.representation == "numerical":
        data = load_evaluation_data(
            source.roots[generation_mode], require_release=require_release
        )
        counts = [
            len(spec["edges"]) - 1
            for spec in data.bundle["schema"]["features"].values()
        ]
        if not counts or min(counts) < 1 or max(counts) != source.num_bins:
            raise ValueError("Numerical bins do not match the selected clinical source")
        return data
    if source.representation != "discrete":
        raise ValueError("Clinical source representation must be numerical or discrete")
    loader = ICUDataset.__new__(ICUDataset)
    loader._register_pickle_aliases()
    saved = joblib.load(source.real_path)
    if not hasattr(saved, "X_value_ids"):
        raise ValueError("Clinical discrete utility requires direct FIDDLE value IDs")
    features, bins = list(saved.features), int(saved.num_bins)
    if len(features) != len(set(features)) or bins != source.num_bins:
        raise ValueError(
            "Clinical source needs distinct features and the configured bin count"
        )
    schema = {
        "features": {
            name: {"edges": (np.arange(bins + 1) - 0.5).tolist()} for name in features
        },
        "demographics": {},
    }
    stays = discrete_real_stays(saved, metadata, features, bins)
    synthetic = (
        discrete_synthetic_stays(
            source.synthetic_paths[generation_mode], features, bins
        )
        if require_release
        else []
    )
    return SimpleNamespace(
        bundle={"schema": schema, "stays": stays}, synthetic=synthetic
    )


def discrete_real_stays(saved, metadata, features, bins):
    """Read stay-axis identities; never pair hourly metadata rows with stay tensors."""
    labels = np.asarray(saved.y)
    centers = np.asarray(saved.stay_sources)
    if labels.dtype.kind == "f" and labels.dtype.itemsize < 8:
        raise ValueError("Float32 discrete stay IDs cannot establish exact identity")
    if not (
        len(labels)
        == len(saved.stay_sources)
        == len(saved.X_value_ids)
        == len(saved.times)
    ):
        raise ValueError("Discrete stay axes do not align")
    stays, keys = [], set()
    for index, label in enumerate(labels):
        center, identifier = str(centers[index]).lower(), label[0]
        if not np.isfinite(identifier) or identifier != int(identifier):
            raise ValueError("Discrete stay identifier is not an exact integer")
        key = (center, int(identifier))
        if key in keys or key not in metadata.index:
            raise ValueError("Duplicate discrete stay or missing clinical metadata")
        keys.add(key)
        row = metadata.loc[key]
        hours = np.asarray(saved.times[index])
        hours = hours[: int(np.argmax(hours)) + 1]
        if not np.isfinite(hours).all() or np.any(hours != np.floor(hours)):
            raise ValueError("Discrete source hours must be finite integers")
        values = np.asarray(saved.X_value_ids[index])[: len(hours)].astype(np.float32)
        if values.shape != (len(hours), len(features)):
            raise ValueError("Discrete time/value axes do not align")
        if (
            np.any(values < 0)
            or np.any(values > bins)
            or np.any(values != np.floor(values))
        ):
            raise ValueError("Discrete bins contain invalid or noninteger values")
        values[values == bins] = np.nan
        stays.append(
            dict(
                centre=center,
                stay_id=key[1],
                patient_uid=int(row.patient_uid),
                split="not_audited",
                death_icu=bool(row.death_icu),
                demographics={},
                hours=torch.as_tensor(hours, dtype=torch.long),
                values=torch.from_numpy(values),
            )
        )
    return stays


def discrete_synthetic_stays(path, features, bins):
    """Reuse legacy nonempty-block decoding while retaining the entire sequence."""
    tokens, labels, centers = joblib.load(path)
    if not len(tokens) == len(labels) == len(centers):
        raise ValueError("Synthetic sequences, labels and centers do not align")
    decoder = EarlyWarningDataset.__new__(EarlyWarningDataset)
    decoder.features, decoder.num_bins = features, bins
    stays = []
    for index, (sequence, label) in enumerate(zip(tokens, labels)):
        if label not in [0, 1]:
            raise ValueError("Synthetic mortality conditioning must be binary")
        matrices, _ = decoder._map_tok_to_idx([sequence])
        one_hot = matrices[0].reshape(-1, len(features), bins)
        values = one_hot.argmax(axis=2).astype(np.float32)
        values[one_hot.sum(axis=2) == 0] = np.nan
        values = torch.from_numpy(values)
        hours = torch.arange(1, len(values) + 1, dtype=torch.long)
        stays.append(
            dict(
                centre="syn",
                stay_id=index,
                patient_uid=index,
                split="synthetic",
                death_icu=bool(label),
                demographics={},
                hours=hours,
                values=values,
                classifier_hours=hours,
                classifier_values=values,
            )
        )
    return stays
