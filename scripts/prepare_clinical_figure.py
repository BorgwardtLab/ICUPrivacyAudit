"""Reuse synthetic summaries and export only aggregate real-reference histograms."""

import json
from pathlib import Path
import shutil

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from syntheticicu.clinical_utility import clinical_metadata
from syntheticicu.data_classes import ICUDataset


def real_hour_histogram(values, times, endpoint_hours, bins):
    """Count physiological feature presence over the full clinical hourly grid."""
    values, times = np.asarray(values), np.asarray(times)
    if values.ndim != 2 or len(values) != len(times):
        raise ValueError("Real reference values and timestamps do not align")
    if not np.isfinite(times).all() or np.any(times != np.floor(times)):
        raise ValueError("Real reference hours must be finite integers")
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Real reference hours must be nonnegative and increasing")
    if not np.isfinite(endpoint_hours) or endpoint_hours <= 0:
        raise ValueError("Real clinical duration must be finite and positive")
    if np.any(values < 0) or np.any(values > bins) or not np.isfinite(values).all():
        raise ValueError("Real reference contains invalid discrete values")
    total_hours = int(np.ceil(np.nextafter(endpoint_hours, -np.inf)))
    inside = (times < endpoint_hours) & (times < total_hours)
    counts = np.sum(values[inside] < bins, axis=1)
    histogram = np.bincount(counts, minlength=values.shape[1] + 1)
    histogram[0] += total_hours - int(inside.sum())
    assert histogram.sum() == total_hours
    assert np.all(histogram >= 0)
    return histogram


def real_histograms(cfg, root):
    """Use exact source stays and current LOS; never export IDs or real sequences."""
    loader = ICUDataset.__new__(ICUDataset)
    loader._register_pickle_aliases()
    source = joblib.load(cfg.real_dataset_path)
    identifiers = np.asarray(source.y)[:, 0]
    if identifiers.dtype.kind == "f" and identifiers.dtype.itemsize < 8:
        raise ValueError("Float32 stay IDs cannot identify the real reference cohort")
    keys = [
        (str(center).lower(), int(identifier))
        for center, identifier in zip(source.stay_sources, identifiers)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Real reference has duplicate center/stay keys")
    metadata = clinical_metadata(cfg.real_metadata_path, cfg.real_hours_max)
    index = pd.MultiIndex.from_tuples(keys, names=["centre", "stay_id"])
    metadata = metadata.reindex(index)
    if metadata.endpoint_hours.isna().any():
        raise ValueError("Real reference cohort has no matching clinical LOS")
    if len(source.features) != 58 or source.num_bins < 2:
        raise ValueError("Real reference requires the same 58 physiological features")
    durations = metadata.endpoint_hours.to_numpy(dtype=np.float64)
    counts = np.zeros(len(source.features) + 1, dtype=np.int64)
    for position, duration in enumerate(durations):
        times = np.asarray(source.times[position])
        times = times[: int(np.argmax(times)) + 1]
        values = np.asarray(source.X_value_ids[position])[: len(times)]
        counts += real_hour_histogram(values, times, duration, source.num_bins)
    maximum = float(durations.max())
    for mode in cfg.modes:
        for path in (root / mode).glob("*_sequences.csv"):
            maximum = max(maximum, pd.read_csv(path, usecols=["hours"]).hours.max())
    edges = np.unique(np.r_[0, np.geomspace(1, max(2, maximum + 1), 35)])
    frequencies, _ = np.histogram(durations, bins=edges)
    pd.DataFrame(
        {"left_hours": edges[:-1], "right_hours": edges[1:], "stays": frequencies}
    ).to_csv(root / "real_duration_histogram.csv", index=False)
    pd.DataFrame({"tokens_per_hour": np.arange(len(counts)), "hours": counts}).to_csv(
        root / "real_tokens_per_hour.csv", index=False
    )
    summary = {
        "cohort": "full uncut discrete-generator source cohort",
        "stays": len(metadata),
        "hours": int(counts.sum()),
        "outcome_prevalence": float(metadata.death_icu.mean()),
        "duration_hours_median": float(np.median(durations)),
        "duration_hours_max": float(durations.max()),
        "feature_tokens_per_hour_mean": float(
            np.dot(np.arange(len(counts)), counts) / counts.sum()
        ),
        "real_dataset": str(cfg.real_dataset_path),
        "real_metadata": str(cfg.real_metadata_path),
        "features": list(source.features),
        "duration_definition": "full clinical los_icu * 24, without utility cutoff",
        "hour_count_definition": "selected-feature presence, including empty clinical hours",
    }
    (root / "real_reference.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
