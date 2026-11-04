"""Extract comparable generated-hour and feature-token histogram inputs."""

import gc
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from syntheticicu.privacy_bridge_numeric import export_audit_corpus


def sequence_hour_counts(sequence):
    """Count feature tokens per hour, excluding demographic prefixes and specials."""
    if len(sequence) > 10000:
        raise ValueError("Unexpected corpus sequence beyond the bounded token budget")
    counts, current, prefix, complete = [], 0, True, False
    for token in sequence:
        if not isinstance(token, str):
            raise TypeError("Histogram extraction requires decoded string tokens")
        if token == "<EOS>":
            complete = True
            break
        if token in {"<SOS>", "<PAD>"}:
            continue
        if token == "<GAP>":
            if not prefix or current:
                counts.append(current)
            current, prefix = 0, False
        elif "_static" not in token:
            current += 1
    if current:
        counts.append(current)
    return np.asarray(counts, dtype=np.int32), complete


def summarize_corpus(path, scheme, mode, destination):
    """Save per-sequence durations and a pooled feature-count histogram."""
    sequences, labels, _ = joblib.load(path)
    labels = np.asarray(labels).reshape(-1)
    if len(sequences) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Each generated sequence needs one binary outcome label")
    rows, hour_histogram = [], np.zeros(10001, dtype=np.int64)
    for index, (sequence, label) in enumerate(zip(sequences, labels)):
        counts, complete = sequence_hour_counts(sequence)
        hour_histogram += np.bincount(counts, minlength=len(hour_histogram))
        rows.append(
            {
                "sequence": index,
                "outcome": int(label),
                "hours": len(counts),
                "feature_tokens": int(counts.sum()),
                "complete": complete,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(destination / f"{scheme}_{mode}_sequences.csv", index=False)
    pd.DataFrame(
        {"tokens_per_hour": np.arange(len(hour_histogram)), "hours": hour_histogram}
    ).query("hours > 0").to_csv(
        destination / f"{scheme}_{mode}_tokens_per_hour.csv", index=False
    )
    result = {
        "scheme": scheme,
        "mode": mode,
        "path": str(path),
        "sequences": len(table),
        "outcome_prevalence": float(labels.mean()),
        "truncated_fraction": float(1 - table.complete.mean()),
        "duration_hours_median": float(table.hours.median()),
        "feature_tokens_per_hour_mean": float(
            table.feature_tokens.sum() / table.hours.sum()
        ),
    }
    del sequences, table
    gc.collect()
    print(result, flush=True)
    return result
