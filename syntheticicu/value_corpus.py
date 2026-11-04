"""Memory-mappable trajectory storage and deterministic conditioning pools."""

from pathlib import Path

import joblib
import numpy as np

from syntheticicu.data_classes import CENTRE_MAP
from syntheticicu.data_numeric import category_index
from syntheticicu.value_distances import scale_values, nearest_packed


def array(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def pack_stays(stays, schema, location, scale):
    """Preserve every available measurement and internal empty hour in float64."""
    if not stays:
        raise ValueError("Cannot pack an empty cohort")
    lengths = [
        int(array(row["hours"])[-1] - array(row["hours"])[0] + 1) for row in stays
    ]
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    features = list(schema["features"])
    values = np.full((offsets[-1], len(features)), np.nan, dtype=np.float64)
    hours = np.empty(offsets[-1], dtype=np.int64)
    demos = np.empty((len(stays), len(schema["demographics"])), dtype=np.int64)
    for index, stay in enumerate(stays):
        source_hours = array(stay["hours"])
        raw = array(stay["values"]).astype(np.float64)
        if raw.shape != (len(source_hours), len(features)) or np.isinf(raw).any():
            raise ValueError("Invalid physiological value axes")
        if (source_hours < 0).any() or np.any(np.diff(source_hours) <= 0):
            raise ValueError("Stay hours must be unique, increasing and nonnegative")
        start, end = offsets[index : index + 2]
        hours[start:end] = np.arange(source_hours[0], source_hours[-1] + 1)
        positions = start + source_hours - source_hours[0]
        values[positions] = scale_values(raw, location, scale)
        for column, (name, spec) in enumerate(schema["demographics"].items()):
            category = category_index(stay["demographics"][name], spec, "clip")
            demos[index, column] = category
    centers = np.asarray(
        [6 if s["centre"] == "__null__" else CENTRE_MAP[s["centre"]] for s in stays],
        dtype=np.int64,
    )
    return dict(
        values=values,
        hours=hours,
        offsets=offsets,
        demographics=demos,
        centers=centers,
        labels=np.asarray([int(s["death_icu"]) for s in stays]),
        splits=np.asarray([s["split"] for s in stays]),
        keys=[f"{center}_{int(s['stay_id'])}" for center, s in zip(centers, stays)],
    )


def save_pack(path, packed):
    """Publish uncompressed arrays so concurrent shards can share file-backed pages."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".joblib.partial")
    joblib.dump(packed, temporary, compress=0)
    temporary.replace(path)


def load_pack(path):
    return joblib.load(path, mmap_mode="r")


def trajectory(packed, index, keep=None):
    start, end = packed["offsets"][index : index + 2]
    rows = np.arange(start, end) if keep is None else start + np.asarray(keep)
    return dict(
        values=packed["values"][rows],
        hours=packed["hours"][rows],
        demographics=packed["demographics"][index],
    )


def cell(packed, index, conditioned):
    return (
        int(packed["labels"][index]),
        int(packed["centers"][index]) if conditioned else -1,
    )


def pools(packed, indices, conditioned, cap, seed):
    """Select identical exact candidate indices for every numerical metric."""
    if cap < 1:
        raise ValueError("Positive candidate budget required")
    groups = {}
    for index in indices:
        groups.setdefault(cell(packed, index, conditioned), []).append(int(index))
    generator = np.random.default_rng(seed)
    result = {}
    for group in sorted(groups):
        candidates = np.asarray(groups[group], dtype=np.int64)
        result[group] = (
            generator.choice(candidates, cap, replace=False)
            if len(candidates) > cap
            else candidates
        )
    return result


def nearest(queries, query_indices, candidates, candidate_indices, cfg):
    """Validate computational bounds before invoking a parallel exact pool search."""
    if not len(query_indices) or not len(candidate_indices):
        raise ValueError("Nonempty query and candidate sets required")
    query_lengths = np.diff(queries["offsets"])[query_indices]
    candidate_lengths = np.diff(candidates["offsets"])[candidate_indices]
    if int(query_lengths.max()) * int(candidate_lengths.max()) > cfg.max_pair_cells:
        raise ValueError(
            "Pair exceeds explicit dynamic-programming cell budget; no truncation"
        )
    distances, indices = nearest_packed(
        queries["values"],
        queries["hours"],
        queries["offsets"],
        queries["demographics"],
        np.asarray(query_indices, dtype=np.int64),
        candidates["values"],
        candidates["hours"],
        candidates["offsets"],
        candidates["demographics"],
        np.asarray(candidate_indices, dtype=np.int64),
        cfg.time_weight,
    )
    if not np.isfinite(distances).all() or (indices < 0).any():
        raise ValueError("Numerical candidate search did not cover every query")
    return distances, indices
