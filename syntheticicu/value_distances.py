"""Full-precision physiological distances; independent of generator training code."""

import hashlib

from numba import njit, prange
import numpy as np


DISTANCE_NAMES = ("dtw_l1",)


def robust_scaling(values):
    """Fit training-only location and nonzero scale without clipping observations."""
    if values.ndim != 2 or np.isinf(values).any():
        raise ValueError("Expected a finite-or-missing measurement matrix")
    quantiles = np.nanquantile(values, [0.25, 0.5, 0.75], axis=0)
    location, scale = quantiles[1], quantiles[2] - quantiles[0]
    extent = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    scale = np.where(scale > 0, scale, extent)
    if (
        not np.isfinite(location).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Each feature needs observed, nonconstant training values")
    return location, scale


def scale_values(values, location, scale):
    """Use an invertible transformation in float64; missing values remain missing."""
    if (
        values.ndim != 2
        or values.shape[1] != len(location)
        or len(location) != len(scale)
    ):
        raise ValueError("Feature/scaling axes differ")
    if np.isinf(values).any() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid numerical measurements or scales")
    return np.ascontiguousarray(
        np.arcsinh((values.astype(np.float64) - location) / scale)
    )


def deletion_indices(length, fraction, identity, seed):
    """Delete nested subsets of hours, retaining original timestamps and values."""
    if length < 1 or not 0 <= fraction < 1:
        raise ValueError("Positive length and deletion fraction in [0,1) required")
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    remove = min(length - 1, int(np.floor(length * fraction + 1e-10)))
    return np.sort(generator.permutation(length)[remove:])


def validate_trajectory(record):
    values, hours, demographics = [
        record[name] for name in ["values", "hours", "demographics"]
    ]
    if values.ndim != 2 or len(values) < 1 or len(values) != len(hours):
        raise ValueError("Nonempty aligned measurement/hour arrays required")
    if (
        np.isinf(values).any()
        or not np.isfinite(hours).all()
        or np.any(hours < 0)
        or np.any(hours != np.floor(hours))
        or np.any(np.diff(hours) <= 0)
    ):
        raise ValueError("Hours must increase strictly; infinite values are invalid")
    if demographics.ndim != 1 or not np.isfinite(demographics).all():
        raise ValueError("Demographic categories must be a finite vector")


def pair_distances(left, right, time_weight=0.1, max_pair_cells=100000000):
    """Return the prespecified DTW-L1 distance without fitting to membership labels."""
    validate_trajectory(left)
    validate_trajectory(right)
    if left["values"].shape[1] != right["values"].shape[1]:
        raise ValueError("Feature axes differ")
    if left["demographics"].shape != right["demographics"].shape:
        raise ValueError("Demographic axes differ")
    if len(left["hours"]) * len(right["hours"]) > max_pair_cells:
        raise ValueError("Pair exceeds explicit dynamic-programming cell budget")
    if time_weight <= 0:
        raise ValueError("Positive clock weight required")
    return _pair(
        left["values"],
        left["hours"],
        left["demographics"],
        right["values"],
        right["hours"],
        right["demographics"],
        time_weight,
    )


@njit(cache=True, parallel=True)
def nearest_packed(
    values_q,
    hours_q,
    offsets_q,
    demo_q,
    queries,
    values_c,
    hours_c,
    offsets_c,
    demo_c,
    candidates,
    time_weight,
):
    """Search the entire supplied pool; no approximate pruning or time truncation."""
    distances = np.full((len(queries), 1), np.inf)
    nearest = np.full((len(queries), 1), -1, dtype=np.int64)
    for index in prange(len(queries)):
        query = queries[index]
        start, end = offsets_q[query], offsets_q[query + 1]
        for candidate in candidates:
            other_start, other_end = offsets_c[candidate], offsets_c[candidate + 1]
            result = _pair(
                values_q[start:end],
                hours_q[start:end],
                demo_q[query],
                values_c[other_start:other_end],
                hours_c[other_start:other_end],
                demo_c[candidate],
                time_weight,
            )
            for metric in range(1):
                if result[metric] < distances[index, metric]:
                    distances[index, metric] = result[metric]
                    nearest[index, metric] = candidate
    return distances, nearest


def recovery_summary(planted, background):
    """Strict rank-1 wins with Wilson intervals; ties are explicitly non-wins."""
    planted, background = np.asarray(planted), np.asarray(background)
    if planted.shape != background.shape or not len(planted):
        raise ValueError("Nonempty paired recovery distances required")
    if not np.isfinite(planted).all() or not np.isfinite(background).all():
        raise ValueError("Recovery distances must be finite")
    wins, count = int(np.sum(planted < background)), len(planted)
    rate, z = wins / count, 1.959963984540054
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half = (
        z
        * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
        / denominator
    )
    return dict(
        n=count,
        wins=wins,
        rank1_rate=rate,
        ci_low=max(0.0, center - half),
        ci_high=min(1.0, center + half),
        ties=int(np.sum(planted == background)),
        median_margin=float(np.median(background - planted)),
    )


@njit(cache=True)
def _local_l1(left, right):
    total, count = 0.0, 0
    for feature in range(len(left)):
        observed_left = np.isfinite(left[feature])
        observed_right = np.isfinite(right[feature])
        if observed_left or observed_right:
            count += 1
            total += (
                abs(left[feature] - right[feature])
                if observed_left and observed_right
                else 1.0
            )
    return total / count if count else 0.0


@njit(cache=True)
def _pair(left, left_hours, left_demo, right, right_hours, right_demo, time_weight):
    previous = np.full(len(right) + 1, np.inf)
    previous[0] = 0.0
    for row in range(len(left)):
        current = np.full(len(right) + 1, np.inf)
        for column in range(len(right)):
            clock = time_weight * abs(
                np.arcsinh(left_hours[row] / 24.0)
                - np.arcsinh(right_hours[column] / 24.0)
            )
            cost = _local_l1(left[row], right[column]) + clock
            current[column + 1] = cost + min(
                previous[column + 1], current[column], previous[column]
            )
        previous = current
    mismatch = float(np.sum(left_demo != right_demo))
    distance = (previous[-1] + mismatch) / (max(len(left), len(right)) + len(left_demo))
    return np.array([distance])
