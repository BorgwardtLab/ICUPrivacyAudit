"""Recover evaluation-only real stays omitted by generator cohort filtering."""

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from syntheticicu.tasks_numeric import _identity


def add_missing_reference_stays(
    data,
    reference,
    core_path="data/core.parquet",
    meta_path="data/metadata.parquet",
):
    """Read original observations; never modify the generator's prepared bundle."""
    available = {_identity(stay) for stay in data.bundle["stays"]}
    missing = [
        stay for stay in reference.bundle["stays"] if _identity(stay) not in available
    ]
    if not missing:
        return []
    features = list(data.bundle["schema"]["features"])
    keys = {(stay["centre"], stay["stay_id"]) for stay in missing}
    source = ds.dataset(core_path, format="parquet")
    selected = ds.field("stay_id").isin([key[1] for key in keys]) & (
        ds.field("stay_time") >= 0
    )
    frame = source.to_table(
        columns=["centre", "stay_id", "stay_time", *features], filter=selected
    ).to_pandas()
    frame = frame.loc[[key in keys for key in zip(frame.centre, frame.stay_id)]].copy()
    metadata = pd.read_parquet(
        meta_path, columns=["centre", "stay_id", "weight_static"]
    )
    frame = frame.merge(metadata, on=["centre", "stay_id"], validate="many_to_one")
    if ((frame.weight_static <= 0) & frame.urine_raw.notna()).any():
        raise ValueError("Recovered urine observation has invalid normalization weight")
    frame["urine_raw"] = frame.urine_raw / frame.weight_static
    frame["hour"] = np.floor(frame.stay_time).astype(np.int64)
    frame = (
        frame.groupby(["centre", "stay_id", "hour"], sort=True)[features]
        .mean()
        .reset_index()
    )
    recovered = []
    for stay in missing:
        selected = frame.loc[
            (frame.centre == stay["centre"]) & (frame.stay_id == stay["stay_id"])
        ]
        selected = selected.loc[selected[features].notna().any(axis=1)]
        values = selected[features].to_numpy(dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError("Infinite recovered clinical observations")
        data.bundle["stays"].append(
            dict(
                stay,
                split="not_audited",
                demographics={},
                hours=torch.as_tensor(selected.hour.to_numpy(), dtype=torch.long),
                values=torch.from_numpy(values),
            )
        )
        recovered.append(
            dict(
                centre=stay["centre"],
                stay_id=stay["stay_id"],
                observation_hours=len(selected),
            )
        )
    print(
        f"Recovered {len(recovered)} evaluation-only stays from raw observations",
        flush=True,
    )
    return recovered
