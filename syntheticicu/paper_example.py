"""Entirely artificial inputs and a CPU demonstration of the paper contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

from syntheticicu.data_numeric import prepare_numerical_data
from syntheticicu.workflow_numeric import train_numerical, generate_numerical
from syntheticicu.clinical_utility import ClinicalWindows
from syntheticicu.value_distances import pair_distances, recovery_summary


def make_data(destination, specification):
    """Materialize small numerical tables and a direct FIDDLE interchange fixture."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(specification["seed"])
    core, meta, values, times, labels, centers, statics = [], [], [], [], [], [], []
    for index in range(specification["stays"]):
        center = specification["centers"][index % len(specification["centers"])]
        outcome = (index // len(specification["centers"])) % 2
        row = dict(
            centre=center,
            stay_id=1000 + index,
            patient_uid=2000 + index,
            death_icu=outcome,
            los_icu=specification["hours_per_stay"] / 24,
            female_static=index % 2,
            age_static=40 + index,
            height_static=160 + index % 20,
            weight_static=60 + index % 30,
        )
        meta.append(row)
        measured = []
        for hour in range(specification["hours_per_stay"]):
            hr = 75 + 5 * outcome + generator.normal(0, 3)
            temp = 36.5 + 0.2 * outcome + generator.normal(0, 0.1)
            if hour % 7 == 0:
                temp = np.nan
            if hour % 13 == 0:
                hr = np.nan
            core.append(
                dict(
                    centre=center,
                    stay_id=1000 + index,
                    stay_time=float(hour),
                    hr_raw=hr,
                    temp_raw=temp,
                )
            )
            measured.append([hr, temp])
        raw = np.asarray(measured)
        binned = np.column_stack(
            [
                np.clip(np.floor((raw[:, 0] - 60) / 7), 0, 4),
                np.clip(np.floor((raw[:, 1] - 36) / 0.3), 0, 4),
            ]
        )
        binned[np.isnan(raw)] = 5
        values.append(torch.tensor(binned, dtype=torch.int32))
        times.append(torch.arange(len(raw), dtype=torch.float32))
        labels.append([1000 + index, outcome, 0])
        centers.append(center)
        statics.append([index % 2, index % 5, index % 5, index % 5])
    pd.DataFrame(core).to_parquet(destination / "core.parquet", index=False)
    pd.DataFrame(meta).to_parquet(destination / "meta.parquet", index=False)
    dataset = SimpleNamespace(
        features=specification["features"],
        num_bins=5,
        X_value_ids=values,
        times=times,
        y=torch.tensor(labels),
        stay_sources=centers,
        statics=torch.tensor(statics, dtype=torch.int32),
        processed_statics=torch.tensor(statics, dtype=torch.int32),
        statics_set=["female_static", "age_static", "height_static", "weight_static"],
        data_table=pd.DataFrame(meta),
        lengths=torch.tensor([len(t) for t in times]),
        missing_mask=[v == 5 for v in values],
        X_feature_ids=[torch.arange(2).repeat(len(t), 1) for t in times],
        X_FIDDLE_bins={
            "hr_raw": [67, 74, 81, 88],
            "temp_raw": [36.3, 36.6, 36.9, 37.2],
        },
        S_FIDDLE_bins={},
        static_missing_token_id=5,
        static_pad_token_id=6,
    )
    joblib.dump(dataset, destination / "discrete.pkl")
    return destination


def run_example(cfg):
    """Train two CPU updates, generate a tiny release and validate real clinical windows."""
    root = Path(cfg.output_dir)
    specification = json.loads(Path(cfg.specification).read_text())
    inputs = make_data(root / "inputs", specification)
    cfg.data.core_path = str(inputs / "core.parquet")
    cfg.data.meta_path = str(inputs / "meta.parquet")
    bundle = prepare_numerical_data(cfg.data)
    torch.save(bundle, root / "prepared.pt")
    train_numerical(cfg)
    release = generate_numerical(cfg)
    metadata = pd.read_parquet(inputs / "meta.parquet")
    endpoints = {(r.centre, r.stay_id): r.los_icu * 24 for r in metadata.itertuples()}
    windows = ClinicalWindows(
        bundle["stays"],
        bundle["schema"],
        lookback=24,
        syn_lookback=0,
        window_size=4,
        feature_set="all",
        min_length=6,
        missingness_threshold="vitals",
        endpoints=endpoints,
        hours_max=9000,
    )
    if not np.all(windows.reference_hours.numpy() - windows.times.numpy() >= 24):
        raise ValueError("Clinical window violates the 24-hour gap")
    record = dict(
        values=np.array([[1.0], [2.0], [3.0]]),
        hours=np.arange(3),
        demographics=np.array([0, 1, 2, 3]),
    )
    distance = pair_distances(record, record)[0]
    summary = dict(
        artificial_data=True,
        training_updates=int(cfg.train.max_steps),
        generated_stays=len(pd.read_parquet(release / "meta.parquet")),
        real_clinical_windows=len(windows),
        literal_dtw_l1=float(distance),
        recovery=recovery_summary(np.array([distance]), np.array([1.0])),
    )
    (root / "example_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
