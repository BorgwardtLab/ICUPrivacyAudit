"""Summarize metadata for the exact generator source cohort without exporting keys."""

from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from syntheticicu.data_classes import ICUDataset


def summary(frame, center):
    result = dict(
        center=center,
        stays=len(frame),
        patients=frame.patient_uid.nunique(),
        female_percent=100 * frame.female_static.mean(),
        mortality_percent=100 * frame.death_icu.mean(),
    )
    quantities = {
        "age": frame.age_static,
        "los_days": frame.los_icu,
        "bmi": frame.weight_static / (frame.height_static / 100) ** 2,
    }
    for name, values in quantities.items():
        for suffix, percentile in [("median", 0.5), ("q25", 0.25), ("q75", 0.75)]:
            result[f"{name}_{suffix}"] = values.quantile(percentile)
    return result


@hydra.main(version_base="1.3", config_path="../config", config_name="paper_cohort")
def main(cfg: DictConfig):
    ICUDataset.__new__(ICUDataset)._register_pickle_aliases()
    dataset = joblib.load(cfg.dataset_path)
    identifiers = np.asarray(dataset.y)[:, 0]
    if identifiers.dtype.kind not in {"i", "u"}:
        raise ValueError("Exact integer source stay identifiers are required")
    keys = pd.DataFrame(dict(centre=dataset.stay_sources, stay_id=identifiers))
    if keys.duplicated().any():
        raise ValueError("Generator source contains duplicate stays")
    metadata = keys.merge(
        pd.read_parquet(cfg.metadata_path),
        on=["centre", "stay_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not metadata._merge.eq("both").all() or metadata.los_icu.isna().any():
        raise ValueError("Incomplete cohort metadata or clinical LOS")
    rows = [summary(group, center) for center, group in metadata.groupby("centre")]
    rows.append(summary(metadata, "all"))
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "cohort_summary.csv", index=False)


if __name__ == "__main__":
    main()
