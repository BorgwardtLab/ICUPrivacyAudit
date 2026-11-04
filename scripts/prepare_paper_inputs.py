"""Recover exact generator stay assignments from a recorded training configuration."""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from syntheticicu import resolvers  # Registers the training configuration resolvers.


@hydra.main(version_base="1.3", config_path="../config", config_name="paper_inputs")
def main(cfg: DictConfig):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    recorded = OmegaConf.load(cfg.hparams).cfg
    if cfg.dataset_path is not None:
        recorded.datamodule.data_path = cfg.dataset_path
    module = hydra.utils.instantiate(recorded.datamodule)
    source = module.base_icu_dataset
    identifiers = np.asarray(source.y)[:, 0]
    if identifiers.dtype.kind not in {"i", "u"}:
        raise ValueError("Exact integer source identifiers are required")
    rows = []
    for partition in ["train", "val", "test"]:
        rows.extend(
            dict(
                centre=str(source.stay_sources[i]).lower(),
                stay_id=int(identifiers[i]),
                split=partition,
            )
            for i in getattr(module, f"{partition}_indices")
        )
    frame = pd.DataFrame(rows).drop_duplicates()
    if frame.duplicated(["centre", "stay_id"]).any():
        raise ValueError("A reference stay crosses generator partitions")
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["centre", "stay_id"]).to_parquet(
        root / "generator_splits.parquet", index=False
    )
    (root / "split_provenance.json").write_text(
        json.dumps(
            dict(
                hparams=str(cfg.hparams),
                counts=frame.split.value_counts().to_dict(),
                unit="exact center/stay",
                patient_disjointness="not established by this manifest",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
