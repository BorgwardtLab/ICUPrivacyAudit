"""Reuse completed clinical discrete cohorts for continuous utility only."""

from functools import partial
import json
from pathlib import Path

import hydra
import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf

from scripts import clinical_pipeline_tasks as pipeline
from syntheticicu.clinical_sources import load_clinical_source
from syntheticicu.clinical_utility import clinical_metadata
from syntheticicu.native_utility_replay import NativeUtilityReplay
from syntheticicu.native_utility_source import add_missing_reference_stays
from syntheticicu.tasks_numeric import active_tasks, _identity


class PreflightComplete(Exception):
    """Stop before classifier extraction after verifying the frozen reference."""


def verify_saved_reference(module, cfg, task):
    import pandas as pd

    path = Path(cfg.reference_utility_root) / cfg.generation_mode / task["name"]
    performance, splits = joblib.load(path / "results.joblib", mmap_mode="r")
    result = performance["all"]
    for partition in ["train", "test"]:
        indices = getattr(module, f"{partition}_indices")
        dataset = module.full_dataset
        real = np.asarray(dataset.stay_sources) != "syn"
        indices = np.asarray(indices)[real[indices]]
        expected = result[f"full_{partition}_labels"].reset_index()
        expected = expected.loc[expected.ids < len(module.data.bundle["stays"])]
        columns = ["ids", "times", "onset"]
        actual = pd.DataFrame(
            {name: getattr(dataset, name).numpy()[indices] for name in columns}
        )
        for frame in [expected, actual]:
            frame.sort_values(columns, inplace=True)
            frame.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(
            expected[columns], actual, check_dtype=False, check_exact=True
        )
    print("Reference saved ROC baseline:", path, flush=True)


@hydra.main(
    version_base="1.3", config_path="../config", config_name="clinical_native_replay"
)
def main(cfg: DictConfig):
    task = next(t for t in active_tasks() if t["index"] == cfg.task_index)
    original_factory = pipeline.task_factory
    reports = []
    reference_modules = []

    def factory(data, metadata, task_cfg, current_task):
        reference_cfg = OmegaConf.load("config/clinical_source/discrete_shuffled.yaml")
        reference = load_clinical_source(
            reference_cfg, cfg.generation_mode, metadata, True
        )
        data.bundle = dict(data.bundle, stays=list(data.bundle["stays"]))
        recovered = add_missing_reference_stays(
            data, reference, cfg.recovery_core_path, cfg.metadata_path
        )
        path = Path(cfg.output_dir) / task["name"]
        path.mkdir(parents=True, exist_ok=True)
        (path / "evaluation_only_recovery.json").write_text(
            json.dumps(recovered, indent=2)
        )
        reference_aliases = {
            _identity(stay): index
            for index, stay in enumerate(
                reference.bundle["stays"] + reference.synthetic
            )
        }
        data.utility_aliases = {
            _identity(stay): reference_aliases[_identity(stay)]
            for stay in data.bundle["stays"]
        }
        offset = len(reference.bundle["stays"])
        data.utility_aliases.update(
            {
                _identity(stay): offset + index
                for index, stay in enumerate(data.synthetic)
            }
        )
        constructor = partial(
            NativeUtilityReplay,
            data,
            reference,
            metadata.endpoint_hours.to_dict(),
            cfg.hours_max,
            reports,
        )

        def construct(**options):
            module = constructor(**options)
            setup = module.setup

            def checked_setup(*args, **kwargs):
                setup(*args, **kwargs)
                reference_modules.append(module.native_reference_module)
                if task["kind"] == "classifier":
                    verify_saved_reference(module.native_reference_module, cfg, task)
                if cfg.preflight:
                    raise PreflightComplete()

            module.setup = checked_setup
            return module

        return construct

    pipeline.task_factory = factory
    try:
        pipeline.main.__wrapped__(cfg)
    except PreflightComplete:
        print(
            "Native discrete reference and continuous window checks passed", flush=True
        )
    finally:
        pipeline.task_factory = original_factory
    destination = Path(cfg.output_dir) / task["name"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "native_reference_checks.json").write_text(
        json.dumps(reports, indent=2)
    )


if __name__ == "__main__":
    main()
