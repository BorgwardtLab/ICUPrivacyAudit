"""Run the active utility matrix with clinical real cutoffs and uncut synthesis."""

from functools import partial
import hashlib
import json
from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from syntheticicu.clinical_sources import load_clinical_source
from syntheticicu.clinical_utility import (
    ClinicalTaskDatamodule,
    ClinicalWindows,
    clinical_metadata,
)
from syntheticicu.tasks_numeric import active_tasks, run_classifier_task


def file_hash(path):
    """Hash the small metadata input in bounded chunks for result provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(partial(stream.read, 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_factory(data, metadata, cfg, task):
    """Bind the paper's clinical windows to validated ICU endpoints."""
    endpoints = metadata.endpoint_hours.to_dict()
    for stay in data.bundle["stays"]:
        key = (stay["centre"], stay["stay_id"])
        if key not in metadata.index:
            raise ValueError("Real stay has no clinical metadata")
        if bool(metadata.loc[key, "death_icu"]) != bool(stay["death_icu"]):
            raise ValueError("Prepared and current clinical mortality labels disagree")
    return partial(ClinicalTaskDatamodule, data, endpoints, cfg.hours_max)


def input_fingerprints(cfg, require_release):
    """Prevent an old completion marker from hiding replacement input artifacts."""
    source = cfg.source
    if source.representation == "numerical":
        root = Path(source.roots[cfg.generation_mode])
        paths = [root / "prepared.pt"]
        if require_release:
            paths += [
                root / "generated" / name
                for name in [
                    "generation.json",
                    "tokens.pt",
                    "core.parquet",
                    "meta.parquet",
                ]
            ]
    elif source.representation == "discrete":
        paths = [Path(source.real_path)]
        if require_release:
            paths.append(Path(source.synthetic_paths[cfg.generation_mode]))
    else:
        raise ValueError("Unknown clinical source representation")
    if "reference_utility_root" in cfg:
        task = active_tasks()[cfg.task_index]
        paths.append(
            Path(cfg.reference_utility_root)
            / cfg.generation_mode
            / task["name"]
            / "results.joblib"
        )
    return {
        str(path.resolve()): {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    }


def result_contract(cfg, task, require_release):
    """Record clocks, source revisions and cohort policy alongside every result."""
    sources = [
        "syntheticicu/clinical_utility.py",
        "syntheticicu/clinical_sources.py",
        "syntheticicu/tasks_numeric.py",
        "syntheticicu/evaluation_numeric.py",
        "syntheticicu/data_classes.py",
        "scripts/clinical_pipeline_tasks.py",
        "scripts/train_classifiers.py",
    ]
    return {
        "configuration": OmegaConf.to_container(cfg, resolve=True),
        "metadata_sha256": file_hash(cfg.metadata_path),
        "task": task,
        "source_sha256": {name: file_hash(name) for name in sources},
        "real_reference": "los_icu * 24; HiRID observation-derived proxy",
        "window_time": "right edge of completed hourly bucket; real gap >=24h",
        "synthetic_lookback_hours": 0,
        "input_fingerprints": input_fingerprints(cfg, require_release),
        "cohort_policy": (
            "matched native discrete selection"
            if "reference_utility_root" in cfg
            else "native clinical selection"
        ),
    }


def write_results(result, modules, destination, marker, contract):
    """Mark completion only after storing outputs and population exclusions."""
    (destination / "windows.json").write_text(
        json.dumps(
            [module.full_dataset.clinical_report for module in modules], indent=2
        )
        + "\n"
    )
    for index, module in enumerate(modules):
        if hasattr(module, "replay_rows"):
            pd.DataFrame(module.replay_rows).to_parquet(
                destination / f"replayed_splits_{index}.parquet", index=False
            )
            (destination / f"split_report_{index}.json").write_text(
                json.dumps(module.split_report, indent=2) + "\n"
            )
    temporary = destination / "results.joblib.partial"
    joblib.dump(result, temporary)
    temporary.replace(destination / "results.joblib")
    temporary = marker.with_suffix(".json.partial")
    temporary.write_text(json.dumps(contract, indent=2) + "\n")
    temporary.replace(marker)
    print(f"Completed clinical task: {marker}", flush=True)


def validate_source(data, metadata, cfg, contract):
    """Exercise real and synthetic clocks before scheduling expensive fits."""
    groups = {}
    for stay in data.bundle["stays"]:
        key = (stay["centre"], stay["stay_id"])
        if metadata.loc[key, "endpoint_hours"] >= 28:
            groups.setdefault(stay["centre"], []).append(stay)
    sample = []
    for group in groups.values():
        ordered = sorted(group, key=lambda stay: len(stay["hours"]))
        sample += ordered[:3] + ordered[-3:]
    synthetic = sorted(data.synthetic, key=lambda stay: len(stay["classifier_hours"]))
    sample += synthetic[:3] + synthetic[-3:]
    unique = {(s["split"], s["centre"], s["stay_id"]): s for s in sample}
    dataset = ClinicalWindows(
        list(unique.values()),
        data.bundle["schema"],
        lookback=24,
        syn_lookback=0,
        window_size=4,
        feature_set="all",
        min_length=6,
        endpoints=metadata.endpoint_hours.to_dict(),
        hours_max=cfg.hours_max,
    )
    real = dataset.stay_sources.to_numpy() != "syn"
    gap = (dataset.reference_hours - dataset.times).numpy()
    if not real.any() or (gap[real] < 24).any():
        raise ValueError("Preflight found invalid real clinical windows")
    report = {
        "contract": contract,
        "real_stays": len(data.bundle["stays"]),
        "synthetic_stays": len(data.synthetic),
        "synthetic_positives": sum(int(s["death_icu"]) for s in data.synthetic),
        "sample_windows": dataset.clinical_report,
        "sample_real_gap_hours_min": float(gap[real].min()),
    }
    destination = Path(cfg.output_dir) / "preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


@hydra.main(
    version_base="1.3", config_path="../config", config_name="clinical_pipeline"
)
def main(cfg: DictConfig):
    """Write new utility outputs; never edit metadata, releases or prepared bundles."""
    repository = Path(hydra.utils.get_original_cwd()).resolve()
    root = Path(cfg.output_dir).resolve()
    if not root.is_relative_to(repository):
        raise ValueError("Clinical task outputs must stay inside this repository")
    if cfg.generation_mode not in {"null", "conditioned"}:
        raise ValueError("generation_mode must be null or conditioned")
    if cfg.real_time_origin != "icu_admission":
        raise ValueError("Clinical grids require ICU-admission-relative source hours")
    if cfg.action not in {"run", "validate"}:
        raise ValueError("action must be run or validate")
    task = next((t for t in active_tasks() if t["index"] == cfg.task_index), None)
    if task is None:
        raise ValueError("Inactive clinical utility task index")
    overrides = OmegaConf.to_container(cfg.classifier_overrides, resolve=True)
    for name, value in [("lookback", 24), ("syn_lookback", 0)]:
        if name in overrides and overrides[name] != value:
            raise ValueError("Clinical task cutoffs are fixed at real=24, synthetic=0")
        overrides[name] = value
    destination = root / task["name"]
    marker = destination / "complete.json"
    require_release = task["kind"] != "classifier" or task["mix"] > 0 or task["augment"]
    contract = result_contract(cfg, task, require_release)
    if cfg.action == "run" and marker.exists():
        if (
            json.loads(marker.read_text()) != contract
            or not (destination / "results.joblib").is_file()
        ):
            raise ValueError(
                "Existing result has different inputs; choose a new output_dir"
            )
        print(f"Already complete: {marker}", flush=True)
        return
    metadata = clinical_metadata(cfg.metadata_path, cfg.hours_max)
    data = load_clinical_source(
        cfg.source, cfg.generation_mode, metadata, require_release
    )
    if require_release and (
        len(data.synthetic) != cfg.release_size
        or sum(int(s["death_icu"]) for s in data.synthetic) != cfg.release_positives
    ):
        raise ValueError(
            "Clinical comparison requires 12000 synthetic stays and 840 positives"
        )
    factory = task_factory(data, metadata, cfg, task)
    if cfg.action == "validate":
        validate_source(data, metadata, cfg, contract)
        return
    modules = []

    def capture_factory(**options):
        module = factory(**options)
        modules.append(module)
        return module

    destination.mkdir(parents=True, exist_ok=True)
    result = run_classifier_task(
        data,
        task,
        destination,
        cfg.seed,
        cfg.threads,
        overrides,
        datamodule_factory=capture_factory,
    )
    write_results(result, modules, destination, marker, contract)


if __name__ == "__main__":
    main()
