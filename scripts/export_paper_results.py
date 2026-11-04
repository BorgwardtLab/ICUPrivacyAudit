"""Read saved paper analyses, validate metrics and export no original identifiers."""

import hashlib
import json
from pathlib import Path

import hydra
import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from syntheticicu.paper_results import (
    export_utility,
    export_attack,
    rarity_and_groups,
    closer_to_training,
)


@hydra.main(version_base="1.3", config_path="../config", config_name="paper_results")
def main(cfg: DictConfig):
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    availability = []
    provenance = []
    for section in [
        "utility",
        "discrete_audits",
        "numerical_audits",
        "canaries",
        "replication",
        "value_corpus",
        "recovery_extensions",
    ]:
        for item in cfg[section]:
            path = Path(item.path)
            if not (path.is_dir() if section == "value_corpus" else path.is_file()):
                availability.append(
                    dict(
                        section=section,
                        name=item.name,
                        status="pending",
                        path=str(path),
                    )
                )
                continue
            target = root / section / item.name
            target.mkdir(parents=True, exist_ok=True)
            provenance.append(
                dict(
                    section=section,
                    name=item.name,
                    path=str(path),
                    bytes=path.stat().st_size,
                    mtime_ns=path.stat().st_mtime_ns,
                )
            )
            metadata = {
                key: value
                for key, value in OmegaConf.to_container(item, resolve=True).items()
                if key not in {"name", "path"}
            }
            availability.append(
                dict(section=section, name=item.name, status="complete")
            )
            if section == "value_corpus":
                frame = pd.read_csv(
                    path / "membership_metrics.csv", keep_default_na=False
                )
                frame = frame.loc[
                    (frame.metric == "dtw_l1") & (frame.statistic == "raw")
                ]
                if len(frame) != 1:
                    raise ValueError("Expected one raw DTW-L1 membership result")
                for row in frame.to_dict("records"):
                    reports.append(
                        {**row, **metadata, "section": "corpus", "name": item.name}
                    )
                recovery = pd.read_csv(path / "recovery.csv")
                recovery.loc[recovery.metric == "dtw_l1"].to_csv(
                    target / "recovery.csv", index=False
                )
                npz = np.load(path / "membership_rocs.npz", allow_pickle=False)
                np.savez_compressed(
                    target / "C.npz",
                    fpr=npz["dtw_l1_raw_fpr"],
                    tpr=npz["dtw_l1_raw_tpr"],
                )
                continue
            if section == "recovery_extensions":
                value = joblib.load(path)
                if not value["check_passed"]:
                    raise ValueError("Unverified historical recovery extension")
                value["canary_merged"].to_csv(target / "recovery.csv", index=False)
                continue
            if section == "utility":
                metrics = export_utility(path, target, cfg.utility_bootstraps, cfg.seed)
                reports.append(
                    dict(section=section, name=item.name, **metadata, **metrics)
                )
            elif section == "canaries":
                frame = pd.read_csv(path, keep_default_na=False)
                frame.to_csv(target / "curve.csv", index=False)
                sibling = path.with_name("curve_rocs.npz")
                if sibling.is_file():
                    (target / "curve_rocs.npz").write_bytes(sibling.read_bytes())
            elif section == "replication":
                value = joblib.load(path)
                arrays = value.get("arrays", value)
                train = arrays.get("generated_to_train")
                holdout = arrays.get(
                    "generated_to_holdout", arrays.get("generated_to_test")
                )
                if train is None or holdout is None:
                    raise ValueError("Missing genuine replication distances")
                if getattr(train, "ndim", 1) == 2:
                    metrics = value.get("metrics")
                    if metrics is None:
                        metrics = [
                            row["metric"] for row in value.get("comparisons", [])
                        ]
                    if not metrics or train.shape[1] != len(metrics):
                        raise ValueError("Replication metric columns are not declared")
                    column = list(metrics).index("dtw_l1")
                    train, holdout = train[:, column], holdout[:, column]
                reports.append(
                    dict(
                        section=section,
                        name=item.name,
                        **metadata,
                        **closer_to_training(train, holdout),
                    )
                )
            elif section == "numerical_audits":
                payload = joblib.load(path)
                audit = payload.get("result", payload)
                audit = audit.get("membership", audit)
                for statistic, summary in audit["summaries"].items():
                    metrics = export_attack(
                        target / f"{statistic}.npz",
                        audit["train_scores"][statistic],
                        audit["holdout_scores"][statistic],
                        summary,
                    )
                    reports.append(
                        dict(
                            section=section,
                            name=item.name,
                            statistic=statistic,
                            **metadata,
                            **metrics,
                        )
                    )
            else:
                audit = joblib.load(path)
                if audit["smoke"]:
                    raise ValueError("Paper exports cannot use smoke audit populations")
                provenance[-1]["population_policy"] = audit["sizes"]
                for statistic, summary in audit["membership_calibrated"].items():
                    metrics = export_attack(
                        target / f"A_{statistic}.npz",
                        audit["per_patient"]["attack_a"]["train"][statistic],
                        audit["per_patient"]["attack_a"]["holdout"][statistic],
                        summary,
                    )
                    reports.append(
                        dict(
                            section="weights",
                            name=item.name,
                            statistic=statistic,
                            **metadata,
                            **metrics,
                        )
                    )
                for mode, value in audit["generation_mia"].items():
                    metrics = export_attack(
                        target / f"C_{mode}.npz",
                        value["train_patient_scores"],
                        value["holdout_patient_scores"],
                        value["summary"],
                        negate=True,
                    )
                    reports.append(
                        dict(
                            section="corpus",
                            name=item.name,
                            mode=mode,
                            n_gen_used=value["n_gen_used"],
                            **metadata,
                            **metrics,
                        )
                    )
                rarity, sizes = rarity_and_groups(audit)
                rarity.to_csv(target / "rarity.csv", index=False)
                sizes.sample(frac=1, random_state=cfg.seed).to_csv(
                    target / "demographic_group_sizes.csv", index=False
                )
                audit["canary"].to_csv(target / "recovery.csv", index=False)
    if not reports:
        raise ValueError(
            "No measured analyses found; populate the paper_results manifest"
        )
    pd.DataFrame(reports).to_csv(root / "metrics.csv", index=False)
    (root / "availability.json").write_text(json.dumps(availability, indent=2))
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (root / "configuration.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
    )
    checksum = root / "SHA256SUMS.txt"
    checksum.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != checksum
        )
    )
    print(f"Exported {len(reports)} verified summaries to {root}", flush=True)


if __name__ == "__main__":
    main()
