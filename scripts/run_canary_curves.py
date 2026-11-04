"""Full-scale fixed-order Attack A controls, isolated from the running pipeline."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from syntheticicu.data_classes import CENTRE_MAP
from syntheticicu.evaluation_numeric import load_evaluation_data
from syntheticicu.privacy_audit import _composite_key
from syntheticicu.privacy_audit_ext import (
    StaticsDecoder,
    run_calibrated_membership_audit,
    summarize_attack,
)
from syntheticicu.privacy_bridge_numeric import NumericalAuditModel
from syntheticicu.workflow_numeric import (
    data_signature,
    load_numerical_model,
    train_numerical,
)


@hydra.main(version_base="1.3", config_path="../config", config_name="canary_curves")
def main(cfg):
    """Dispatch only the bounded stages in the submitted canary graph."""
    configure_temporary_directory()
    root = Path(cfg.output_dir).resolve()
    if not root.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Canary outputs must stay inside the repository")
    if cfg.count != 500:
        raise ValueError("The reviewed canary count is 500")
    if cfg.factor not in cfg.factors or list(cfg.factors) != [5, 20, 50]:
        raise ValueError(
            "Expected the reviewed nonredundant three-point duplication curve"
        )
    root.mkdir(parents=True, exist_ok=True)
    actions = {
        "prepare": prepare,
        "continuous_prepare": continuous_prepare,
        "train": train,
        "score": score,
        "collect": collect,
    }
    actions[cfg.action](cfg, root)


def configure_temporary_directory():
    """Keep multiprocessing listener sockets within Linux's Unix-path limit."""
    identifier = os.environ.get("SLURM_JOB_ID", "local")
    if identifier != "local" and not identifier.isdecimal():
        raise ValueError("Expected an integer Slurm job ID")
    runtime_root = Path(
        os.environ.get(
            "SYNTHETICICU_RUNTIME_ROOT", str(Path(__file__).resolve().parents[1])
        )
    )
    if not runtime_root.is_absolute():
        raise ValueError("Runtime root must be an absolute path")
    directory = runtime_root / "outputs" / "t" / identifier
    representative = directory / "pymp-12345678" / "listener-12345678"
    if len(os.fsencode(str(representative))) > 107:
        raise ValueError("Multiprocessing temporary socket path exceeds Linux limit")
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(directory)
    # Imports may have cached the inherited, longer path before main starts.
    tempfile.tempdir = str(directory)
    print(f"Canary multiprocessing temporary directory: {directory}", flush=True)
    return directory


def historical_testbed():
    """Load the legacy CLI only in discrete stages: it registers global resolvers."""
    import scripts.run_testbed_full as testbed

    return testbed


def key(record):
    return _composite_key(record["stay_id"], CENTRE_MAP[str(record["centre"]).lower()])


def choose_canaries(records, count, seed):
    """Use the historical sorted exact stay-key selection, independent of chunks."""
    keys = sorted({key(row) for row in records if row["split"] == "train"})
    if count < 1 or count > len(keys):
        raise ValueError("Canary count must fit the unique training stays")
    chosen = np.random.default_rng(seed).choice(len(keys), count, replace=False)
    return sorted(keys[index] for index in chosen)


def expand_records(records, selected, factor):
    """Repeat training chunks without changing identity or validation/test records."""
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("Duplication factor must be a positive integer")
    selected = set(selected)
    training = {key(row) for row in records if row["split"] == "train"}
    if not selected or not selected <= training:
        raise ValueError("Every canary must be an existing training stay")
    expanded = list(records)
    for row in records:
        if row["split"] == "train" and key(row) in selected:
            expanded.extend([row] * (factor - 1))
    return expanded


def write_json(path, value):
    """Publish manifests atomically so retries never read incomplete JSON."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def check_or_write(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing canary contract differs: {path}")
    else:
        write_json(path, value)


def saved_config(root):
    return root / "discrete_hparams.yaml"


def prepare(cfg, root):
    """Verify the saved fixed-order model's partitions before any GPU allocation."""
    testbed = historical_testbed()
    source = Path(cfg.discrete_hparams)
    target = saved_config(root)
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError("Frozen reference hparams differ")
    target.write_bytes(source.read_bytes())
    reference = pd.read_parquet(cfg.reference_splits)
    rows = reference.to_dict("records")
    if reference.duplicated(["centre", "stay_id"]).any():
        raise ValueError("Reference split identities are not unique")
    selected = choose_canaries(rows, int(cfg.count), int(cfg.seed))
    loaded, values = testbed.load_cfg(
        str(target), str(root / "preflight"), world_size=2
    )
    if loaded.datamodule.shuffle_tokens_within_timepoint:
        raise ValueError("The reference must have within-hour shuffling disabled")
    if int(loaded.seed) != cfg.seed:
        raise ValueError("Reference seed/resolution differs from the reviewed curve")
    datamodule = hydra.utils.instantiate(loaded.datamodule)
    if datamodule.tokenizer.num_bins != 5:
        raise ValueError("Expected K=5 reference tokens")
    base = datamodule.base_icu_dataset
    identities = base.y[:, 0].tolist()
    sources = list(base.stay_sources)
    counts = {}
    for split, attribute in [("train", "train"), ("val", "val"), ("test", "test")]:
        actual = {
            _composite_key(identities[index], CENTRE_MAP[str(sources[index]).lower()])
            for index in getattr(datamodule, f"{attribute}_indices")
        }
        expected = {key(row) for row in rows if row["split"] == split}
        if actual != expected:
            raise ValueError(f"Saved discrete {split} cohort differs from reference")
        counts[split] = len(actual)
    check_or_write(root / "canaries.json", selected)
    check_or_write(
        root / "contract.json",
        {
            "unit": "exact_center_stay_all_chunks",
            "count": int(cfg.count),
            "seed": int(cfg.seed),
            "factors": list(cfg.factors),
            "cohort": counts,
            "hparams_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "reference": values,
            "shuffled": False,
        },
    )
    print("Verified fixed-order full-scale canary contract", counts, flush=True)


def continuous_prepare(cfg, root):
    """Expand only record references; share tensor storage and preserve raw stays."""
    selected = json.loads((root / "canaries.json").read_text())
    data = load_evaluation_data(cfg.continuous_root, require_release=False)
    original = data_signature(data.bundle)
    counts = json.loads((root / "contract.json").read_text())["cohort"]
    reference = pd.read_parquet(cfg.reference_splits)
    reference_rows = reference.to_dict("records")
    for split in ["train", "val", "test"]:
        actual = {key(row) for row in data.bundle["records"] if row["split"] == split}
        expected = {key(row) for row in reference_rows if row["split"] == split}
        if actual != expected or len(actual) != counts[split]:
            raise ValueError(f"Continuous {split} identities differ from discrete")
    for factor in cfg.factors:
        directory = point(root, "continuous", factor)
        directory.mkdir(parents=True, exist_ok=True)
        contract = {"base_signature": original, "factor": factor, "canaries": selected}
        check_or_write(directory / "injection.json", contract)
        if (directory / "prepared.pt").exists():
            continue
        bundle = dict(data.bundle)
        bundle["records"] = expand_records(
            data.bundle["records"], selected, int(factor)
        )
        bundle["preparation_contract"] = {
            **bundle["preparation_contract"],
            "canary_injection": contract,
        }
        bundle["signature"] = data_signature(bundle)
        temporary = directory / "prepared.pt.partial"
        torch.save(bundle, temporary)
        temporary.replace(directory / "prepared.pt")
        print(
            f"Prepared continuous x{factor}: {len(bundle['records'])} chunks",
            flush=True,
        )


def point(root, representation, factor):
    if representation not in {"discrete", "continuous"}:
        raise ValueError("Unknown representation")
    return root / representation / f"dup{factor}_n500"


def train(cfg, root):
    """Reuse each audited training implementation and checkpoint-selection rule."""
    directory = point(root, cfg.representation, cfg.factor)
    if cfg.representation == "discrete":
        manifest = directory / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text())["finished"]:
            return
        historical_testbed().train(
            argparse.Namespace(
                hparams=str(saved_config(root)),
                out=str(directory),
                dup=int(cfg.factor),
                n_canaries=int(cfg.count),
                data_path=None,
            )
        )
    else:
        configuration = hydra.compose(config_name="numerical_matched")
        configuration.output_dir = str(directory)
        # Injection is explicit in prepared.pt, bound into its data signature.
        # The generic UID-based injector must not repeat these stays a second time.
        if (
            configuration.train.n_canaries != 0
            or configuration.train.duplication_factor != 1
        ):
            raise ValueError("Do not combine UID injection with exact-stay injection")
        if configuration.train.shuffle_within_hour:
            raise ValueError("Continuous shuffling must remain disabled")
        if (directory / "model.pt").exists():
            return
        train_numerical(configuration)


def score(cfg, root):
    """Audit each original record once, including all raw and swapped statistics."""
    directory = point(root, cfg.representation, cfg.factor)
    selected = set(json.loads((root / "canaries.json").read_text()))
    if cfg.representation == "discrete":
        manifest = json.loads((directory / "manifest.json").read_text())
        if not manifest["finished"]:
            raise ValueError("Discrete training is incomplete")
        if set(json.loads((directory / "canaries.json").read_text())) != selected:
            raise ValueError(
                "Discrete injection does not match the shared canary selection"
            )
        testbed = historical_testbed()
        configuration, _ = testbed.load_cfg(str(saved_config(root)), str(directory), 1)
        model = (
            testbed.get_model_class(configuration)
            .load_from_checkpoint(
                manifest["best_model_path"],
                new_cfg=configuration,
                weights_only=False,
            )
            .to("cuda")
            .float()
            .eval()
        )
        save_scores(cfg, directory, model, selected, "categorical")
    else:
        data = load_evaluation_data(cfg.continuous_root, require_release=False)
        injection = json.loads((directory / "injection.json").read_text())
        if data_signature(data.bundle) != injection["base_signature"]:
            raise ValueError("Continuous source data changed after injection")
        core, checkpoint = load_numerical_model(directory / "model.pt", "cuda")
        expanded = dict(data.bundle)
        expanded["preparation_contract"] = {
            **expanded["preparation_contract"],
            "canary_injection": injection,
        }
        if checkpoint["data_signature"] != data_signature(expanded):
            raise ValueError("Model was not trained on the declared canary dose")
        for kind in ["categorical", "numerical"]:
            model = NumericalAuditModel(core.float(), data, score_kind=kind)
            save_scores(cfg, directory, model, selected, kind)


def save_scores(cfg, directory, model, selected, kind):
    path = directory / f"attack_a_{kind}.joblib"
    if path.exists():
        return
    audit = run_calibrated_membership_audit(
        model,
        decoder=StaticsDecoder(model=model),
        k=0.2,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        chunk_aggregation="max",
    )
    for statistic, scores in audit["train_scores"].items():
        if not selected <= set(scores):
            raise ValueError(f"Missing selected canaries in {statistic}")
    temporary = path.with_suffix(".joblib.partial")
    joblib.dump(
        {
            "audit": audit,
            "canary_keys": sorted(selected),
            "factor": int(cfg.factor),
            "score_kind": kind,
        },
        temporary,
    )
    temporary.replace(path)


def collect(cfg, root):
    """Export curve tables and ROC arrays directly from saved full-population scores."""
    rows, rocs = [], {}

    for factor in cfg.factors:
        directory = point(root, cfg.representation, factor)
        kinds = (
            ["categorical"]
            if cfg.representation == "discrete"
            else ["categorical", "numerical"]
        )
        for kind in kinds:
            result = joblib.load(directory / f"attack_a_{kind}.joblib")
            selected = set(result["canary_keys"])
            audit = result["audit"]
            for statistic, training in audit["train_scores"].items():
                canary = [v for k, v in training.items() if k in selected]
                normal = [v for k, v in training.items() if k not in selected]
                holdout = list(audit["holdout_scores"][statistic].values())
                for contrast, positive, negative in [
                    ("canary_vs_holdout", canary, holdout),
                    ("canary_vs_normal", canary, normal),
                    ("normal_vs_holdout", normal, holdout),
                ]:
                    summary = summarize_attack(positive, negative, verbose=False)
                    rows.append(
                        {
                            "representation": cfg.representation,
                            "factor": factor,
                            "score_kind": kind,
                            "statistic": statistic,
                            "contrast": contrast,
                            **summary,
                        }
                    )
                    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
                    fpr, tpr, _ = roc_curve(labels, np.r_[positive, negative])
                    prefix = f"dup{factor}_{kind}_{statistic}_{contrast}"
                    rocs[prefix + "_fpr"], rocs[prefix + "_tpr"] = fpr, tpr
    target = root / cfg.representation
    plot_curve(rows, target)
    pd.DataFrame(rows).to_csv(target / "curve.csv", index=False)
    np.savez_compressed(target / "curve_rocs.npz", **rocs)
    print(f"Exported {len(rows)} curve comparisons to {target}", flush=True)


def plot_curve(rows, target):
    """Plot both aggregate separation and low-FPR canary recovery."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    frame = frame[frame.contrast == "canary_vs_holdout"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for (kind, statistic), group in frame.groupby(["score_kind", "statistic"]):
        group = group.sort_values("factor")
        label = f"{kind}: {statistic}"
        bounds = np.asarray(group.auc_ci.tolist())
        axes[0].errorbar(
            group.factor,
            group.auc,
            yerr=np.vstack([group.auc - bounds[:, 0], bounds[:, 1] - group.auc]),
            marker="o",
            label=label,
        )
        axes[1].plot(group.factor, group["tpr@fpr=0.01"], marker="o", label=label)
    axes[0].axhline(0.5, color="gray", linestyle=":")
    axes[1].axhline(0.01, color="gray", linestyle=":")
    axes[0].set_ylabel("Canary vs holdout AUROC")
    axes[1].set_ylabel("TPR at 1% FPR")
    for axis in axes:
        axis.set_xlabel("Training duplication factor")
        axis.set_xticks(sorted(frame.factor.unique()))
        axis.set_ylim(0, 1.01)
    axes[0].legend(fontsize=7)
    for extension in ["pdf", "png"]:
        figure.savefig(target / f"curve.{extension}", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
