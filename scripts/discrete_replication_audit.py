"""CPU-only Attack B for the two discrete K=5 releases in both generation modes."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import hydra
import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch

from syntheticicu.privacy_audit_ext import replication_distances
from syntheticicu.data_classes import CENTRE_MAP
from syntheticicu.privacy_audit import _composite_key, collect_real_sequences_with_ids
from syntheticicu.privacy_audit_ext import (
    prepare_sequences,
    compare_distance_distributions,
)


@hydra.main(
    version_base="1.3", config_path="../config", config_name="discrete_replication"
)
def main(cfg: DictConfig):
    """Prepare an exact real cache once, or score a single generated release."""
    root = Path(cfg.output_root).resolve()
    if not root.is_relative_to(Path(hydra.utils.get_original_cwd()).resolve()):
        raise ValueError("Attack B outputs must remain in the repository")
    root.mkdir(parents=True, exist_ok=True)
    if cfg.action == "prepare":
        prepare(cfg, root)
    elif cfg.action == "score":
        score(cfg, root)
    else:
        raise ValueError("Unknown discrete Attack B action")


def prepare_real_group(sequences, identifiers, classes, centers):
    """Retain first chunks and exact center/stay identities before comparison."""
    if len({len(rows) for rows in [sequences, identifiers, classes, centers]}) != 1:
        raise ValueError("Real sequence and conditioning axes differ")
    prepared, mask = prepare_sequences(
        sequences,
        canonicalize=True,
        chunk_zero_only=True,
        return_mask=True,
    )
    keys = [
        _composite_key(identifier, center)
        for identifier, center, keep in zip(identifiers, centers, mask)
        if keep
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate first-chunk center/stay identity")
    if not prepared:
        raise ValueError("No initial real chunks available")
    return dict(
        seqs=prepared,
        classes=np.asarray(classes)[mask],
        centers=np.asarray(centers)[mask],
        keys=keys,
    )


def prepare_generated(sequences, labels, centers):
    """Apply the same complete-sequence policy as the existing corpus audits."""
    labels, centers = np.asarray(labels), np.asarray(centers)
    if not len(sequences) == len(labels) == len(centers):
        raise ValueError("Synthetic sequence and conditioning axes differ")
    prepared, mask = prepare_sequences(
        sequences,
        canonicalize=True,
        require_eos=True,
        return_mask=True,
    )
    if not prepared:
        raise ValueError("No complete generated sequences")
    return dict(seqs=prepared, classes=labels[mask], centers=centers[mask]), mask


def publish_json(path, value):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def publish_result(path, value):
    temporary = path.with_suffix(".joblib.partial")
    joblib.dump(value, temporary)
    temporary.replace(path)


def prepare(cfg, root):
    """Load physiology once, validate exact partitions, and cache token sequences."""
    marker = root / "prepared.json"
    if marker.exists():
        if not (root / "real_sequences.joblib").exists():
            raise FileNotFoundError("Cache completion marker has no sequence artifact")
        return
    configuration = OmegaConf.load(cfg.reference_hparams).cfg
    if configuration.datamodule.shuffle_tokens_within_timepoint:
        raise ValueError(
            "Reference extraction must use fixed order before canonicalization"
        )
    datamodule = hydra.utils.instantiate(configuration.datamodule)
    if datamodule.tokenizer.num_bins != 5:
        raise ValueError("Expected the discrete K=5 source")
    reference = pd.read_parquet(cfg.reference_splits)
    if reference.duplicated(["centre", "stay_id"]).any():
        raise ValueError("Duplicate reference split identities")
    model = SimpleNamespace(_datamodule=datamodule)
    groups = {}
    for partition, subset in [
        ("train", datamodule.train_icu_dataset),
        ("test", datamodule.test_icu_dataset),
    ]:
        sequences, identifiers, classes, centers = collect_real_sequences_with_ids(
            model,
            subset,
            batch_size=32,
            max_batches=None,
            collect_statics=False,
            collect_labels=True,
            collect_centers=True,
        )
        group = prepare_real_group(sequences, identifiers, classes, centers)
        expected = {
            _composite_key(row.stay_id, CENTRE_MAP[str(row.centre).lower()])
            for row in reference.loc[reference.split.eq(partition)].itertuples()
        }
        if set(group["keys"]) != expected:
            raise ValueError(
                f"Real {partition} first chunks differ from reference stays"
            )
        groups[partition] = group
        del sequences
    if set(groups["train"]["keys"]) & set(groups["test"]["keys"]):
        raise ValueError("Real reference and holdout stay identities overlap")
    publish_result(root / "real_sequences.joblib", groups)
    publish_json(
        marker,
        dict(
            hparams_sha256=hashlib.sha256(
                Path(cfg.reference_hparams).read_bytes()
            ).hexdigest(),
            real_train_stays=len(groups["train"]["seqs"]),
            real_holdout_stays=len(groups["test"]["seqs"]),
            first_chunks_only=True,
            canonicalized=True,
            utility_clipping=False,
        ),
    )
    print(marker.read_text(), flush=True)


def score(cfg, root):
    """Reuse the continuous rerun's B calculation on each discrete synthetic corpus."""
    if cfg.representation not in {"shuffled", "fixed"}:
        raise ValueError("Unknown discrete representation")
    if cfg.generation_mode not in {"null", "conditioned"}:
        raise ValueError("Unknown generation mode")
    destination = root / cfg.representation / cfg.generation_mode
    marker = destination / "complete.json"
    if marker.exists():
        if not (destination / "results.joblib").exists():
            raise FileNotFoundError("Completed Attack B has no distance result")
        return
    groups = joblib.load(root / "real_sequences.joblib")
    source = Path(cfg.sources[cfg.representation][cfg.generation_mode])
    sequences, labels, centers = joblib.load(source)
    labels, centers = [
        value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        for value in [labels, centers]
    ]
    if len(labels) != 12000 or not np.isin(labels, [0, 1]).all() or labels.sum() != 840:
        raise ValueError("Expected 12,000 generated stays with exactly 7% positives")
    if cfg.generation_mode == "null" and not np.all(centers == 6):
        raise ValueError("Null release contains center-conditioned rows")
    if (
        cfg.generation_mode == "conditioned"
        and not np.isin(centers, list(CENTRE_MAP.values())).all()
    ):
        raise ValueError("Conditioned release contains unknown/null centers")
    generated, mask = prepare_generated(sequences, labels, centers)
    result = replication_distances(
        generated,
        groups["train"],
        groups["test"],
        cfg.generation_mode == "conditioned",
        cfg.max_candidates_per_cell,
        cfg.seed,
    )
    result["distance_comparison"] = compare_distance_distributions(
        result["generated_to_train"],
        result["generated_to_holdout"],
        result["holdout_to_train"],
    )
    result["generated_indices"] = np.flatnonzero(mask)
    result["holdout_keys"] = groups["test"]["keys"]
    destination.mkdir(parents=True, exist_ok=True)
    publish_result(destination / "results.joblib", result)
    publish_json(
        marker,
        dict(
            configuration=OmegaConf.to_container(cfg, resolve=True),
            source=str(source),
            generated_total=len(mask),
            generated_complete=int(mask.sum()),
            real_train_stays=len(groups["train"]["seqs"]),
            real_holdout_stays=len(groups["test"]["seqs"]),
            direction="synthetic -> real; holdout -> training baseline",
            metric="shared canonical token normalized edit distance",
            candidate_policy="sampled from full real partitions; cap per outcome/center cell",
            utility_clipping=False,
        ),
    )
    print(marker.read_text(), flush=True)


if __name__ == "__main__":
    main()
