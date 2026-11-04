"""Run full-value corpus membership, replication and paired corruption controls."""

import json
import os
from pathlib import Path

import hydra
import joblib
from numba import set_num_threads
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import roc_curve

from syntheticicu.evaluation_numeric import load_evaluation_data
from syntheticicu.generation_numeric import _checkpoint_hash
from syntheticicu.privacy_audit_ext import (
    compare_distance_distributions,
    summarize_attack,
)
from syntheticicu.value_corpus import (
    array,
    cell,
    load_pack,
    nearest,
    pack_stays,
    pools,
    save_pack,
    trajectory,
)
from syntheticicu.value_distances import (
    DISTANCE_NAMES,
    deletion_indices,
    pair_distances,
    recovery_summary,
    robust_scaling,
)
from syntheticicu.workflow_numeric import data_signature


@hydra.main(
    version_base="1.3", config_path="../config", config_name="value_corpus_audit"
)
def main(cfg: DictConfig):
    root = Path(cfg.output_dir).resolve()
    if not root.is_relative_to(Path(hydra.utils.get_original_cwd()).resolve()):
        raise ValueError("Value audit outputs must stay inside the repository")
    if cfg.generation_mode not in {"null", "conditioned"}:
        raise ValueError("Unknown generation mode")
    expected = [value / 10 for value in range(9)]
    if list(cfg.drop_fractions) != expected:
        raise ValueError("Expected the reviewed 0%,10%,...,80% deletion grid")
    if not 1 <= cfg.threads <= int(os.environ.get("SLURM_CPUS_PER_TASK", cfg.threads)):
        raise ValueError("Numerical kernel threads exceed allocated CPUs")
    set_num_threads(int(cfg.threads))
    root.mkdir(parents=True, exist_ok=True)
    actions = dict(
        prepare_real=prepare_real,
        prepare_release=prepare_release,
        membership=membership,
        replication=replication,
        floor=floor,
        collect=collect,
        discrete_floor=discrete_floor,
        floor_collect=floor_collect,
    )
    actions[cfg.action](cfg, root)


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.suffix == ".json":
        temporary.write_text(json.dumps(value, indent=2) + "\n")
    else:
        joblib.dump(value, temporary, compress=0)
    temporary.replace(path)


def prepare_real(cfg, root):
    """Fit scaling only on members, and retain all available real physiology."""
    if (root / "real_complete.json").exists():
        return
    data = load_evaluation_data(cfg.numerical_root, require_release=False)
    training = [s for s in data.bundle["stays"] if s["split"] == "train"]
    matrix = np.concatenate([array(s["values"]) for s in training])
    location, scale = robust_scaling(matrix)
    del matrix
    packed = pack_stays(data.bundle["stays"], data.bundle["schema"], location, scale)
    if len(packed["keys"]) != len(set(packed["keys"])):
        raise ValueError("Duplicate real exact stay identities")
    candidates = sorted(
        packed["keys"][i] for i in np.flatnonzero(packed["splits"] == "train")
    )
    selected = np.random.default_rng(cfg.seed).choice(
        len(candidates), cfg.n_canaries, replace=False
    )
    canaries = [candidates[index] for index in selected]
    save_pack(root / "real.joblib", packed)
    publish(
        root / "parameters.joblib",
        dict(
            location=location,
            scale=scale,
            schema=data.bundle["schema"],
            canaries=canaries,
        ),
    )
    configuration = OmegaConf.to_container(cfg, resolve=True)
    for field in ["action", "generation_mode", "shard_index", "threads"]:
        configuration.pop(field)
    publish(
        root / "real_complete.json",
        dict(
            data_signature=data_signature(data.bundle),
            configuration=configuration,
            stays=len(packed["keys"]),
            hours=int(packed["offsets"][-1]),
            max_hours=int(np.diff(packed["offsets"]).max()),
            scope="complete available trajectories; no utility clipping or chunk truncation",
            precision="preserve supplied float32 measurements; invertible float64 asinh transform",
            scaling="training-only median/IQR; full training range if IQR is zero",
            canary_selection="200 exact training center/stay keys; shared across metrics and modes",
            reference_policy="validation stays only; explicit per-cell cap, never target self-matches",
        ),
    )
    print((root / "real_complete.json").read_text(), flush=True)


def prepare_release(cfg, root):
    destination = root / cfg.generation_mode
    if (destination / "release_complete.json").exists():
        return
    source = Path(cfg.numerical_root)
    if cfg.generation_mode == "conditioned":
        source /= "conditioned"
    data = load_evaluation_data(source)
    reference = json.loads((root / "real_complete.json").read_text())
    if data_signature(data.bundle) != reference["data_signature"]:
        raise ValueError("Synthetic and real numerical populations differ")
    if data.manifest["model_sha256"] != _checkpoint_hash(source / "model.pt"):
        raise ValueError("Synthetic corpus does not match its audited checkpoint")
    if (
        len(data.synthetic) != 12000
        or sum(s["death_icu"] for s in data.synthetic) != 840
    ):
        raise ValueError("Expected 12,000 synthetic stays with exactly 7% positives")
    parameters = joblib.load(root / "parameters.joblib")
    complete = [stay for stay in data.synthetic if stay["terminated"]]
    packed = pack_stays(
        complete, data.bundle["schema"], parameters["location"], parameters["scale"]
    )
    save_pack(destination / "synthetic.joblib", packed)
    real = load_pack(root / "real.joblib")
    conditioned = cfg.generation_mode == "conditioned"
    all_pools = {
        "synthetic": pools(
            packed,
            np.arange(len(packed["keys"])),
            conditioned,
            cfg.candidates_per_cell,
            cfg.seed,
        )
    }
    for split in ["train", "val", "test"]:
        all_pools[split] = pools(
            real,
            np.flatnonzero(real["splits"] == split),
            conditioned,
            cfg.candidates_per_cell,
            cfg.seed,
        )
    expected = set(all_pools["synthetic"])
    if any(set(value) != expected for value in all_pools.values()):
        raise ValueError("Missing conditioning cell; no cross-cell fallback permitted")
    publish(destination / "pools.joblib", all_pools)
    publish(
        destination / "release_complete.json",
        dict(
            data_signature=reference["data_signature"],
            model_sha256=data.manifest["model_sha256"],
            generated_total=len(data.synthetic),
            generated_complete=len(complete),
            candidates_per_cell={
                name: {str(key): len(value) for key, value in group.items()}
                for name, group in all_pools.items()
            },
            metric_names=list(DISTANCE_NAMES),
            generation_mode=cfg.generation_mode,
            candidate_policy="up to 140 per conditioning cell, sampled once and shared by all metrics",
        ),
    )
    print((destination / "release_complete.json").read_text(), flush=True)


def shared(cfg, root):
    directory = root / cfg.generation_mode
    real, synthetic = load_pack(root / "real.joblib"), load_pack(
        directory / "synthetic.joblib"
    )
    all_pools = joblib.load(directory / "pools.joblib")
    return directory, real, synthetic, all_pools


def grouped_nearest(cfg, query_pack, query_indices, candidate_pack, candidate_pools):
    output = np.empty((len(query_indices), 1))
    neighbors = np.empty((len(query_indices), 1), dtype=np.int64)
    conditioned = cfg.generation_mode == "conditioned"
    for group in sorted(candidate_pools):
        positions = [
            i
            for i, index in enumerate(query_indices)
            if cell(query_pack, index, conditioned) == group
        ]
        if not positions:
            continue
        selected = np.asarray(query_indices)[positions]
        output[positions], neighbors[positions] = nearest(
            query_pack,
            selected,
            candidate_pack,
            candidate_pools[group],
            cfg,
        )
    covered = {cell(query_pack, index, conditioned) for index in query_indices}
    if not covered <= set(candidate_pools):
        raise ValueError("Uncovered numerical query conditioning cell")
    return output, neighbors


def batches(cfg, indices):
    if not 0 <= cfg.shard_index < cfg.shards:
        raise ValueError("Invalid shard index")
    selected = np.asarray(indices)[cfg.shard_index :: cfg.shards]
    return [
        selected[start : start + cfg.query_batch_size]
        for start in range(0, len(selected), cfg.query_batch_size)
    ]


def membership(cfg, root):
    """Score all real members/holdout against the same release and independent reference."""
    directory, real, synthetic, all_pools = shared(cfg, root)
    indices = np.flatnonzero(np.isin(real["splits"], ["train", "test"]))
    for number, selected in enumerate(batches(cfg, indices)):
        path = (
            directory
            / "membership"
            / f"shard{cfg.shard_index}"
            / f"batch{number}.joblib"
        )
        if path.exists():
            continue
        generated, neighbors = grouped_nearest(
            cfg, real, selected, synthetic, all_pools["synthetic"]
        )
        publish(
            path,
            dict(
                indices=selected,
                distances=generated,
                neighbors=neighbors,
            ),
        )
        print(f"Membership shard {cfg.shard_index}: batch {number + 1}", flush=True)
    publish(
        directory / "membership" / f"shard{cfg.shard_index}.json", {"complete": True}
    )


def replication(cfg, root):
    """Save genuine synthetic-to-real distances and holdout-to-training controls."""
    directory, real, synthetic, all_pools = shared(cfg, root)
    specifications = [
        ("generated", synthetic, np.arange(len(synthetic["keys"])), ["train", "test"]),
        ("baseline", real, np.flatnonzero(real["splits"] == "test"), ["train"]),
    ]
    for name, queries, indices, partitions in specifications:
        for number, selected in enumerate(batches(cfg, indices)):
            path = (
                directory
                / "replication"
                / f"shard{cfg.shard_index}"
                / f"{name}{number}.joblib"
            )
            if path.exists():
                continue
            result = {"indices": selected, "query_kind": name}
            for partition in partitions:
                result[partition], result[f"nearest_{partition}"] = grouped_nearest(
                    cfg,
                    queries,
                    selected,
                    real,
                    all_pools[partition],
                )
            publish(path, result)
            print(
                f"Replication {name} shard {cfg.shard_index}: batch {number + 1}",
                flush=True,
            )
    publish(
        directory / "replication" / f"shard{cfg.shard_index}.json", {"complete": True}
    )


def floor(cfg, root):
    """Recover copies after nested deletion using the frozen shared masks."""
    directory, real, synthetic, all_pools = shared(cfg, root)
    parameters = joblib.load(root / "parameters.joblib")
    lookup = {key: index for index, key in enumerate(real["keys"])}
    chosen = np.asarray([lookup[key] for key in parameters["canaries"]])
    for selected in batches(cfg, chosen):
        pending = np.asarray(
            [
                index
                for index in selected
                if not (directory / "floor" / f"patient{index}.joblib").exists()
            ],
            dtype=np.int64,
        )
        if not len(pending):
            continue
        backgrounds, _ = grouped_nearest(
            cfg, real, pending, synthetic, all_pools["synthetic"]
        )
        for position, index in enumerate(pending):
            path = directory / "floor" / f"patient{index}.joblib"
            if path.exists():
                continue
            background = backgrounds[position]
            original = trajectory(real, index)
            rows = []
            for fraction in cfg.drop_fractions:
                keep = deletion_indices(
                    len(original["hours"]), fraction, real["keys"][index], cfg.seed
                )
                planted = trajectory(real, index, keep)
                distance = pair_distances(
                    original, planted, cfg.time_weight, cfg.max_pair_cells
                )
                if fraction == 0 and np.any(distance != 0):
                    raise ValueError("A literal planted copy must have zero distance")
                rows.append(
                    dict(
                        fraction=fraction,
                        hours=len(original["hours"]),
                        kept=len(keep),
                        planted=distance,
                        background=background,
                    )
                )
            publish(path, dict(key=real["keys"][index], rows=rows))
    publish(directory / "floor" / f"shard{cfg.shard_index}.json", {"complete": True})


def floor_collect(cfg, root):
    collect_floor(cfg, root / cfg.generation_mode)


def shard_files(cfg, directory, kind):
    for shard in range(cfg.shards):
        if not (directory / kind / f"shard{shard}.json").exists():
            raise ValueError(f"Incomplete {kind} shard {shard}")
    return sorted((directory / kind).glob("shard*/*.joblib"))


def collect(cfg, root):
    """Export individual attacks and full distance arrays; never choose on final holdout."""
    directory, real, synthetic, _ = shared(cfg, root)
    parts = [joblib.load(path) for path in shard_files(cfg, directory, "membership")]
    indices = np.concatenate([part["indices"] for part in parts])
    expected = np.flatnonzero(np.isin(real["splits"], ["train", "test"]))
    if len(indices) != len(set(indices)) or set(indices) != set(expected):
        raise ValueError("Membership collection must cover each real stay exactly once")
    distances = np.concatenate([part["distances"] for part in parts])
    member = real["splits"][indices] == "train"
    reports, rocs = [], {}
    for column, name in enumerate(DISTANCE_NAMES):
        for label, scores in [
            ("raw", -distances[:, column]),
        ]:
            summary = summarize_attack(scores[member], scores[~member], verbose=False)
            reports.append(dict(metric=name, statistic=label, **summary))
            fpr, tpr, _ = roc_curve(member.astype(int), scores)
            rocs[f"{name}_{label}_fpr"], rocs[f"{name}_{label}_tpr"] = fpr, tpr
    pd.DataFrame(reports).to_csv(directory / "membership_metrics.csv", index=False)
    np.savez_compressed(directory / "membership_rocs.npz", **rocs)
    publish(
        directory / "membership_scores.joblib",
        dict(indices=indices, distances=distances, members=member),
    )
    collect_replication(cfg, directory, real, synthetic)
    collect_floor(cfg, directory)
    publish(
        directory / "complete.json",
        dict(
            complete=True,
            metrics=list(DISTANCE_NAMES),
            scopes="full trajectories; continuous bin-only floor paired on same data",
        ),
    )


def collect_replication(cfg, directory, real, synthetic):
    reports, arrays = [], {}
    files = shard_files(cfg, directory, "replication")
    for kind, count in [
        ("generated", len(synthetic["keys"])),
        ("baseline", int(np.sum(real["splits"] == "test"))),
    ]:
        parts = [joblib.load(path) for path in files if path.name.startswith(kind)]
        indices = np.concatenate([part["indices"] for part in parts])
        if len(indices) != count or len(set(indices)) != count:
            raise ValueError(
                "Replication does not cover every intended query exactly once"
            )
        for partition in ["train", "test"] if kind == "generated" else ["train"]:
            arrays[f"{kind}_to_{partition}"] = np.concatenate(
                [part[partition] for part in parts]
            )
        arrays[f"{kind}_indices"] = indices
    for column, name in enumerate(DISTANCE_NAMES):
        comparison = compare_distance_distributions(
            arrays["generated_to_train"][:, column],
            arrays["generated_to_test"][:, column],
            arrays["baseline_to_train"][:, column],
        )
        reports.append(dict(metric=name, comparison=comparison))
    publish(
        directory / "replication_results.joblib",
        dict(arrays=arrays, comparisons=reports),
    )


def collect_floor(cfg, directory):
    parts = [
        joblib.load(path)
        for path in sorted((directory / "floor").glob("patient*.joblib"))
    ]
    if (
        len(parts) != cfg.n_canaries
        or len({part["key"] for part in parts}) != cfg.n_canaries
    ):
        raise ValueError("Incomplete or duplicated canary population")
    rows = []
    for position, fraction in enumerate(cfg.drop_fractions):
        for column, metric in enumerate(DISTANCE_NAMES):
            planted = np.asarray(
                [part["rows"][position]["planted"][column] for part in parts]
            )
            background = np.asarray(
                [part["rows"][position]["background"][column] for part in parts]
            )
            rows.append(
                dict(
                    metric=metric,
                    fraction=fraction,
                    **recovery_summary(planted, background),
                )
            )
    pd.DataFrame(rows).to_csv(directory / "recovery.csv", index=False)


def discrete_floor(cfg, root):
    """Extend the existing discrete initial-chunk floor to the same nine deletion levels."""
    from scripts.discrete_replication_audit import prepare_generated

    configuration = OmegaConf.load(cfg.discrete_config)
    source = configuration.sources[cfg.discrete_representation][cfg.generation_mode]
    sequences, labels, centers = joblib.load(source)
    generated, _ = prepare_generated(sequences, array(labels), array(centers))
    real = joblib.load(cfg.discrete_cache)["train"]
    ordered = sorted(real["keys"])
    chosen = np.random.default_rng(cfg.seed).choice(
        len(ordered), cfg.n_canaries, replace=False
    )
    lookup = {key: index for index, key in enumerate(real["keys"])}
    candidates = pools(
        dict(labels=generated["classes"], centers=generated["centers"]),
        np.arange(len(generated["seqs"])),
        cfg.generation_mode == "conditioned",
        cfg.candidates_per_cell,
        cfg.seed,
    )
    rows = []
    for selected in chosen:
        key = ordered[selected]
        index = lookup[key]
        group = (
            int(real["classes"][index]),
            int(real["centers"][index]) if cfg.generation_mode == "conditioned" else -1,
        )
        sequence = real["seqs"][index]
        background = min(
            Levenshtein.normalized_distance(sequence, generated["seqs"][candidate])
            for candidate in candidates[group]
        )
        demo, blocks, terminal = split_bin_hours(sequence)
        for fraction in cfg.drop_fractions:
            keep = deletion_indices(len(blocks), fraction, key, cfg.seed)
            planted = demo + ["<GAP>"]
            body = list(sequence[:-1]) if terminal else list(sequence)
            for position, hour in enumerate(keep):
                planted.extend(blocks[hour])
                if position + 1 < len(keep) or body[-1] == "<GAP>":
                    planted.append("<GAP>")
            # Preserve the source's final separator/EOS convention at zero deletion.
            if fraction == 0:
                planted = list(sequence)
            else:
                if terminal:
                    planted.append("<EOS>")
            rows.append(
                dict(
                    key=key,
                    fraction=fraction,
                    background=background,
                    planted=Levenshtein.normalized_distance(sequence, planted),
                )
            )
    frame = pd.DataFrame(rows)
    summaries = [
        dict(
            fraction=fraction,
            **recovery_summary(group.planted.to_numpy(), group.background.to_numpy()),
        )
        for fraction, group in frame.groupby("fraction")
    ]
    destination = root / "discrete" / cfg.discrete_representation / cfg.generation_mode
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "recovery_patients.csv", index=False)
    pd.DataFrame(summaries).to_csv(destination / "recovery.csv", index=False)
    publish(
        destination / "complete.json",
        dict(
            scope="existing initial chunks",
            fractions=list(cfg.drop_fractions),
            source=str(source),
            canonicalized=True,
            candidates_per_cell=cfg.candidates_per_cell,
        ),
    )


def split_bin_hours(sequence):
    first = sequence.index("<GAP>")
    demo, blocks, current = list(sequence[:first]), [], []
    for token in sequence[first + 1 :]:
        if token == "<EOS>":
            break
        if token == "<GAP>":
            blocks.append(current)
            current = []
        else:
            current.append(token)
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError("Discrete canary contains no timepoints")
    return demo, blocks, "<EOS>" in sequence


if __name__ == "__main__":
    main()
