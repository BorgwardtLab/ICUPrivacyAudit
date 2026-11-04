import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from scipy.stats import chisquare
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process
from tqdm import tqdm
from collections import defaultdict

from syntheticicu.data_classes import CENTRE_MAP

STATICS_COLUMNS = ["female_static", "age_static", "height_static", "weight_static"]
NULL_CENTER_LABEL = 6  # unconditioned generation (learned null-center embedding)

# collate_fn is rebuilt on every val_dataloader() call, so we cache it per datamodule
_COLLATE_FN_CACHE = {}


def _composite_key(stay_id, center_id):
    # stay ids repeat across centres, so the patient identity is the pair
    return f"{int(center_id)}_{int(stay_id)}"


def _center_from_key(key):
    """Composite keys are 'centre_stayid', so the centre is already embedded
    and no extra lookup is needed to parse it back out."""
    return int(key.split("_")[0])


def exact_batch_keys(subset, offset, IDs):
    """
    Exact (stay_ids, centre_ids) for the rows of one batch of an unshuffled
    make_eval_loader over `subset`, starting at row `offset` of the subset.

    The collated IDs are float32 (ICUDataset.__getitem__ and pad_collate), which
    rounds stay ids above 2**24 (all of MIIV) and merges distinct patients under
    one key. The base dataset still holds them as int64, and an unshuffled loader
    over a Subset yields subset.indices in order without dropping rows, so row j
    of the batch is subset.indices[offset + j]. Every row is checked against the
    collated stay id and centre, so a misaligned mapping raises instead of keying
    a score to the wrong patient.
    """
    if not isinstance(subset, Subset):
        raise TypeError(
            f"exact keys need a Subset of the base dataset, got {type(subset).__name__}"
        )
    base = subset.dataset
    n_rows = IDs.shape[0]
    rows = np.asarray(subset.indices)[offset : offset + n_rows]
    if len(rows) != n_rows:
        raise ValueError(
            f"batch at offset {offset} has {n_rows} rows, subset has {len(rows)} left"
        )

    y = base.y
    exact_ids = (
        y[rows, 0].numpy() if torch.is_tensor(y) else np.asarray(y)[rows, 0]
    ).astype(np.int64)
    # same resolution as ICUDataset.__getitem__
    exact_centers = np.array(
        [
            CENTRE_MAP.get(str(s).lower(), 0)
            for s in np.asarray(base.stay_sources)[rows]
        ],
        dtype=np.int64,
    )

    ids_np = IDs.cpu().numpy() if torch.is_tensor(IDs) else np.asarray(IDs)
    bad = np.flatnonzero(
        ids_np[:, 0].astype(np.int64) != exact_ids.astype(np.float32).astype(np.int64)
    )
    if ids_np.shape[1] > 3:
        bad = np.union1d(
            bad, np.flatnonzero(ids_np[:, -1].astype(np.int64) != exact_centers)
        )
    if len(bad):
        i = int(bad[0])
        raise RuntimeError(
            f"loader rows misaligned with subset.indices: {len(bad)}/{n_rows} rows in the batch "
            f"at offset {offset} disagree, first at row {i}: collated id={ids_np[i, 0]!r} "
            f"centre={ids_np[i, -1]!r}, base row {rows[i]} id={exact_ids[i]} centre={exact_centers[i]}"
        )
    return exact_ids, exact_centers


def _aggregate_by_patient(ids, centers, values, method="max"):
    # a patient contributes several chunks; max for scores where higher is riskier,
    # min for distances where closer is riskier
    buckets = defaultdict(list)
    for sid, cid, v in zip(ids, centers, values):
        buckets[_composite_key(sid, cid)].append(v)
    agg_fn = {"max": np.max, "min": np.min, "mean": np.mean}[method]
    return {k: float(agg_fn(v)) for k, v in buckets.items()}


def get_stratified_patient_keys(model, dataset, n_per_center, batch_size=32, seed=0):
    """
    Collects (stay_id, centre) for the whole dataset and returns a set of
    composite keys already stratified by center. Computed once and reused
    across all attacks in the main run, so they evaluate exactly the same
    patient cohort and `combine_risk_signals` does not collapse due to two
    independent random samples barely overlapping.
    """
    # note: stratification happens over chunks, so the number of unique patients
    # per centre ends up below n_per_center
    _, ids, centers = collect_real_sequences_with_ids(
        model,
        dataset,
        batch_size=batch_size,
        max_batches=None,
        collect_statics=False,
        collect_labels=False,
        collect_centers=True,
    )
    ids_arr, centers_arr = np.array(ids), np.array(centers)
    stratified, keep_idx = stratified_subsample_by_center(
        {"ids": ids_arr, "centers": centers_arr},
        centers_arr,
        n_per_center,
        seed=seed,
    )
    return {
        _composite_key(sid, cid)
        for sid, cid in zip(stratified["ids"], stratified["centers"])
    }


def stratified_subsample_by_center(arrays_dict, centers, n_per_center, seed=0):
    """
    Subsamples to n_per_center per center so no single center dominates the
    evaluated cohort by sheer numerical representation. arrays_dict: {name: array-like}
    with all arrays positionally aligned with `centers`. Returns a dict with
    the same keys, already subsampled.
    """
    rng = np.random.default_rng(seed)
    centers_arr = np.asarray(centers)
    keep_idx = []
    for centre in np.unique(centers_arr):
        centre_idx = np.where(centers_arr == centre)[0]
        n_take = min(n_per_center, len(centre_idx))
        if n_take < n_per_center:
            print(
                f"  WARNING: centre={centre} only has {len(centre_idx)} samples "
                f"(< n_per_center={n_per_center}); using all of them."
            )
        keep_idx.extend(rng.choice(centre_idx, n_take, replace=False).tolist())
    keep_idx = np.array(sorted(keep_idx))

    out = {}
    for name, arr in arrays_dict.items():
        if isinstance(arr, list):
            out[name] = [arr[i] for i in keep_idx]
        else:
            out[name] = np.asarray(arr)[keep_idx]
    return out, keep_idx


def stratified_resample_keys(score_dict, n_per_center, seed):
    """
    Draws a fresh stratified subsample of patient keys from an already computed
    {key: score} dict, grouping by the center parsed from each composite key.
    Used for cheap bootstrap resampling without recomputing model likelihoods
    or edit distances.
    """
    rng = np.random.default_rng(seed)
    by_center = defaultdict(list)
    for k in score_dict:
        by_center[_center_from_key(k)].append(k)
    keep = []
    for keys in by_center.values():
        n_take = min(n_per_center, len(keys))
        keep.extend(rng.choice(keys, n_take, replace=False).tolist())
    return set(keep)


def detect_center_conditioning(
    gen_centers, real_centers, min_unique=2, null_label=NULL_CENTER_LABEL
):
    # decides whether the synthetic corpus carries a centre trace, so the same
    # code path works for both generation modes without manual reconfiguration
    gen_centers_arr = np.asarray(gen_centers)
    real_centers_arr = np.asarray(real_centers)
    conditioned_mask = gen_centers_arr != null_label
    n_null = (~conditioned_mask).sum()
    n_conditioned = conditioned_mask.sum()
    gen_unique = set(np.unique(gen_centers_arr[conditioned_mask]).tolist())
    real_unique = set(np.unique(real_centers_arr).tolist())
    shared = gen_unique & real_unique
    has_signal = len(gen_unique) >= min_unique and len(shared) >= min_unique
    print(f"  Conditioned synthetics: {n_conditioned} | Unconditioned: {n_null}")
    print(
        f"  Synthetic centers: {sorted(gen_unique)} | Real centers: {sorted(real_unique)} | Shared: {sorted(shared)}"
    )
    print(
        f"  -> Center-conditioning signal {'DETECTED' if has_signal else 'NOT detected'}."
    )
    return has_signal


def bootstrap_auc_scores(scores_a, scores_b, n_boot=1000, seed=0):
    """scores_a = members (label=1), scores_b = non-members (label=0)."""
    # resampling both groups with replacement gives the sampling variability of
    # the AUC, which is what the confidence intervals in the figures report
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    aucs = []
    for _ in range(n_boot):
        a_bs = rng.choice(a, len(a), replace=True)
        b_bs = rng.choice(b, len(b), replace=True)
        s = np.concatenate([a_bs, b_bs])
        l = np.concatenate([np.ones(len(a_bs)), np.zeros(len(b_bs))])
        aucs.append(roc_auc_score(l, s))
    return (
        float(np.mean(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def compute_local_auc(train_scores, holdout_scores, percentile=95, n_boot=1000, seed=0):
    """AUC restricted to the upper tail (score >= given percentile in train)."""
    # a global AUC slightly above 0.5 can come from a diffuse shift of the whole
    # distribution with no identifiable patient, so we also look at the tail only
    train_scores = np.asarray(train_scores)
    holdout_scores = np.asarray(holdout_scores)
    cutoff = np.percentile(train_scores, percentile)
    train_tail = train_scores[train_scores >= cutoff]
    holdout_tail = holdout_scores[holdout_scores >= cutoff]

    s = np.concatenate([train_tail, holdout_tail])
    l = np.concatenate([np.ones(len(train_tail)), np.zeros(len(holdout_tail))])
    local_auc = roc_auc_score(l, s)
    mean_ci, ci_lo, ci_hi = bootstrap_auc_scores(
        train_tail, holdout_tail, n_boot=n_boot, seed=seed
    )

    frac_holdout_exceeding = float((holdout_scores >= cutoff).mean())

    return {
        "cutoff": float(cutoff),
        "percentile": percentile,
        "local_auc": float(local_auc),
        "local_auc_ci": (ci_lo, ci_hi),
        "n_train_tail": int(len(train_tail)),
        "n_holdout_tail": int(len(holdout_tail)),
        "frac_holdout_exceeding_cutoff": frac_holdout_exceeding,
    }


# Stratified bootstrap, measures dependence of the headline results
# on the single stratified sample drawn in the main run, without
# recomputing model likelihoods or edit distances
def bootstrap_across_stratified_draws(
    full_train_scores,
    full_holdout_scores,  # membership Min-K%, patient-level, all patients
    full_gen_train,
    full_gen_holdout,  # generation distance, patient-level, all patients
    full_train_statics,  # for the height_static confound check
    n_per_center=3000,
    n_boot=30,
    percentile=99.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    metrics = defaultdict(list)

    for _ in tqdm(range(n_boot), desc="Bootstrapping stratified draws"):
        draw_seed = int(rng.integers(0, 1_000_000))
        train_keys = stratified_resample_keys(
            full_train_scores, n_per_center, draw_seed
        )
        holdout_keys = stratified_resample_keys(
            full_holdout_scores, n_per_center, draw_seed
        )

        # Membership AUC on this draw
        t_scores = np.array([full_train_scores[k] for k in train_keys])
        h_scores = np.array([full_holdout_scores[k] for k in holdout_keys])
        s = np.concatenate([t_scores, h_scores])
        l = np.concatenate([np.ones(len(t_scores)), np.zeros(len(h_scores))])
        metrics["membership_auc"].append(roc_auc_score(l, s))

        # Generation-distance AUC on the same draw (intersected with availability)
        gt_keys = train_keys & set(full_gen_train.keys())
        gh_keys = holdout_keys & set(full_gen_holdout.keys())
        has_gen = len(gt_keys) > 10 and len(gh_keys) > 10
        if has_gen:
            # negated because for distances closer means riskier
            gt = np.array([-full_gen_train[k] for k in gt_keys])
            gh = np.array([-full_gen_holdout[k] for k in gh_keys])
            sg = np.concatenate([gt, gh])
            lg = np.concatenate([np.ones(len(gt)), np.zeros(len(gh))])
            metrics["generation_auc"].append(roc_auc_score(lg, sg))

        # Combined risk + height_static confounding, same draw
        risks_mink = per_patient_percentile_risk(
            {k: full_train_scores[k] for k in train_keys},
            {k: full_holdout_scores[k] for k in holdout_keys},
            True,
        )
        if has_gen:
            risks_gen = per_patient_percentile_risk(
                {k: full_gen_train[k] for k in gt_keys},
                {k: full_gen_holdout[k] for k in gh_keys},
                False,
            )
            combined = combine_risk_signals([risks_mink, risks_gen], method="max")
        else:
            combined = risks_mink

        tail = flag_high_risk_patients(combined, percentile=percentile)
        flagged = set(tail["flagged"].keys())
        all_ids = [k for k in combined if k in full_train_statics]

        if len(flagged) >= 5:
            values_by_id = {
                k: full_train_statics[k][2] for k in all_ids
            }  # height_static, index 2
            res = subgroup_composition_test(values_by_id, flagged, all_ids)
            metrics["height_p_pooled"].append(
                min(1.0, res["p_value"] * 4)
            )  # Bonferroni, n_tests=4

            # how over-represented the worst centre is in the risk tail
            flagged_centers = pd.Series([_center_from_key(k) for k in flagged])
            all_centers = pd.Series([_center_from_key(k) for k in all_ids])
            fp, op = flagged_centers.value_counts(
                normalize=True
            ), all_centers.value_counts(normalize=True)
            metrics["max_center_ratio"].append(float((fp / op).max()))

    summary = {}
    for name, vals in metrics.items():
        arr = np.array(vals)
        summary[name] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "n": len(arr),
        }
    return summary, dict(metrics)


def _unpack_batch(batch):
    # batch layout depends on whether temporal positional embeddings are enabled
    if len(batch) == 5:
        X, time_positions, attention_mask, S, IDs = batch
    else:
        X, attention_mask, S, IDs = batch
        time_positions = None
    return X, time_positions, attention_mask, S, IDs


# Legacy (scripts/run_privacy_audit.py). The stay ids returned here, and keyed
# by run_membership_audit below, are the collated float32 ones: MIIV ids are
# rounded and distinct patients share a key. Do not reuse for any keyed analysis
# (grouping, matching, cohort restriction); run_calibrated_membership_audit in
# privacy_audit_ext keys through exact_batch_keys.
def compute_batch_token_scores(model, batch):
    vocabulary = model.vocabulary
    pad_id = vocabulary["<PAD>"]
    eos_id = vocabulary["<EOS>"]
    X, time_positions, attention_mask, S, IDs = _unpack_batch(batch)
    if time_positions is not None:
        time_positions = time_positions.to(model.device)

    X = X.to(model.device)
    attention_mask = attention_mask.to(model.device)
    IDs = IDs.to(model.device)
    # label row is [stay_id, outcome, onset, centre_id]
    class_labels = IDs[:, 1]
    center_ids = IDs[:, -1] if IDs.size(1) > 3 else None

    with torch.no_grad():
        logits = model.forward(
            input_ids=X,
            class_labels=class_labels,
            center_ids=center_ids,
            time_positions=time_positions,
            attention_mask=attention_mask,
        )

    if hasattr(logits, "logits"):
        logits = logits.logits

    # drop the conditioning prefix so that logits[t] lines up with x[t+1]
    T = X.size(1)
    prefix = model.actual_prefix_len
    logits = logits[:, : T + prefix, :]
    shift_logits = logits[:, prefix:-1, :]
    shift_labels = X[:, 1:]

    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # EOS is structural: its probability tracks sequence length rather than
    # physiology, so keeping it would add a spurious membership signal
    valid_mask = shift_labels != pad_id
    structural_mask = shift_labels == eos_id
    content_mask = valid_mask & (~structural_mask)

    return {
        "token_logp": token_logp.cpu().numpy(),
        "content_mask": content_mask.cpu().numpy(),
        "class_labels": class_labels.cpu().numpy(),
        "stay_ids": IDs[:, 0].cpu().numpy(),
        "token_ids": X.cpu().numpy(),
        "statics": S.cpu().numpy() if torch.is_tensor(S) else np.asarray(S),
        "center_ids": (
            center_ids.cpu().numpy() if center_ids is not None else np.zeros(len(IDs))
        ),
    }


def min_k_percent_prob(logp_seq: np.ndarray, k: float = 0.2) -> float:
    # average log-probability of the k fraction of least likely tokens; a
    # memorised trajectory leaves no surprising token behind
    n_k = max(1, int(len(logp_seq) * k))
    return float(np.sort(logp_seq)[:n_k].mean())


def mean_logp(logp_seq: np.ndarray) -> float:
    # baseline that averages over the whole sequence and therefore mixes
    # memorisation with the model's general fluency
    return float(logp_seq.mean()) if len(logp_seq) else 0.0


def make_eval_loader(
    datamodule, subset, batch_size=32, shuffle=False, seed=0, num_workers=0
):
    # workers keep the order of an unshuffled loader, so exact_batch_keys still
    # lines up; they only overlap __getitem__ with the forward passes
    key = id(datamodule)
    if key not in _COLLATE_FN_CACHE:
        _COLLATE_FN_CACHE[key] = datamodule.val_dataloader().collate_fn
    collate_fn = _COLLATE_FN_CACHE[key]
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# A) Membership audit
# Legacy: keys patients with the rounded float32 ids from compute_batch_token_scores.
# Do not reuse for any keyed analysis.
def run_membership_audit(
    model,
    batch_size: int = 32,
    k_values=(0.05, 0.1, 0.2, 0.4),
    max_batches_per_split: int = None,
    use_test_as_holdout: bool = True,
    k_for_downstream: float = 0.2,
    chunk_aggregation: str = "max",
    n_per_center: int = None,
    restrict_to_patient_keys: dict = None,
    local_auc_percentile: float = 95,
):
    # attacks the weights, so it does not depend on the generation mode and is
    # computed once and shared across both synthetic corpora
    dm = model._datamodule
    train_loader = make_eval_loader(dm, dm.train_icu_dataset, batch_size)
    holdout_subset = dm.test_icu_dataset if use_test_as_holdout else dm.val_icu_dataset
    holdout_loader = make_eval_loader(dm, holdout_subset, batch_size)

    def collect(loader, membership_flag):
        per_seq_scores = {k: [] for k in k_values}
        per_seq_mean, classes, stay_ids, statics, center_ids, seq_lengths = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        total_batches = (
            len(loader)
            if max_batches_per_split is None
            else min(len(loader), max_batches_per_split)
        )
        desc_name = (
            "TRAIN (members)" if membership_flag == 1 else "HOLDOUT (non-members)"
        )
        progress_bar = tqdm(
            enumerate(loader),
            total=total_batches,
            desc=f"Scoring {desc_name}",
            unit="batch",
        )

        for b_idx, batch in progress_bar:
            if max_batches_per_split is not None and b_idx >= max_batches_per_split:
                break
            out = compute_batch_token_scores(model, batch)
            B = out["token_logp"].shape[0]
            for i in range(B):
                mask = out["content_mask"][i]
                logp_seq = out["token_logp"][i][mask]
                if len(logp_seq) == 0:
                    continue
                for k in k_values:
                    per_seq_scores[k].append(min_k_percent_prob(logp_seq, k))
                per_seq_mean.append(mean_logp(logp_seq))
                classes.append(out["class_labels"][i])
                stay_ids.append(out["stay_ids"][i])
                statics.append(out["statics"][i])
                center_ids.append(out["center_ids"][i])
                seq_lengths.append(int(mask.sum()))
        return (
            per_seq_scores,
            np.array(per_seq_mean),
            np.array(classes),
            np.array(stay_ids),
            np.array(statics),
            np.array(center_ids),
            np.array(seq_lengths),
        )

    print("Scoring TRAIN (members)...")
    (
        train_scores_k,
        train_mean,
        train_classes,
        train_ids,
        train_statics,
        train_centers,
        train_seq_lengths,
    ) = collect(train_loader, 1)
    print(f"  n = {len(train_mean)}")

    print("Scoring HOLDOUT (non-members)...")
    (
        holdout_scores_k,
        holdout_mean,
        holdout_classes,
        holdout_ids,
        holdout_statics,
        holdout_centers,
        holdout_seq_lengths,
    ) = collect(holdout_loader, 0)
    print(f"  n = {len(holdout_mean)}")

    if restrict_to_patient_keys is not None:
        # use the shared cohort so the three attacks score the same patients
        print("\nFiltering to FIXED stratified patient set (shared across attacks)...")
        train_keys_set = restrict_to_patient_keys["train"]
        holdout_keys_set = restrict_to_patient_keys["holdout"]

        train_mask = np.array(
            [
                _composite_key(s, c) in train_keys_set
                for s, c in zip(train_ids, train_centers)
            ]
        )
        holdout_mask = np.array(
            [
                _composite_key(s, c) in holdout_keys_set
                for s, c in zip(holdout_ids, holdout_centers)
            ]
        )

        train_mean, train_classes, train_ids = (
            train_mean[train_mask],
            train_classes[train_mask],
            train_ids[train_mask],
        )
        train_statics, train_centers, train_seq_lengths = (
            train_statics[train_mask],
            train_centers[train_mask],
            train_seq_lengths[train_mask],
        )
        train_scores_k = {k: np.array(v)[train_mask] for k, v in train_scores_k.items()}

        holdout_mean, holdout_classes, holdout_ids = (
            holdout_mean[holdout_mask],
            holdout_classes[holdout_mask],
            holdout_ids[holdout_mask],
        )
        holdout_statics, holdout_centers, holdout_seq_lengths = (
            holdout_statics[holdout_mask],
            holdout_centers[holdout_mask],
            holdout_seq_lengths[holdout_mask],
        )
        holdout_scores_k = {
            k: np.array(v)[holdout_mask] for k, v in holdout_scores_k.items()
        }
        print(
            f"  Post-filter: n_train={len(train_mean)}, n_holdout={len(holdout_mean)}"
        )

    elif n_per_center is not None:
        # fallback path when no shared cohort was precomputed
        print(
            f"\nStratifying by center: {n_per_center} chunks/center max (train and holdout)"
        )
        train_dict = {
            "mean": train_mean,
            "classes": train_classes,
            "ids": train_ids,
            "statics": train_statics,
            "centers": train_centers,
            "lengths": train_seq_lengths,
        }
        for k in k_values:
            train_dict[f"score_k{k}"] = np.array(train_scores_k[k])
        train_dict, _ = stratified_subsample_by_center(
            train_dict, train_centers, n_per_center, seed=0
        )
        train_mean, train_classes, train_ids = (
            train_dict["mean"],
            train_dict["classes"],
            train_dict["ids"],
        )
        train_statics, train_centers, train_seq_lengths = (
            train_dict["statics"],
            train_dict["centers"],
            train_dict["lengths"],
        )
        train_scores_k = {k: train_dict[f"score_k{k}"] for k in k_values}

        holdout_dict = {
            "mean": holdout_mean,
            "classes": holdout_classes,
            "ids": holdout_ids,
            "statics": holdout_statics,
            "centers": holdout_centers,
            "lengths": holdout_seq_lengths,
        }
        for k in k_values:
            holdout_dict[f"score_k{k}"] = np.array(holdout_scores_k[k])
        holdout_dict, _ = stratified_subsample_by_center(
            holdout_dict, holdout_centers, n_per_center, seed=0
        )
        holdout_mean, holdout_classes, holdout_ids = (
            holdout_dict["mean"],
            holdout_dict["classes"],
            holdout_dict["ids"],
        )
        holdout_statics, holdout_centers, holdout_seq_lengths = (
            holdout_dict["statics"],
            holdout_dict["centers"],
            holdout_dict["lengths"],
        )
        holdout_scores_k = {k: holdout_dict[f"score_k{k}"] for k in k_values}
        print(
            f"  Post-stratification: n_train={len(train_mean)}, n_holdout={len(holdout_mean)}"
        )

    results = {"n_train": len(train_mean), "n_holdout": len(holdout_mean)}

    print("\nAggregate Min-K% Prob AUC (chunk-level):")
    aucs = {}
    for k in k_values:
        scores = np.concatenate([train_scores_k[k], holdout_scores_k[k]])
        labels = np.concatenate(
            [np.ones(len(train_scores_k[k])), np.zeros(len(holdout_scores_k[k]))]
        )
        aucs[k] = roc_auc_score(labels, scores)
        print(f"  k={k:>4}: AUC = {aucs[k]:.3f}")
    results["aggregate_auc_by_k"] = aucs

    scores = np.concatenate([train_mean, holdout_mean])
    labels = np.concatenate([np.ones(len(train_mean)), np.zeros(len(holdout_mean))])
    results["aggregate_auc_meanlogp"] = roc_auc_score(labels, scores)
    print(f"  mean-logp baseline: AUC = {results['aggregate_auc_meanlogp']:.3f}")

    # reported for transparency only; k is fixed a priori to avoid selecting the
    # best performing value after the fact
    best_k_argmax = max(aucs, key=aucs.get)
    print(
        f"\n[DIAGNOSTIC ONLY] argmax-AUC k = {best_k_argmax} (AUC={aucs[best_k_argmax]:.3f})"
    )

    k_dl = k_for_downstream if k_for_downstream in k_values else k_values[0]
    train_scores_arr = np.array(train_scores_k[k_dl])
    holdout_scores_arr = np.array(holdout_scores_k[k_dl])

    print(f"\nStratified by class label (k={k_dl}):")
    for cls in np.unique(np.concatenate([train_classes, holdout_classes])):
        t_idx, h_idx = train_classes == cls, holdout_classes == cls
        if t_idx.sum() < 20 or h_idx.sum() < 20:
            continue
        s = np.concatenate([train_scores_arr[t_idx], holdout_scores_arr[h_idx]])
        l = np.concatenate([np.ones(t_idx.sum()), np.zeros(h_idx.sum())])
        print(
            f"  class={cls}: AUC = {roc_auc_score(l, s):.3f} (n_train={t_idx.sum()}, n_holdout={h_idx.sum()})"
        )

    local_auc_res = compute_local_auc(
        train_scores_arr, holdout_scores_arr, percentile=local_auc_percentile
    )
    print(
        f"\nLocal AUC (score >= P{local_auc_percentile}): {local_auc_res['local_auc']:.3f} "
        f"[{local_auc_res['local_auc_ci'][0]:.3f}, {local_auc_res['local_auc_ci'][1]:.3f}] "
        f"(n_train_tail={local_auc_res['n_train_tail']}, n_holdout_tail={local_auc_res['n_holdout_tail']})"
    )
    print(
        f"  Fraction of holdout exceeding the cutoff: {local_auc_res['frac_holdout_exceeding_cutoff']*100:.2f}%"
    )
    results["local_auc"] = local_auc_res

    # if the tail were made of trivially short sequences the signal would be an
    # artefact rather than memorisation, so we compare lengths inside and outside it
    tail_mask_train = train_scores_arr >= local_auc_res["cutoff"]
    tail_mask_holdout = holdout_scores_arr >= local_auc_res["cutoff"]
    print(f"\nMean sequence length (trivial-sequence check in the tail):")
    print(
        f"  Train, tail:   {train_seq_lengths[tail_mask_train].mean():.1f} | Train, rest: {train_seq_lengths[~tail_mask_train].mean():.1f}"
    )
    print(
        f"  Holdout, tail: {holdout_seq_lengths[tail_mask_holdout].mean():.1f} | Holdout, rest: {holdout_seq_lengths[~tail_mask_holdout].mean():.1f}"
    )

    train_scores_patient = _aggregate_by_patient(
        train_ids, train_centers, train_scores_arr, method=chunk_aggregation
    )
    holdout_scores_patient = _aggregate_by_patient(
        holdout_ids, holdout_centers, holdout_scores_arr, method=chunk_aggregation
    )
    s = np.concatenate(
        [list(train_scores_patient.values()), list(holdout_scores_patient.values())]
    )
    l = np.concatenate(
        [np.ones(len(train_scores_patient)), np.zeros(len(holdout_scores_patient))]
    )
    patient_level_auc = roc_auc_score(l, s)
    mean_auc_p, ci_lo_p, ci_hi_p = bootstrap_auc_scores(
        list(train_scores_patient.values()), list(holdout_scores_patient.values())
    )
    print(
        f"\nPatient-level AUC: {patient_level_auc:.3f} [{ci_lo_p:.3f}, {ci_hi_p:.3f}] "
        f"(n_train={len(train_scores_patient)}, n_holdout={len(holdout_scores_patient)})"
    )
    results["patient_level_auc"] = patient_level_auc
    results["patient_level_auc_ci"] = (ci_lo_p, ci_hi_p)

    def _by_patient(ids, centers, arr):
        # last chunk wins; fine for metadata that is constant across a patient
        buckets = {}
        for sid, cid, v in zip(ids, centers, arr):
            buckets[_composite_key(sid, cid)] = (
                v.tolist() if hasattr(v, "tolist") else v
            )
        return buckets

    results["train_scores_best_k"] = train_scores_patient
    results["holdout_scores_best_k"] = holdout_scores_patient
    results["train_statics"] = _by_patient(train_ids, train_centers, train_statics)
    results["holdout_statics"] = _by_patient(
        holdout_ids, holdout_centers, holdout_statics
    )
    results["train_classes"] = {
        k: int(v)
        for k, v in _by_patient(train_ids, train_centers, train_classes).items()
    }
    results["holdout_classes"] = {
        k: int(v)
        for k, v in _by_patient(holdout_ids, holdout_centers, holdout_classes).items()
    }
    results["k_used_for_patient_scores"] = k_dl
    results["best_k_by_aggregate_auc"] = best_k_argmax

    results["_raw"] = {
        "train_scores_arr": train_scores_arr,
        "holdout_scores_arr": holdout_scores_arr,
        "train_seq_lengths": train_seq_lengths,
        "holdout_seq_lengths": holdout_seq_lengths,
        "train_centers": train_centers,
        "holdout_centers": holdout_centers,
        "train_classes": train_classes,
        "holdout_classes": holdout_classes,
    }
    return results


# Edit distance utilities
def edit_distance_matrix(
    queries, choices, workers=-1, chunk_size=500, show_progress=False
):
    # computed in blocks so the full matrix never has to fit in memory at once
    if len(queries) == 0 or len(choices) == 0:
        return np.zeros((len(queries), len(choices)))
    chunks = []
    ranger = range(0, len(queries), chunk_size)
    if show_progress:
        ranger = tqdm(list(ranger), desc="Distance matrix", unit="chunk")
    for start in ranger:
        q_chunk = queries[start : start + chunk_size]
        chunks.append(
            process.cdist(
                q_chunk,
                choices,
                scorer=Levenshtein.normalized_distance,
                workers=workers,
            )
        )
    return np.vstack(chunks)


def sequence_duplicate_rate(
    generated,
    real_pool,
    threshold=0.05,
    max_real_candidates=2000,
    rng_seed=0,
    workers=-1,
):
    # fraction of synthetics whose nearest real neighbour falls below threshold
    rng = np.random.default_rng(rng_seed)
    if len(real_pool) > max_real_candidates:
        idx = rng.choice(len(real_pool), max_real_candidates, replace=False)
        real_pool = [real_pool[i] for i in idx]
    dist_matrix = edit_distance_matrix(generated, real_pool, workers=workers)
    return float(np.mean(dist_matrix.min(axis=1) < threshold))


def _tokens_to_strings(token_ids_row, id_to_token, pad_id, sos_ids):
    # drops padding and the start marker, keeps GAP and EOS as content
    return [
        id_to_token[int(t)]
        for t in token_ids_row
        if int(t) != pad_id and int(t) not in sos_ids
    ]


def collect_real_sequences_with_ids(
    model,
    subset,
    batch_size=32,
    max_batches=None,
    collect_statics=True,
    collect_labels=False,
    collect_centers=False,
    shuffle=None,
):
    if hasattr(model, "collect_privacy_sequences"):
        return model.collect_privacy_sequences(
            subset,
            batch_size,
            max_batches,
            collect_statics,
            collect_labels,
            collect_centers,
            shuffle,
        )
    dm = model._datamodule
    # when only part of the data is read, shuffle so the sample is not just the
    # first N records of the loader
    if shuffle is None:
        shuffle = max_batches is not None
    if shuffle:
        # exact_batch_keys maps loader rows to subset.indices by position, which a
        # shuffled loader breaks
        raise ValueError(
            "collect_real_sequences_with_ids keys rows by position and cannot shuffle. "
            "Pass max_batches=None, or shuffle=False to read the first max_batches batches."
        )
    loader = make_eval_loader(dm, subset, batch_size, shuffle=False)

    id_to_token = {v: k for k, v in dm.tokenizer.vocab.items()}
    pad_id = dm.tokenizer.vocab["<PAD>"]
    sos_ids = {v for k, v in dm.tokenizer.vocab.items() if k.startswith("<SOS")}

    seqs, stay_ids, statics, labels, centers = [], [], [], [], []
    total_b = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress_bar = tqdm(
        enumerate(loader),
        total=total_b,
        desc="Extracting token streams + IDs",
        unit="batch",
        leave=False,
    )

    offset = 0
    for b_idx, batch in progress_bar:
        if max_batches is not None and b_idx >= max_batches:
            break
        X, _, _, S, IDs = _unpack_batch(batch)
        exact_ids, exact_centers = exact_batch_keys(subset, offset, IDs)
        offset += IDs.shape[0]
        S_np = S.numpy() if torch.is_tensor(S) else np.asarray(S)
        IDs_np = IDs.numpy() if torch.is_tensor(IDs) else np.asarray(IDs)
        for idx, row in enumerate(X.numpy()):
            seqs.append(_tokens_to_strings(row, id_to_token, pad_id, sos_ids))
            stay_ids.append(int(exact_ids[idx]))
            if collect_statics:
                statics.append(S_np[idx])
            if collect_labels:
                labels.append(int(IDs_np[idx, 1]))
            if collect_centers:
                centers.append(int(exact_centers[idx]))

    out = [seqs, stay_ids]
    if collect_statics:
        out.append(statics)
    if collect_labels:
        out.append(labels)
    if collect_centers:
        out.append(centers)
    return tuple(out)


def collect_real_sequences(model, subset, batch_size=32, max_batches=None):
    seqs, _stay_ids = collect_real_sequences_with_ids(
        model,
        subset,
        batch_size=batch_size,
        max_batches=max_batches,
        collect_statics=False,
        collect_labels=False,
    )
    return seqs


def real_real_baseline_distance(
    query_seqs, candidate_pool_seqs, max_candidates=2000, seed=0
):
    """Baseline: holdout-vs-train distance (real-real, no identity relationship)."""
    # sets the floor of similarity that comes from the domain itself, shared
    # protocols, finite vocabulary and a common hourly grid
    rng = np.random.default_rng(seed)
    candidates = candidate_pool_seqs
    if len(candidates) > max_candidates:
        idx = rng.choice(len(candidates), max_candidates, replace=False)
        candidates = [candidates[i] for i in idx]
    dist_matrix = edit_distance_matrix(query_seqs, candidates, show_progress=True)
    return dist_matrix.min(axis=1)


# B) Replication
# Legacy: reads a capped sample through collect_real_sequences_with_ids, which now
# raises on the shuffled path it relies on.
def run_replication_audit(
    model,
    generated_samples_path: str,
    n_generated_eval: int = 500,
    n_real_train_eval: int = 2000,
    n_real_holdout_eval: int = 2000,
    threshold: float = 0.05,
    batch_size: int = 32,
    restrict_to_same_center: str = "auto",
    threshold_sweep=None,
):
    # measures verbatim copying, independent of any notion of membership
    import joblib

    generated_samples, generated_labels, generated_centers = joblib.load(
        generated_samples_path
    )
    gen_seqs = [list(seq) for seq in generated_samples]
    gen_centers_full = np.asarray(
        generated_centers.cpu().numpy()
        if torch.is_tensor(generated_centers)
        else generated_centers
    ).astype(int)

    rng = np.random.default_rng(0)
    if len(gen_seqs) > n_generated_eval:
        idx = rng.choice(len(gen_seqs), n_generated_eval, replace=False)
        gen_seqs = [gen_seqs[i] for i in idx]
        gen_centers_arr = gen_centers_full[idx]
    else:
        gen_centers_arr = gen_centers_full

    print("Collecting real TRAIN sequences...")
    real_train_seqs, _real_train_ids, real_train_centers = (
        collect_real_sequences_with_ids(
            model,
            model._datamodule.train_icu_dataset,
            batch_size=batch_size,
            max_batches=max(1, n_real_train_eval // batch_size),
            collect_statics=False,
            collect_labels=False,
            collect_centers=True,
        )
    )
    print("Collecting real HOLDOUT sequences...")
    real_holdout_seqs, _real_holdout_ids, real_holdout_centers = (
        collect_real_sequences_with_ids(
            model,
            model._datamodule.test_icu_dataset,
            batch_size=batch_size,
            max_batches=max(1, n_real_holdout_eval // batch_size),
            collect_statics=False,
            collect_labels=False,
            collect_centers=True,
        )
    )
    real_train_centers = np.asarray(real_train_centers)
    real_holdout_centers = np.asarray(real_holdout_centers)

    # sanity check on token representation: if synthetic and real sequences did
    # not share a vocabulary every edit distance would be close to 1 and the null
    # result would be a formatting artefact rather than evidence of privacy
    gen_seq0 = list(generated_samples[0])
    real_seq0 = real_train_seqs[0]

    print("--- format inspection ---")
    print(f"Synthetic - element type: {type(gen_seq0[0])} | sample: {gen_seq0[:5]}")
    print(f"Real      - element type: {type(real_seq0[0])} | sample: {real_seq0[:5]}")
    print(f"Lengths -> synthetic: {len(gen_seq0)} | real: {len(real_seq0)}")

    vocab_gen = set(token for seq in generated_samples[:50] for token in seq)
    vocab_real = set(token for seq in real_train_seqs[:50] for token in seq)
    overlap = vocab_gen.intersection(vocab_real)

    print(f"Synthetic vocab size: {len(vocab_gen)}")
    print(f"Real vocab size: {len(vocab_real)}")
    print(f"Tokens shared by both: {len(overlap)}")

    print(
        f"\nComparing {len(gen_seqs)} synthetic vs {len(real_train_seqs)} train / {len(real_holdout_seqs)} holdout..."
    )
    dup_train = sequence_duplicate_rate(
        gen_seqs,
        real_train_seqs,
        threshold=threshold,
        max_real_candidates=n_real_train_eval,
    )
    dup_holdout = sequence_duplicate_rate(
        gen_seqs,
        real_holdout_seqs,
        threshold=threshold,
        max_real_candidates=n_real_holdout_eval,
    )
    # if copying were memorisation the rate against train should exceed the rate
    # against holdout; matching rates mean we are measuring domain similarity
    print(f"  near-duplicate vs TRAIN:   {dup_train:.4f}")
    print(f"  near-duplicate vs HOLDOUT: {dup_holdout:.4f}")

    print("\nComputing REAL-REAL baseline distance (holdout vs train)...")
    real_real_dists = real_real_baseline_distance(
        real_holdout_seqs[:2000], real_train_seqs, max_candidates=n_real_train_eval
    )

    print("\nComputing full synth-vs-real distance distributions...")
    dist_syn_train = edit_distance_matrix(
        gen_seqs, real_train_seqs[:n_real_train_eval], show_progress=True
    ).min(axis=1)
    dist_syn_holdout = edit_distance_matrix(
        gen_seqs, real_holdout_seqs[:n_real_holdout_eval], show_progress=True
    ).min(axis=1)

    # reporting the whole curve avoids depending on a single arbitrary threshold
    thr_results = {}
    if threshold_sweep is not None:
        thr_results = {
            "thresholds": np.asarray(threshold_sweep),
            "dup_rate_train": np.array(
                [(dist_syn_train < t).mean() for t in threshold_sweep]
            ),
            "dup_rate_holdout": np.array(
                [(dist_syn_holdout < t).mean() for t in threshold_sweep]
            ),
            "dup_rate_realreal": np.array(
                [(real_real_dists < t).mean() for t in threshold_sweep]
            ),
        }

    print("\nChecking for center-conditioning signal...")
    all_real_centers = np.concatenate([real_train_centers, real_holdout_centers])
    use_center = (
        detect_center_conditioning(gen_centers_arr, all_real_centers)
        if restrict_to_same_center == "auto"
        else bool(restrict_to_same_center)
    )

    # same-centre comparison is the regime where copying would be most likely
    dup_train_by_center, dup_holdout_by_center = {}, {}
    if use_center:
        gen_conditioned_mask = gen_centers_arr != NULL_CENTER_LABEL
        print(
            "\nDuplicate rate restricted to same-center (conditioned synthetics only):"
        )
        for centre in np.unique(all_real_centers):
            gen_idx = np.where(gen_conditioned_mask & (gen_centers_arr == centre))[0]
            train_idx = np.where(real_train_centers == centre)[0]
            holdout_idx = np.where(real_holdout_centers == centre)[0]
            if len(gen_idx) < 10 or len(train_idx) < 10 or len(holdout_idx) < 10:
                continue
            gen_c = [gen_seqs[i] for i in gen_idx]
            train_c = [real_train_seqs[i] for i in train_idx]
            holdout_c = [real_holdout_seqs[i] for i in holdout_idx]
            d_t = sequence_duplicate_rate(
                gen_c, train_c, threshold=threshold, max_real_candidates=len(train_c)
            )
            d_h = sequence_duplicate_rate(
                gen_c,
                holdout_c,
                threshold=threshold,
                max_real_candidates=len(holdout_c),
            )
            dup_train_by_center[int(centre)] = d_t
            dup_holdout_by_center[int(centre)] = d_h
            print(f"  centre={centre}: dup_train={d_t:.4f}  dup_holdout={d_h:.4f}")

    return {
        "dup_vs_train": dup_train,
        "dup_vs_holdout": dup_holdout,
        "dup_vs_train_by_center": dup_train_by_center,
        "dup_vs_holdout_by_center": dup_holdout_by_center,
        "center_restricted": use_center,
        "n_generated_evaluated": len(gen_seqs),
        "_raw": {
            "dist_syn_train": dist_syn_train,
            "dist_syn_holdout": dist_syn_holdout,
            "real_real_dists": real_real_dists,
            "threshold_sweep": thr_results,
        },
    }


# C) Generation-based MIA
# Legacy: superseded by privacy_audit_ext.run_generation_mia_v2. Its ids come from
# collect_real_sequences_with_ids, so they are exact, but the capped read (neither
# collect_full nor restrict_to_patient_keys) raises on the shuffled path.
def run_generation_mia_audit(
    model,
    generated_samples_path,
    n_generated_eval: int = 500,
    n_real_train_eval: int = 100,
    n_real_holdout_eval: int = 100,
    batch_size: int = 32,
    restrict_to_same_class: bool = True,
    restrict_to_same_center: str = "auto",
    chunk_aggregation: str = "min",
    n_per_center: int = None,
    restrict_to_patient_keys: dict = None,
    collect_full: bool = False,  # force max_batches=None regardless of restrict_to_patient_keys
    local_auc_percentile: float = 95,
):
    # the data-release scenario: the adversary only sees the synthetic corpus,
    # so membership is predicted from proximity to it
    import joblib

    generated_samples, generated_labels, generated_centers = joblib.load(
        generated_samples_path
    )
    gen_seqs = [list(seq) for seq in generated_samples]
    gen_labels = np.asarray(
        generated_labels.cpu().numpy()
        if torch.is_tensor(generated_labels)
        else generated_labels
    ).astype(int)
    gen_centers_full = np.asarray(
        generated_centers.cpu().numpy()
        if torch.is_tensor(generated_centers)
        else generated_centers
    ).astype(int)

    rng = np.random.default_rng(0)
    if len(gen_seqs) > n_generated_eval:
        idx = rng.choice(len(gen_seqs), n_generated_eval, replace=False)
        gen_seqs = [gen_seqs[i] for i in idx]
        gen_labels = gen_labels[idx]
        gen_centers_arr = gen_centers_full[idx]
    else:
        gen_centers_arr = gen_centers_full

    # When restricting to fixed keys, we must collect the full dataset before filtering
    # beforehand only sees the first N records of a non-shuffled loader,
    # which almost never covers all patients in the fixed key set. The
    # expensive step (edit-distance computation) still only runs on the
    # filtered, small cohort.
    force_full = collect_full or (restrict_to_patient_keys is not None)
    train_max_batches = None if force_full else max(1, n_real_train_eval // batch_size)
    holdout_max_batches = (
        None if force_full else max(1, n_real_holdout_eval // batch_size)
    )

    print("Collecting real TRAIN sequences...")
    real_train_seqs, train_ids, train_statics, train_classes, train_centers = (
        collect_real_sequences_with_ids(
            model,
            model._datamodule.train_icu_dataset,
            batch_size=batch_size,
            max_batches=train_max_batches,
            collect_statics=True,
            collect_labels=True,
            collect_centers=True,
        )
    )
    train_ids = np.array(train_ids)
    train_statics = np.array(train_statics)
    train_classes, train_centers = np.array(train_classes), np.array(train_centers)

    print("Collecting real HOLDOUT sequences...")
    (
        real_holdout_seqs,
        holdout_ids,
        holdout_statics,
        holdout_classes,
        holdout_centers,
    ) = collect_real_sequences_with_ids(
        model,
        model._datamodule.test_icu_dataset,
        batch_size=batch_size,
        max_batches=holdout_max_batches,
        collect_statics=True,
        collect_labels=True,
        collect_centers=True,
    )
    holdout_ids = np.array(holdout_ids)
    holdout_statics = np.array(holdout_statics)
    holdout_classes, holdout_centers = np.array(holdout_classes), np.array(
        holdout_centers
    )

    if restrict_to_patient_keys is not None:
        print("\nFiltering to FIXED stratified patient set (shared across attacks)...")
        train_keys_set = restrict_to_patient_keys["train"]
        holdout_keys_set = restrict_to_patient_keys["holdout"]

        train_mask = np.array(
            [
                _composite_key(s, c) in train_keys_set
                for s, c in zip(train_ids, train_centers)
            ]
        )
        real_train_seqs = [real_train_seqs[i] for i in np.where(train_mask)[0]]
        train_ids, train_statics = train_ids[train_mask], train_statics[train_mask]
        train_classes, train_centers = (
            train_classes[train_mask],
            train_centers[train_mask],
        )

        holdout_mask = np.array(
            [
                _composite_key(s, c) in holdout_keys_set
                for s, c in zip(holdout_ids, holdout_centers)
            ]
        )
        real_holdout_seqs = [real_holdout_seqs[i] for i in np.where(holdout_mask)[0]]
        holdout_ids, holdout_statics = (
            holdout_ids[holdout_mask],
            holdout_statics[holdout_mask],
        )
        holdout_classes, holdout_centers = (
            holdout_classes[holdout_mask],
            holdout_centers[holdout_mask],
        )
        print(
            f"  Post-filter: n_train={len(real_train_seqs)}, n_holdout={len(real_holdout_seqs)}"
        )

    elif n_per_center is not None:
        print(f"\nStratifying by center: {n_per_center} per center")
        train_dict = {
            "seqs": real_train_seqs,
            "ids": train_ids,
            "statics": train_statics,
            "classes": train_classes,
            "centers": train_centers,
        }
        train_dict, _ = stratified_subsample_by_center(
            train_dict, train_centers, n_per_center, seed=0
        )
        real_train_seqs, train_ids, train_statics = (
            train_dict["seqs"],
            train_dict["ids"],
            train_dict["statics"],
        )
        train_classes, train_centers = train_dict["classes"], train_dict["centers"]

        holdout_dict = {
            "seqs": real_holdout_seqs,
            "ids": holdout_ids,
            "statics": holdout_statics,
            "classes": holdout_classes,
            "centers": holdout_centers,
        }
        holdout_dict, _ = stratified_subsample_by_center(
            holdout_dict, holdout_centers, n_per_center, seed=0
        )
        real_holdout_seqs, holdout_ids, holdout_statics = (
            holdout_dict["seqs"],
            holdout_dict["ids"],
            holdout_dict["statics"],
        )
        holdout_classes, holdout_centers = (
            holdout_dict["classes"],
            holdout_dict["centers"],
        )
        print(
            f"  Post-stratification: n_train={len(real_train_seqs)}, n_holdout={len(real_holdout_seqs)}"
        )

    print("\nChecking for center-conditioning signal in generated data...")
    all_real_centers = np.concatenate([train_centers, holdout_centers])
    use_center = (
        detect_center_conditioning(gen_centers_arr, all_real_centers)
        if restrict_to_same_center == "auto"
        else bool(restrict_to_same_center)
    )

    def compute_scores(real_seqs, real_classes, real_centers_arr):
        # best case adversary: knows outcome and centre and only compares inside
        # that cell, which removes the component of marginal distribution matching
        scores = np.zeros(len(real_seqs))
        if not restrict_to_same_class:
            return edit_distance_matrix(real_seqs, gen_seqs, show_progress=True).min(
                axis=1
            )

        gen_conditioned_mask = gen_centers_arr != NULL_CENTER_LABEL
        for cls in np.unique(real_classes):
            centers_to_iterate = (
                np.unique(real_centers_arr[real_classes == cls])
                if use_center
                else [None]
            )
            for centre in centers_to_iterate:
                if use_center:
                    idx = np.where(
                        (real_classes == cls) & (real_centers_arr == centre)
                    )[0]
                    cls_gen_idx = np.where(
                        (gen_labels == cls)
                        & gen_conditioned_mask
                        & (gen_centers_arr == centre)
                    )[0]
                else:
                    idx = np.where(real_classes == cls)[0]
                    cls_gen_idx = np.where(gen_labels == cls)[0]
                if len(idx) == 0:
                    continue
                if len(cls_gen_idx) == 0:
                    # empty cell, fall back to the whole class pool and record it
                    fallback_idx = np.where(gen_labels == cls)[0]
                    candidate_seqs = (
                        [gen_seqs[i] for i in fallback_idx]
                        if len(fallback_idx) > 0
                        else gen_seqs
                    )
                    print(
                        f"  WARNING: no candidates for class={cls}, centre={centre}; using full class pool."
                    )
                else:
                    candidate_seqs = [gen_seqs[i] for i in cls_gen_idx]
                sub_seqs = [real_seqs[i] for i in idx]
                dist_matrix = edit_distance_matrix(
                    sub_seqs, candidate_seqs, show_progress=True
                )
                scores[idx] = dist_matrix.min(axis=1)
        return scores

    print(
        f"\nAttacking (restrict_class={restrict_to_same_class}, restrict_center={use_center})..."
    )
    train_scores = compute_scores(real_train_seqs, train_classes, train_centers)
    holdout_scores = compute_scores(real_holdout_seqs, holdout_classes, holdout_centers)

    # negated: membership is predicted when the distance is small
    s = np.concatenate([-train_scores, -holdout_scores])
    l = np.concatenate([np.ones(len(train_scores)), np.zeros(len(holdout_scores))])
    mia_auc = roc_auc_score(l, s)

    print(f"\nAttack AUC (chunk-level): {mia_auc:.3f}")
    for cls in np.unique(np.concatenate([train_classes, holdout_classes])):
        t_idx, h_idx = train_classes == cls, holdout_classes == cls
        if t_idx.sum() < 10 or h_idx.sum() < 10:
            continue
        ss = np.concatenate([-train_scores[t_idx], -holdout_scores[h_idx]])
        ll = np.concatenate([np.ones(t_idx.sum()), np.zeros(h_idx.sum())])
        print(f"  class={cls}: AUC = {roc_auc_score(ll, ss):.3f}")

    local_auc_res = compute_local_auc(
        -train_scores, -holdout_scores, percentile=local_auc_percentile
    )
    print(
        f"Local AUC (P{local_auc_percentile}+): {local_auc_res['local_auc']:.3f} "
        f"[{local_auc_res['local_auc_ci'][0]:.3f}, {local_auc_res['local_auc_ci'][1]:.3f}]"
    )

    train_dist_patient = _aggregate_by_patient(
        train_ids, train_centers, train_scores, method=chunk_aggregation
    )
    holdout_dist_patient = _aggregate_by_patient(
        holdout_ids, holdout_centers, holdout_scores, method=chunk_aggregation
    )
    s = np.concatenate(
        [
            -np.array(list(train_dist_patient.values())),
            -np.array(list(holdout_dist_patient.values())),
        ]
    )
    l = np.concatenate(
        [np.ones(len(train_dist_patient)), np.zeros(len(holdout_dist_patient))]
    )
    patient_level_auc = roc_auc_score(l, s)
    print(
        f"Patient-level AUC: {patient_level_auc:.3f} (n_train={len(train_dist_patient)}, n_holdout={len(holdout_dist_patient)})"
    )

    def _by_patient(ids, centers, arr):
        return {
            _composite_key(sid, cid): (v.tolist() if hasattr(v, "tolist") else v)
            for sid, cid, v in zip(ids, centers, arr)
        }

    return {
        "mia_auc": mia_auc,
        "patient_level_auc": patient_level_auc,
        "center_restricted": use_center,
        "local_auc": local_auc_res,
        "train_patient_risks": train_dist_patient,
        "holdout_patient_risks": holdout_dist_patient,
        "train_statics": _by_patient(train_ids, train_centers, train_statics),
        "holdout_statics": _by_patient(holdout_ids, holdout_centers, holdout_statics),
        "train_classes": {
            k: int(v)
            for k, v in _by_patient(train_ids, train_centers, train_classes).items()
        },
        "holdout_classes": {
            k: int(v)
            for k, v in _by_patient(
                holdout_ids, holdout_centers, holdout_classes
            ).items()
        },
        "_raw": {"train_scores": train_scores, "holdout_scores": holdout_scores},
    }


# D) Patient-level risk
def per_patient_percentile_risk(member_scores, holdout_scores, higher_is_riskier=True):
    # puts the two attacks on a common scale: r(x) is the false positive rate the
    # adversary would have to accept in order to flag this patient
    holdout_vals = np.array(list(holdout_scores.values()), dtype=float)
    n_holdout = len(holdout_vals)
    holdout_sorted = np.sort(holdout_vals if higher_is_riskier else -holdout_vals)
    risks = {}
    for pid, s in member_scores.items():
        val = s if higher_is_riskier else -s
        risks[pid] = np.searchsorted(holdout_sorted, val, side="left") / n_holdout
    return risks


def bootstrap_risk_ci(
    member_scores, holdout_scores, higher_is_riskier=True, n_boot=500, ci=0.95, seed=0
):
    # the percentile is estimated from a finite holdout, so each individual risk
    # gets its own interval
    rng = np.random.default_rng(seed)
    holdout_vals = np.array(list(holdout_scores.values()), dtype=float)
    n_holdout = len(holdout_vals)
    alpha = (1 - ci) / 2
    boot_risks = {pid: [] for pid in member_scores}
    for _ in range(n_boot):
        resampled = rng.choice(holdout_vals, size=n_holdout, replace=True)
        resampled_sorted = np.sort(resampled if higher_is_riskier else -resampled)
        for pid, s in member_scores.items():
            val = s if higher_is_riskier else -s
            boot_risks[pid].append(
                np.searchsorted(resampled_sorted, val, side="left") / n_holdout
            )
    return {
        pid: {
            "point": float(np.mean(v)),
            "ci_low": float(np.percentile(v, 100 * alpha)),
            "ci_high": float(np.percentile(v, 100 * (1 - alpha))),
        }
        for pid, v in boot_risks.items()
    }


def empirical_survival_function(risks):
    vals = np.sort(np.array(list(risks.values())))[::-1]
    n = len(vals)
    return vals, np.arange(1, n + 1) / n


def flag_high_risk_patients(risks, percentile=99.0):
    vals = np.array(list(risks.values()))
    threshold = np.percentile(vals, percentile)
    flagged = {pid: r for pid, r in risks.items() if r >= threshold}
    return {
        "threshold": float(threshold),
        "n_flagged": len(flagged),
        "flagged": flagged,
    }


def combine_risk_signals(risk_dicts, method="max"):
    # max means a patient is at risk if either channel exposes them; the
    # intersection of keys is why the shared cohort matters
    common_ids = set(risk_dicts[0].keys())
    for rd in risk_dicts[1:]:
        common_ids &= set(rd.keys())
    combined = {}
    for pid in common_ids:
        vals = [rd[pid] for rd in risk_dicts]
        combined[pid] = max(vals) if method == "max" else float(np.mean(vals))
    return combined


# E) Subgroup composition analysis
def subgroup_composition_test(
    values_by_id, flagged_ids, all_ids, n_bins=4, is_categorical=False, bin_labels=None
):
    # compares the composition of the risk tail against the composition of the
    # whole cohort under the null of no disparity
    all_ids = [i for i in all_ids if i in values_by_id]
    vals = np.array([values_by_id[i] for i in all_ids], dtype=float)
    n_unique = len(np.unique(vals))
    if not is_categorical and n_unique <= n_bins:
        is_categorical = True

    if not is_categorical:
        edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        bin_idx = np.clip(np.digitize(vals, edges) - 1, 0, n_bins - 1)
        cats = bin_labels or [f"Q{i+1}" for i in range(n_bins)]
        cat_of_id = {i: cats[b] for i, b in zip(all_ids, bin_idx)}
    else:
        rounded_vals = np.round(vals, 6)
        cat_of_id = {i: v for i, v in zip(all_ids, rounded_vals)}
        cats = sorted(set(cat_of_id.values()))

    flagged_set = set(flagged_ids) & set(all_ids)
    n_flagged, n_total = len(flagged_set), len(all_ids)
    overall_counts = np.array([sum(cat_of_id[i] == c for i in all_ids) for c in cats])
    overall_props = overall_counts / n_total
    expected_flagged = overall_props * n_flagged
    observed_flagged = np.array(
        [sum(cat_of_id[i] == c for i in flagged_set) for c in cats]
    )

    nonzero = expected_flagged > 0
    obs_f, exp_f = observed_flagged[nonzero], expected_flagged[nonzero]
    chi2, p = chisquare(obs_f, f_exp=exp_f)
    # residuals and relative change carry the direction, which the p value does not
    pearson_residuals = (obs_f - exp_f) / np.sqrt(exp_f)
    relative_change_pct = (obs_f - exp_f) / exp_f * 100

    return {
        "categories": [c for c, keep in zip(cats, nonzero) if keep],
        "observed_flagged": obs_f,
        "expected_flagged": exp_f,
        "pearson_residuals": pearson_residuals,
        "relative_change_pct": relative_change_pct,
        "chi2": float(chi2),
        "p_value": float(p),
        "n_flagged": n_flagged,
        "n_total": n_total,
    }


def run_full_subgroup_analysis(
    risks,
    statics_by_id,
    statics_columns=STATICS_COLUMNS,
    percentile=99.0,
    n_bins=4,
    bonferroni_correct=True,
):
    tail = flag_high_risk_patients(risks, percentile=percentile)
    flagged_ids = set(tail["flagged"].keys())
    all_ids = [pid for pid in risks if pid in statics_by_id]
    n_tests = len(statics_columns)

    print(
        f"Extreme-risk tail: {tail['n_flagged']} patients (risk >= {tail['threshold']:.3f}) out of {len(all_ids)}"
    )

    results = {}
    for col_idx, col_name in enumerate(statics_columns):
        values_by_id = {pid: statics_by_id[pid][col_idx] for pid in all_ids}
        is_categorical = col_name.lower().startswith(("female", "sex", "male"))
        res = subgroup_composition_test(
            values_by_id,
            flagged_ids,
            all_ids,
            n_bins=n_bins,
            is_categorical=is_categorical,
        )
        if bonferroni_correct:
            res["p_value_bonferroni"] = min(1.0, res["p_value"] * n_tests)
        results[col_name] = res
        p_report = res.get("p_value_bonferroni", res["p_value"])
        sig = (
            "***"
            if p_report < 0.001
            else "**" if p_report < 0.01 else "*" if p_report < 0.05 else "n.s."
        )
        print(f"\n{col_name}: chi2={res['chi2']:.2f}, p={p_report:.2e} [{sig}]")
        for cat, resid, pct in zip(
            res["categories"], res["pearson_residuals"], res["relative_change_pct"]
        ):
            print(f"    {cat}: residual={resid:+.2f}, change={pct:+.1f}%")

    return {"tail": tail, "subgroup_results": results}


def run_center_controlled_subgroup_check(
    combined_risk,
    statics_by_id,
    target_center,
    statics_columns=STATICS_COLUMNS,
    percentile=99.0,
):
    """Repeats the subgroup test restricted to a single center, to rule out confounding."""
    # note that losing significance here can also be loss of power from the
    # smaller n, which is why the stratified test in privacy_audit_ext is preferred
    center_ids = {
        pid: v
        for pid, v in combined_risk.items()
        if pid.startswith(f"{target_center}_")
    }
    center_statics = {
        pid: v
        for pid, v in statics_by_id.items()
        if pid.startswith(f"{target_center}_")
    }
    print(
        f"\n--- Subgroup check within center {target_center} only (n={len(center_ids)}) ---"
    )
    return run_full_subgroup_analysis(
        center_ids,
        center_statics,
        statics_columns=statics_columns,
        percentile=percentile,
    )
