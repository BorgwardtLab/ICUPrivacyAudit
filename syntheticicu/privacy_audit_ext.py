"""Patient-level likelihood, conditioning-reference and corpus privacy audits."""

from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import chi2, ks_2samp
from tqdm import tqdm

from syntheticicu.privacy_audit import (
    NULL_CENTER_LABEL,
    _composite_key,
    _aggregate_by_patient,
    _unpack_batch,
    exact_batch_keys,
    min_k_percent_prob,
    mean_logp,
    make_eval_loader,
    edit_distance_matrix,
    bootstrap_auc_scores,
    compute_local_auc,
    flag_high_risk_patients,
    subgroup_composition_test,
    combine_risk_signals,
)

GAP_TOKEN = "<GAP>"
EOS_TOKEN = "<EOS>"
DEMOGRAPHIC_ORDER = ["female_static", "age_static", "height_static", "weight_static"]


class StaticsDecoder:
    """
    Converts the `statics` tensor coming out of the dataloader into an (N, 4)
    matrix of bin indices in canonical order [female, age, height, weight].

    Supports the two representations `data_classes.py` can produce:
      - (N, sum of bins) one-hot, produced by `_direct_statics_to_onehot`
        (17 for 2/5/5/5)
      - (N, 4) direct indices (`processed_statics`)

    The order of the one-hot blocks is inferred from `static_names`, which is the
    order `_direct_statics_to_onehot` actually used, not the order of
    `demographic_features`. It is never assumed.
    """

    def __init__(self, model=None, static_names=None, demographic_bins=None):
        # the bins the tokenizer and _direct_statics_to_onehot used, which follow
        # the dataset (2/5/5/5 for a 5-bin dataset)
        if demographic_bins is None and model is not None:
            tokenizer = getattr(model._datamodule, "tokenizer", None)
            demographic_bins = getattr(tokenizer, "demographic_bins", None)
        if demographic_bins is None:
            demographic_bins = {
                "female_static": 2,
                "age_static": 5,
                "height_static": 5,
                "weight_static": 5,
            }
        self.demographic_bins = dict(demographic_bins)

        if static_names is None and model is not None:
            base = model._datamodule.base_icu_dataset
            static_names = getattr(base, "static_names", None)
            if static_names is None or len(static_names) == 0:
                static_names = getattr(base, "statics_set", None)

        # static_names is usually an np.ndarray, so no `or` and no boolean
        # evaluation: numpy raises ValueError with more than one element
        if static_names is None or len(static_names) == 0:
            static_names = DEMOGRAPHIC_ORDER
        self.static_names = [str(n) for n in np.asarray(static_names).ravel().tolist()]

        self.block_order = self._infer_block_order(self.static_names)
        self.expected_onehot_width = sum(
            self.demographic_bins[f] for f in self.block_order
        )

        self.offsets = {}
        off = 0
        for feature in self.block_order:
            self.offsets[feature] = off
            off += self.demographic_bins[feature]

    def _infer_block_order(self, static_names):
        """Order of the one-hot blocks, inferred by first appearance."""
        order = []
        for name in static_names:
            matched = next(
                (f for f in self.demographic_bins if name == f or name.startswith(f)),
                None,
            )
            if matched is not None and matched not in order:
                order.append(matched)
        missing = [f for f in self.demographic_bins if f not in order]
        if missing:
            # static_names did not cover everything, fall back to the documented
            # tokenizer order
            order = order + missing
        return order

    def describe(self):
        lines = [
            f"  static_names      : {self.static_names}",
            f"  block order       : {self.block_order}",
            f"  one-hot offsets   : {self.offsets}",
            f"  expected width    : {self.expected_onehot_width}",
        ]
        return "\n".join(lines)

    def decode(self, statics_array):
        """
        statics_array: (N, 17) one-hot or (N, 4) indices.
        Returns (N, 4) int, in order [female, age, height, weight].
        """
        S = np.asarray(statics_array)
        if S.ndim == 1:
            S = S[None, :]

        if S.shape[1] == len(self.block_order):
            # already direct indices, but possibly in static_names order
            out = np.zeros((S.shape[0], 4), dtype=int)
            for j, feature in enumerate(self.block_order):
                out[:, DEMOGRAPHIC_ORDER.index(feature)] = S[:, j].astype(int)
            return out

        if S.shape[1] != self.expected_onehot_width:
            raise ValueError(
                f"Unexpected statics width: {S.shape[1]}. Expected "
                f"{self.expected_onehot_width} (one-hot) or {len(self.block_order)} "
                f"(indices). Check `static_names` and `demographic_bins`."
            )

        out = np.zeros((S.shape[0], 4), dtype=int)
        for feature in self.block_order:
            start = self.offsets[feature]
            width = self.demographic_bins[feature]
            block = S[:, start : start + width]
            # argmax is safe: one-hot with median imputation guarantees one active bin
            out[:, DEMOGRAPHIC_ORDER.index(feature)] = block.argmax(axis=1)
        return out

    def sanity_check(self, statics_array, verbose=True):
        """Checks that each decoded column has the right cardinality."""
        # this is what would have caught the original bug: a decoded female
        # column with five bins means the offsets are wrong
        dec = self.decode(statics_array)
        ok = True
        report = []
        for j, feature in enumerate(DEMOGRAPHIC_ORDER):
            n_bins = self.demographic_bins[feature]
            uniq = np.unique(dec[:, j])
            good = uniq.min() >= 0 and uniq.max() < n_bins
            ok &= bool(good)
            report.append(
                f"  {feature:<16} expected bins={n_bins}  observed={sorted(uniq.tolist())}"
                f"  {'OK' if good else 'FAIL'}"
            )
        if verbose:
            print("Sanity check on decoded statics:")
            print("\n".join(report))
        if not ok:
            raise AssertionError("Statics decoding is not consistent.")
        return dec


def decode_statics_dict(statics_by_id, decoder):
    """Applies the decoder to a {patient_key: vector} dict from the original audit."""
    keys = list(statics_by_id.keys())
    if not keys:
        return {}
    arr = np.asarray([statics_by_id[k] for k in keys])
    dec = decoder.decode(arr)
    return {k: dec[i].tolist() for i, k in enumerate(keys)}


# sequence canonicalisation and filtering


def canonicalize_sequence(tokens, sort_within_block=True):
    """
    Sorts the tokens inside each block delimited by <GAP>.

    Reason: the model is trained with `shuffle_tokens_within_timepoint` and
    p_shuffle>0, so the order inside a timepoint carries no information.
    Levenshtein does penalise it, so two timepoints with the same measurements in
    a different order end up far apart. Canonicalising makes the metric invariant
    to the same symmetry the model was trained under.
    """
    if not sort_within_block:
        return list(tokens)
    out, block = [], []
    for t in tokens:
        if t == GAP_TOKEN:
            out.extend(sorted(block))
            out.append(GAP_TOKEN)
            block = []
        else:
            block.append(t)
    out.extend(sorted(block))
    return out


def is_chunk_zero(tokens):
    """
    True if the sequence is an initial chunk (it carries the demographic block).

    `_chunk_long_sequences` produces two different structures: chunk 0
    (SOS + 4 demographic tokens + GAP + content) and continuation chunks
    (content directly, no SOS and no demographics). Comparing a synthetic shaped
    like chunk-0 against a continuation chunk inflates the distance through a
    prefix mismatch, not through a lack of memorisation.
    """
    for t in tokens:
        if isinstance(t, str) and t.startswith("<"):
            continue
        return isinstance(t, str) and "_static_" in t
    return False


def is_complete(tokens):
    """
    False if the sequence never emitted <EOS>, meaning generation was cut off at
    the token limit. Those sequences are artefacts rather than synthetic
    patients, and with a distance normalised by max(len) they have an
    artificially high distance floor.
    """
    return EOS_TOKEN in tokens


def prepare_sequences(
    seqs, canonicalize=True, chunk_zero_only=False, require_eos=False, return_mask=False
):
    """Applies canonicalisation and filters; returns (filtered_seqs, mask)."""
    seqs = [list(s) for s in seqs]
    mask = np.ones(len(seqs), dtype=bool)
    if chunk_zero_only:
        mask &= np.array([is_chunk_zero(s) for s in seqs])
    if require_eos:
        mask &= np.array([is_complete(s) for s in seqs])
    kept = [
        canonicalize_sequence(seqs[i]) if canonicalize else seqs[i]
        for i in np.where(mask)[0]
    ]
    if return_mask:
        return kept, mask
    return kept


def sequence_diagnostics(gen_seqs, real_seqs, label="", max_missing_mass=0.01):
    """
    Diagnostic table that belongs in the paper, not only in the console.

    The vocabulary check runs over the full corpus and the full real set and
    asserts on the share of generated token mass that falls on tokens absent
    from the real set, not on the share of distinct tokens. Counting distinct
    tokens on a 500-sequence slice made the check depend on the bin count:
    finer bins add rare tokens that a small sample misses (K=2 gave 95.2% on
    the slice and 126 vs 126 on the full corpus), while a token emitted once in
    a million cannot move the edit distances.
    """

    def stats(seqs, name):
        lens = np.array([len(s) for s in seqs])
        return {
            "set": name,
            "n": len(seqs),
            "len_mean": float(lens.mean()) if len(lens) else np.nan,
            "len_p50": float(np.percentile(lens, 50)) if len(lens) else np.nan,
            "len_p95": float(np.percentile(lens, 95)) if len(lens) else np.nan,
            "frac_chunk0": (
                float(np.mean([is_chunk_zero(s) for s in seqs])) if seqs else np.nan
            ),
            "frac_no_eos": (
                float(np.mean([not is_complete(s) for s in seqs])) if seqs else np.nan
            ),
        }

    df = pd.DataFrame([stats(gen_seqs, "synthetic"), stats(real_seqs, "real")])
    counts_gen = Counter(t for s in gen_seqs for t in s)
    vocab_real = set(t for s in real_seqs for t in s)
    missing = [t for t in counts_gen if t not in vocab_real]
    missing_mass = sum(counts_gen[t] for t in missing) / max(
        1, sum(counts_gen.values())
    )
    print(f"\n--- Sequence diagnostics {label} ---")
    print(df.to_string(index=False))
    print(
        f"  Vocabulary: synthetic={len(counts_gen)} real={len(vocab_real)} "
        f"overlap={len(counts_gen) - len(missing)} | synthetic-only={len(missing)} "
        f"tokens, {missing_mass:.4%} of the generated mass"
    )
    # without a shared vocabulary every edit distance would sit near 1 and the
    # null result would be a formatting artefact
    assert missing_mass <= max_missing_mass, (
        f"{missing_mass:.2%} of the generated token mass falls on tokens absent from "
        "the real set. Edit distances would be a formatting artefact rather than "
        "evidence of privacy."
    )
    return df


def calibrate_threshold(real_real_dists, percentile=1.0):
    """
    Near-duplicate threshold calibrated from the real-real distribution.

    The 0.05 inherited from the text literature is unreachable in this space: not
    even two distinct real patients ever cross it. A meaningful threshold is a
    low percentile of the real-real distribution, that is, "closer than any two
    real patients typically are".
    """
    tau = float(np.percentile(np.asarray(real_real_dists), percentile))
    print(f"  Calibrated threshold (P{percentile} real-real): tau = {tau:.4f}")
    return tau


# attack metrics


def tpr_at_fpr(labels, scores, fpr_targets=(0.001, 0.01, 0.1)):
    """TPR at low FPR, the canonical MIA metric since Carlini et al. (2022)."""
    # this is the regime an adversary would actually operate in: flagging someone
    # while almost never being wrong
    fpr, tpr, _ = roc_curve(labels, scores)
    return {f"tpr@fpr={f:g}": float(np.interp(f, fpr, tpr)) for f in fpr_targets}


def summarize_attack(
    member_scores,
    nonmember_scores,
    name="attack",
    local_percentile=95,
    n_boot=1000,
    verbose=True,
):
    """Unified summary: AUC with CI, TPR at low FPR, local AUC in the tail."""
    a = np.asarray(member_scores, dtype=float)
    b = np.asarray(nonmember_scores, dtype=float)
    labels = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    scores = np.concatenate([a, b])

    auc = float(roc_auc_score(labels, scores))
    mean_auc, lo, hi = bootstrap_auc_scores(a, b, n_boot=n_boot)
    tprs = tpr_at_fpr(labels, scores)
    try:
        local = compute_local_auc(a, b, percentile=local_percentile, n_boot=n_boot)
    except Exception as exc:  # empty holdout tail or another edge case
        local = {
            "local_auc": float("nan"),
            "local_auc_ci": (float("nan"), float("nan")),
            "error": str(exc),
        }

    out = {
        "name": name,
        "n_members": int(len(a)),
        "n_nonmembers": int(len(b)),
        "auc": auc,
        "auc_ci": (lo, hi),
        "local_auc": local,
        **tprs,
    }
    if verbose:
        print(
            f"  [{name}] AUC={auc:.4f} [{lo:.4f}, {hi:.4f}] | "
            + " | ".join(f"{k}={v:.4f}" for k, v in tprs.items())
            + f" | local AUC(P{local_percentile})={local.get('local_auc', float('nan')):.4f}"
        )
    return out


# attack A, calibrated by conditioning swap
# Stand-in for shadow models that does not require retraining.
#
# Uncalibrated Min-K% scores intrinsically easy sequences highly (short, common
# tokens, typical physiology), not only memorised ones. Shadow models estimate
# that difficulty by retraining. Here we exploit that the model is conditional:
# the same sequence is scored under its true conditioning and under altered
# conditionings, and we take the difference.
#
#   easy sequence         -> scores high under any conditioning  -> diff ~ 0
#   memorised under (y,c) -> scores high only there              -> diff > 0
#
# This is not LiRA (the distribution of scores under non-membership is never
# estimated), but it controls the dominant confounder with three forward passes.


def _forward_token_logp(
    model, X, time_positions, attention_mask, class_labels, center_ids
):
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
    # drop the conditioning prefix so logits[t] lines up with x[t+1]
    T = X.size(1)
    prefix = model.actual_prefix_len
    logits = logits[:, : T + prefix, :]
    shift_logits = logits[:, prefix:-1, :]
    shift_labels = X[:, 1:]
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    return log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)


def _static_token_ids(model):
    # demographic tokens sit inside the scored sequence, so we need their ids to
    # mask them in the circularity check
    vocab = model._datamodule.tokenizer.vocab
    return {tid for tok, tid in vocab.items() if "_static_" in tok}


def compute_batch_scores_calibrated(
    model,
    batch,
    exact_ids,
    exact_centers,
    k=0.2,
    class_values=(0, 1),
    exclude_static_tokens=False,
    static_ids=None,
):
    """
    Returns, per sequence in the batch:
      s_true    : Min-K% under the true conditioning
      s_nullc   : Min-K% with the null centre embedding
      s_flip    : Min-K% with the outcome flipped
    plus the metadata needed for patient-level aggregation. The patient keys come
    from `exact_ids` / `exact_centers` (exact_batch_keys), not from the collated
    float32 IDs, which only drive the conditioning.
    """
    if hasattr(model, "score_privacy_batch"):
        return model.score_privacy_batch(
            batch,
            exact_ids,
            exact_centers,
            k=k,
            exclude_static_tokens=exclude_static_tokens,
        )
    vocabulary = model.vocabulary
    pad_id, eos_id = vocabulary["<PAD>"], vocabulary["<EOS>"]

    X, time_positions, attention_mask, S, IDs = _unpack_batch(batch)
    X = X.to(model.device)
    attention_mask = attention_mask.to(model.device)
    IDs = IDs.to(model.device)
    if time_positions is not None:
        time_positions = time_positions.to(model.device)

    class_labels = IDs[:, 1]
    center_ids = IDs[:, -1] if IDs.size(1) > 3 else torch.zeros_like(class_labels)

    # this flip works for 0/1 and 1/2 encodings alike
    lo, hi = min(class_values), max(class_values)
    flipped_labels = (lo + hi) - class_labels
    null_centers = torch.full_like(center_ids, NULL_CENTER_LABEL)

    # three forward passes instead of N retrainings
    lp_true = _forward_token_logp(
        model, X, time_positions, attention_mask, class_labels, center_ids
    )
    lp_nullc = _forward_token_logp(
        model, X, time_positions, attention_mask, class_labels, null_centers
    )
    lp_flip = _forward_token_logp(
        model, X, time_positions, attention_mask, flipped_labels, center_ids
    )

    shift_labels = X[:, 1:]
    valid = shift_labels != pad_id
    content = valid & (shift_labels != eos_id)
    if exclude_static_tokens and static_ids:
        static_tensor = torch.tensor(sorted(static_ids), device=shift_labels.device)
        is_static = torch.isin(shift_labels, static_tensor)
        content = content & (~is_static)

    content_np = content.cpu().numpy()
    out = {"s_true": [], "s_nullc": [], "s_flip": [], "mean_true": [], "lengths": []}
    arrays = {
        "s_true": lp_true.cpu().numpy(),
        "s_nullc": lp_nullc.cpu().numpy(),
        "s_flip": lp_flip.cpu().numpy(),
    }
    keep = []
    for i in range(X.size(0)):
        mask = content_np[i]
        if mask.sum() == 0:
            continue
        keep.append(i)
        for key, arr in arrays.items():
            out[key].append(min_k_percent_prob(arr[i][mask], k))
        out["mean_true"].append(mean_logp(arrays["s_true"][i][mask]))
        out["lengths"].append(int(mask.sum()))

    keep = np.array(keep, dtype=int)
    statics = S.cpu().numpy() if torch.is_tensor(S) else np.asarray(S)
    return {
        **{key: np.array(val) for key, val in out.items()},
        "stay_ids": np.asarray(exact_ids)[keep],
        "center_ids": np.asarray(exact_centers)[keep],
        "class_labels": class_labels.cpu().numpy()[keep],
        "statics": statics[keep],
    }


def run_calibrated_membership_audit(
    model,
    batch_size=32,
    k=0.2,
    chunk_aggregation="max",
    decoder=None,
    exclude_static_tokens=False,
    max_batches_per_split=None,
    num_workers=0,
):
    """
    Attack A with four statistics:
      raw        = s_true                 (the original, uncalibrated)
      cal_center = s_true - s_nullc       (calibrated by centre swap)
      cal_class  = s_true - s_flip        (calibrated by outcome swap)
      cal_both   = s_true - max(s_nullc, s_flip)   (the most conservative)

    Returns per-patient scores for each statistic, plus metadata.
    """
    dm = model._datamodule
    static_ids = _static_token_ids(model) if exclude_static_tokens else None

    # class values come from the model's label space, not from a data batch: a
    # batch holding one class gave lo == hi, flipped labels to -1 and indexed the
    # label embedding out of range. Labels are 0..num_classes-1 (the last
    # embedding row is the null class), and the flip only makes sense for two
    num_classes = int(model.label_embedder.num_classes)
    if num_classes != 2:
        raise ValueError(
            f"cal_class flips a binary outcome; the model has {num_classes} classes"
        )
    class_values = (0, 1)

    def collect(subset, name):
        loader = make_eval_loader(dm, subset, batch_size, num_workers=num_workers)
        total = (
            len(loader)
            if max_batches_per_split is None
            else min(len(loader), max_batches_per_split)
        )
        acc = defaultdict(list)
        offset = 0
        for b_idx, batch in tqdm(
            enumerate(loader), total=total, desc=f"Calibrated scoring {name}"
        ):
            if max_batches_per_split is not None and b_idx >= max_batches_per_split:
                break
            exact_ids, exact_centers = exact_batch_keys(
                subset, offset, _unpack_batch(batch)[-1]
            )
            offset += len(exact_ids)
            out = compute_batch_scores_calibrated(
                model,
                batch,
                exact_ids,
                exact_centers,
                k=k,
                class_values=class_values,
                exclude_static_tokens=exclude_static_tokens,
                static_ids=static_ids,
            )
            for key, val in out.items():
                acc[key].append(val)
        return {
            key: np.concatenate(val) if val else np.array([])
            for key, val in acc.items()
        }

    train = collect(dm.train_icu_dataset, "TRAIN")
    holdout = collect(dm.test_icu_dataset, "HOLDOUT")

    def build(d):
        stats = {
            "raw": d["s_true"],
            "cal_center": d["s_true"] - d["s_nullc"],
            "cal_class": d["s_true"] - d["s_flip"],
            "cal_both": d["s_true"] - np.maximum(d["s_nullc"], d["s_flip"]),
        }
        per_patient = {}
        for name, arr in stats.items():
            per_patient[name] = _aggregate_by_patient(
                d["stay_ids"], d["center_ids"], arr, method=chunk_aggregation
            )
        return per_patient

    train_pp, holdout_pp = build(train), build(holdout)

    print("\nAttack A, raw vs calibrated (patient level):")
    summaries = {}
    for name in ["raw", "cal_center", "cal_class", "cal_both"]:
        summaries[name] = summarize_attack(
            list(train_pp[name].values()), list(holdout_pp[name].values()), name=name
        )

    # per-patient metadata, with statics already decoded
    def meta(d):
        keys = [_composite_key(s, c) for s, c in zip(d["stay_ids"], d["center_ids"])]
        stat = decoder.decode(d["statics"]) if decoder is not None else d["statics"]
        table = {}
        for i, key in enumerate(keys):
            entry = table.setdefault(key, {"n_chunks": 0, "total_len": 0})
            entry["n_chunks"] += 1
            entry["total_len"] += int(d["lengths"][i])
            entry["center"] = int(d["center_ids"][i])
            entry["cls"] = int(d["class_labels"][i])
            entry["statics"] = (
                stat[i].tolist() if hasattr(stat[i], "tolist") else list(stat[i])
            )
        return table

    return {
        "train_scores": train_pp,
        "holdout_scores": holdout_pp,
        "train_meta": meta(train),
        "holdout_meta": meta(holdout),
        "summaries": summaries,
        "class_values": class_values,
        "_raw": {"train": train, "holdout": holdout},
    }


# matched train and holdout cohort
# The original audit compares all of train against all of holdout. If they differ
# in length, centre composition or chunks per patient, and the per-centre
# truncation cap (`_resolve_truncation_for_centre`) guarantees some of that, then
# part of the AUC is covariate shift rather than memorisation.


def meta_to_frame(meta):
    rows = []
    for key, v in meta.items():
        row = {
            "key": key,
            "center": v["center"],
            "cls": v["cls"],
            "n_chunks": v["n_chunks"],
            "total_len": v["total_len"],
        }
        for j, name in enumerate(DEMOGRAPHIC_ORDER):
            row[name] = v["statics"][j] if j < len(v["statics"]) else -1
        rows.append(row)
    return pd.DataFrame(rows).set_index("key")


def match_cohorts(
    train_meta,
    holdout_meta,
    length_caliper=0.15,
    seed=0,
    match_on=("center", "cls", *DEMOGRAPHIC_ORDER),
):
    """
    Greedy 1:1 matching. Exact on `match_on`, and within each stratum the nearest
    neighbour in `total_len` with a relative caliper.

    Returns aligned (train_keys, holdout_keys) and a balance report.
    """
    rng = np.random.default_rng(seed)
    tdf, hdf = meta_to_frame(train_meta), meta_to_frame(holdout_meta)

    buckets = defaultdict(list)
    for key, row in tdf.iterrows():
        buckets[tuple(row[c] for c in match_on)].append((row["total_len"], key))
    for stratum in buckets:
        buckets[stratum].sort()

    # random order over the holdout so the greedy pass is not biased by whatever
    # order the rows happen to come in
    matched_t, matched_h = [], []
    h_order = rng.permutation(len(hdf))
    for pos in tqdm(h_order, desc="Matching cohorts"):
        key_h = hdf.index[pos]
        row = hdf.iloc[pos]
        stratum = tuple(row[c] for c in match_on)
        pool = buckets.get(stratum)
        if not pool:
            continue
        target = row["total_len"]
        lens = np.array([p[0] for p in pool])
        j = int(np.argmin(np.abs(lens - target)))
        if abs(lens[j] - target) > length_caliper * max(1.0, target):
            continue
        matched_t.append(pool[j][1])
        matched_h.append(key_h)
        pool.pop(j)  # sampling without replacement keeps the matching 1:1

    report = {
        "n_matched": len(matched_t),
        "n_train_available": len(tdf),
        "n_holdout_available": len(hdf),
        "match_rate_holdout": len(matched_h) / max(1, len(hdf)),
    }
    if matched_t:
        bt, bh = tdf.loc[matched_t], hdf.loc[matched_h]
        report["balance"] = pd.DataFrame(
            {
                "train_mean": bt[["total_len", "n_chunks"]].mean(),
                "holdout_mean": bh[["total_len", "n_chunks"]].mean(),
            }
        )
    print(
        f"  Matched {report['n_matched']} pairs "
        f"({report['match_rate_holdout']:.1%} of the holdout)"
    )
    if "balance" in report:
        print(report["balance"].to_string())
    return matched_t, matched_h, report


def unmatched_covariate_report(train_meta, holdout_meta):
    """Quantifies the imbalance before matching. Belongs in the paper."""
    tdf, hdf = meta_to_frame(train_meta), meta_to_frame(holdout_meta)
    rows = []
    for col in ["total_len", "n_chunks"]:
        rows.append(
            {
                "variable": col,
                "train_mean": tdf[col].mean(),
                "holdout_mean": hdf[col].mean(),
                "std_diff": (tdf[col].mean() - hdf[col].mean())
                / np.sqrt(0.5 * (tdf[col].var() + hdf[col].var())),
            }
        )
    df = pd.DataFrame(rows)
    print("\nTrain vs holdout imbalance (standardised difference):")
    print(df.to_string(index=False))
    # chunks per patient is the mechanism behind the per-centre confounding
    print("\nChunks per patient and per centre (confounding mechanism):")
    print(
        tdf.groupby("center")["n_chunks"].agg(["mean", "median", "count"]).to_string()
    )
    return df


# neighbourhood calibration on the holdout


def neighborhood_calibrate(
    target_scores,
    target_meta,
    reference_scores,
    reference_meta,
    k_neighbors=25,
    strata=("center", "cls"),
):
    """
    score(x) minus the mean score of its k closest neighbours in the holdout.

    The holdout acts as a non-member reference population, stratified by
    difficulty. This is the right way to use subsets of real patients without
    retraining anything.
    """
    rdf = meta_to_frame(reference_meta)
    rdf["score"] = pd.Series(reference_scores)
    rdf = rdf.dropna(subset=["score"])

    buckets = {}
    for stratum, grp in rdf.groupby(list(strata)):
        grp = grp.sort_values("total_len")
        buckets[stratum if isinstance(stratum, tuple) else (stratum,)] = (
            grp["total_len"].to_numpy(),
            grp["score"].to_numpy(),
        )

    tdf = meta_to_frame(target_meta)
    out = {}
    for key, row in tdf.iterrows():
        if key not in target_scores:
            continue
        stratum = tuple(row[c] for c in strata)
        if stratum not in buckets:
            continue
        # neighbours are taken by length within the same centre and outcome
        lens, scores = buckets[stratum]
        pos = int(np.searchsorted(lens, row["total_len"]))
        lo, hi = max(0, pos - k_neighbors // 2), min(
            len(scores), pos + k_neighbors // 2 + 1
        )
        if hi - lo < 5:
            continue
        out[key] = float(target_scores[key] - scores[lo:hi].mean())
    return out


# attack C, pool control and power sweep
# Two problems with the original:
#
# (a) Comparison confounded by pool size. With 12k synthetics:
#       conditioned -> cells (class x centre) -> ~140 candidates/cell
#       null        -> cells (class)          -> ~840 candidates/cell
#     The null mode had 6x more candidates, so 0.497 vs 0.500 was not comparable.
#     `max_candidates_per_cell` evens them out.
#
# (b) Without a power curve there is no way to tell "no leakage" from "the pool
#     was small". We cannot go above 12k, but we can go below: if the AUC is flat
#     across 1.5 orders of magnitude, extrapolating is defensible.


def _cell_indices(real_classes, real_centers, gen_labels, gen_centers, use_center):
    """Yields (real_idx, candidate_idx, cell_label) per cell."""
    gen_conditioned = gen_centers != NULL_CENTER_LABEL
    for cls in np.unique(real_classes):
        centers = np.unique(real_centers[real_classes == cls]) if use_center else [None]
        for centre in centers:
            if use_center:
                idx = np.where((real_classes == cls) & (real_centers == centre))[0]
                cand = np.where(
                    (gen_labels == cls) & gen_conditioned & (gen_centers == centre)
                )[0]
            else:
                idx = np.where(real_classes == cls)[0]
                cand = np.where(gen_labels == cls)[0]
            if len(idx) == 0:
                continue
            if len(cand) == 0:
                cand = np.where(gen_labels == cls)[0]
                print(
                    f"  WARNING: no candidates for class={cls}, centre={centre}; "
                    "falling back to the full class pool."
                )
            yield idx, cand, (int(cls), None if centre is None else int(centre))


def generation_scores_with_sweep(
    real_seqs,
    real_classes,
    real_centers,
    gen_seqs,
    gen_labels,
    gen_centers,
    use_center=True,
    max_candidates_per_cell=None,
    pool_sweep=None,
    seed=0,
):
    """
    Minimum distance from each real to the synthetic pool, per cell
    (class x centre), for the full pool and for every size in the sweep.

    The sweep reuses the same distance matrix: subsampling the pool is just
    taking the minimum over a subset of columns, so the marginal cost is zero.
    """
    rng = np.random.default_rng(seed)
    n_real = len(real_seqs)
    pool_sweep = (
        [] if pool_sweep is None else [int(s) for s in np.asarray(pool_sweep).ravel()]
    )
    scores = np.full(n_real, np.nan)
    sweep_scores = {s: np.full(n_real, np.nan) for s in pool_sweep}
    cell_report = []

    for idx, cand, cell in _cell_indices(
        real_classes, real_centers, gen_labels, gen_centers, use_center
    ):
        if max_candidates_per_cell is not None and len(cand) > max_candidates_per_cell:
            cand = rng.choice(cand, max_candidates_per_cell, replace=False)
        sub_real = [real_seqs[i] for i in idx]
        sub_gen = [gen_seqs[j] for j in cand]
        dist = edit_distance_matrix(sub_real, sub_gen, show_progress=len(idx) > 2000)
        scores[idx] = dist.min(axis=1)
        cell_report.append(
            {
                "cls": cell[0],
                "centre": cell[1],
                "n_real": len(idx),
                "n_candidates": len(cand),
            }
        )
        for size in pool_sweep:
            take = min(size, dist.shape[1])
            cols = rng.choice(dist.shape[1], take, replace=False)
            sweep_scores[size][idx] = dist[:, cols].min(axis=1)

    return scores, sweep_scores, pd.DataFrame(cell_report)


def run_generation_mia_v2(
    model,
    generated_samples_path,
    real_train,
    real_holdout,
    use_center="auto",
    canonicalize=True,
    chunk_zero_only=True,
    require_eos=True,
    max_candidates_per_cell=None,
    pool_sweep=(10, 25, 50, 100),
    chunk_aggregation="min",
    seed=0,
    label="",
):
    """
    real_train / real_holdout: dicts with keys
      seqs, ids, centers, classes   (already collected and cohort filtered)
    """
    import joblib

    generated_samples, generated_labels, generated_centers = joblib.load(
        generated_samples_path
    )
    gen_seqs_raw = [list(s) for s in generated_samples]
    gen_labels = np.asarray(
        generated_labels.cpu().numpy()
        if torch.is_tensor(generated_labels)
        else generated_labels
    ).astype(int)
    gen_centers = np.asarray(
        generated_centers.cpu().numpy()
        if torch.is_tensor(generated_centers)
        else generated_centers
    ).astype(int)

    sequence_diagnostics(gen_seqs_raw, real_train["seqs"], label=label)

    gen_seqs, gen_mask = prepare_sequences(
        gen_seqs_raw,
        canonicalize=canonicalize,
        chunk_zero_only=False,
        require_eos=require_eos,
        return_mask=True,
    )
    gen_labels, gen_centers = gen_labels[gen_mask], gen_centers[gen_mask]
    print(
        f"  Synthetics after dropping incomplete ones: {len(gen_seqs)}/{len(gen_seqs_raw)} "
        f"({1 - gen_mask.mean():.1%} dropped for not emitting <EOS>)"
    )

    def prep_real(d):
        seqs, mask = prepare_sequences(
            d["seqs"],
            canonicalize=canonicalize,
            chunk_zero_only=chunk_zero_only,
            require_eos=False,
            return_mask=True,
        )
        return {
            "seqs": seqs,
            "ids": np.asarray(d["ids"])[mask],
            "centers": np.asarray(d["centers"])[mask],
            "classes": np.asarray(d["classes"])[mask],
            "mask": mask,
        }

    rt, rh = prep_real(real_train), prep_real(real_holdout)
    print(
        f"  Reals after filtering to chunk-0: train={len(rt['seqs'])}, holdout={len(rh['seqs'])}"
    )

    if use_center == "auto":
        conditioned = gen_centers != NULL_CENTER_LABEL
        shared = set(np.unique(gen_centers[conditioned]).tolist()) & set(
            np.unique(np.concatenate([rt["centers"], rh["centers"]])).tolist()
        )
        use_center = len(shared) >= 2
        print(f"  Centre conditioning detected: {use_center}")

    pool_sweep = (
        [] if pool_sweep is None else [int(s) for s in np.asarray(pool_sweep).ravel()]
    )
    common = dict(
        gen_seqs=gen_seqs,
        gen_labels=gen_labels,
        gen_centers=gen_centers,
        use_center=use_center,
        max_candidates_per_cell=max_candidates_per_cell,
        pool_sweep=pool_sweep,
        seed=seed,
    )
    train_scores, train_sweep, cells_t = generation_scores_with_sweep(
        rt["seqs"], rt["classes"], rt["centers"], **common
    )
    holdout_scores, holdout_sweep, _ = generation_scores_with_sweep(
        rh["seqs"], rh["classes"], rh["centers"], **common
    )

    print("\n  Pool size per cell (what the adversary sees):")
    print(cells_t.to_string(index=False))

    tp = _aggregate_by_patient(
        rt["ids"], rt["centers"], train_scores, method=chunk_aggregation
    )
    hp = _aggregate_by_patient(
        rh["ids"], rh["centers"], holdout_scores, method=chunk_aggregation
    )
    # negated because closer means riskier here
    main = summarize_attack(
        [-v for v in tp.values()], [-v for v in hp.values()], name=f"gen-MIA {label}"
    )

    sweep_summary = []
    for size in pool_sweep:
        tps = _aggregate_by_patient(
            rt["ids"], rt["centers"], train_sweep[size], method=chunk_aggregation
        )
        hps = _aggregate_by_patient(
            rh["ids"], rh["centers"], holdout_sweep[size], method=chunk_aggregation
        )
        s = summarize_attack(
            [-v for v in tps.values()],
            [-v for v in hps.values()],
            name=f"pool={size}/cell",
            verbose=False,
        )
        sweep_summary.append(
            {
                "candidates_per_cell": size,
                "auc": s["auc"],
                "ci_low": s["auc_ci"][0],
                "ci_high": s["auc_ci"][1],
            }
        )
    full_cells = int(cells_t["n_candidates"].median()) if len(cells_t) else 0
    sweep_summary.append(
        {
            "candidates_per_cell": full_cells,
            "auc": main["auc"],
            "ci_low": main["auc_ci"][0],
            "ci_high": main["auc_ci"][1],
        }
    )
    sweep_df = pd.DataFrame(sweep_summary).sort_values("candidates_per_cell")
    print("\n  Power curve (AUC vs candidates per cell):")
    print(sweep_df.to_string(index=False))

    return {
        "summary": main,
        "pool_sweep": sweep_df,
        "cells": cells_t,
        "train_patient_scores": tp,
        "holdout_patient_scores": hp,
        "use_center": use_center,
        "n_gen_used": len(gen_seqs),
        "_raw": {
            "train_scores": train_scores,
            "holdout_scores": holdout_scores,
            "train_prepared": rt,
            "holdout_prepared": rh,
            "gen_seqs": gen_seqs,
            "gen_labels": gen_labels,
            "gen_centers": gen_centers,
        },
    }


def _split_blocks(tokens):
    # splits the sequence into timepoints, each ending at its GAP token
    blocks, cur = [], []
    for t in tokens:
        cur.append(t)
        if t == GAP_TOKEN:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def make_canary(seq, variant, rng, drop_fraction=0.10):
    """literal | shuffled (within the timepoint) | dropped (timepoints removed)."""
    if variant == "literal":
        return list(seq)

    if variant == "shuffled":
        # only touches the within-hour order, the symmetry the model does not use
        out = []
        for block in _split_blocks(seq):
            body = [t for t in block if t != GAP_TOKEN]
            rng.shuffle(body)
            out.extend(body + ([GAP_TOKEN] if block and block[-1] == GAP_TOKEN else []))
        return out

    if variant == "dropped":
        blocks = _split_blocks(seq)
        if len(blocks) <= 1:
            return list(seq)
        n_drop = min(max(1, int(round(len(blocks) * drop_fraction))), len(blocks) - 1)
        drop_idx = set(rng.choice(len(blocks), n_drop, replace=False).tolist())
        return [t for i, b in enumerate(blocks) if i not in drop_idx for t in b]

    raise ValueError(f"Unknown variant: {variant}")


def run_canary_audit_v3(
    real_train_seqs,
    real_classes,
    real_centers,
    gen_seqs,
    gen_labels,
    gen_centers,
    n_canaries=200,
    drop_fractions=(0.05, 0.10, 0.20, 0.40, 0.80),
    canonicalize_settings=(True, False),
    max_candidates_per_cell=140,
    seed=0,
):
    """
    For each (canonicalisation, variant) it measures:
      rank1_rate : fraction of sources whose nearest neighbour is their own canary
      margin     : d(best synthetic) - d(canary), median. Positive means detectable.
      d_canary   : median source to canary distance

    `literal` should always give rank1 = 1.0; if it does not, the metric is broken.
    """
    rng = np.random.default_rng(seed)
    real_classes = np.asarray(real_classes)
    real_centers = np.asarray(real_centers)
    gen_labels = np.asarray(gen_labels)

    chunk0 = [
        i for i in range(len(real_train_seqs)) if is_chunk_zero(real_train_seqs[i])
    ]
    src_idx = rng.choice(chunk0, min(n_canaries, len(chunk0)), replace=False)

    variants = [("literal", None), ("shuffled", None)] + [
        ("dropped", f) for f in drop_fractions
    ]
    rows = []

    for canon in canonicalize_settings:
        prep = (lambda s: canonicalize_sequence(s)) if canon else (lambda s: list(s))

        # synthetic pool per class, prepared once per setting
        pools = {}
        for cls in np.unique(real_classes):
            cand = np.where(gen_labels == cls)[0]
            if len(cand) == 0:
                cand = np.arange(len(gen_seqs))
            if (
                max_candidates_per_cell is not None
                and len(cand) > max_candidates_per_cell
            ):
                cand = rng.choice(cand, max_candidates_per_cell, replace=False)
            pools[int(cls)] = [prep(gen_seqs[j]) for j in cand]

        for variant, frac in variants:
            # same seed per variant so corruptions are comparable across settings
            vrng = np.random.default_rng(seed + 1)
            rank1, margins, d_cans = [], [], []
            label = variant if frac is None else f"dropped@{int(frac*100)}%"

            for i in tqdm(src_idx, desc=f"Canary canon={canon} {label}", leave=False):
                src = prep(real_train_seqs[i])
                canary = prep(
                    make_canary(
                        real_train_seqs[i], variant, vrng, drop_fraction=frac or 0.10
                    )
                )
                pool = pools[int(real_classes[i])]
                if not pool:
                    continue
                d_pool = float(edit_distance_matrix([src], pool)[0].min())
                d_can = float(edit_distance_matrix([src], [canary])[0][0])
                rank1.append(d_can < d_pool)
                margins.append(d_pool - d_can)
                d_cans.append(d_can)

            rows.append(
                {
                    "canonicalize": canon,
                    "variant": label,
                    "n": len(rank1),
                    "rank1_rate": float(np.mean(rank1)) if rank1 else np.nan,
                    "median_margin": float(np.median(margins)) if margins else np.nan,
                    "median_d_canary": float(np.median(d_cans)) if d_cans else np.nan,
                }
            )

    df = pd.DataFrame(rows)
    print("\nCanary: rank-1 recovery (threshold free)")
    print(df.to_string(index=False))

    lit = df[(df.variant == "literal") & (df.canonicalize)]
    if len(lit) and lit.iloc[0]["rank1_rate"] < 0.99:
        print(
            "  WARNING: the literal canary is not recovered at ~100%. The distance "
            "metric is broken; do not interpret any null result until it is fixed."
        )

    sh_on = df[(df.variant == "shuffled") & (df.canonicalize)]
    sh_off = df[(df.variant == "shuffled") & (~df.canonicalize)]
    if len(sh_on) and len(sh_off):
        delta = sh_on.iloc[0]["rank1_rate"] - sh_off.iloc[0]["rank1_rate"]
        print(
            f"\n  Canonicalisation ablation on the shuffled variant: "
            f"rank1 {sh_off.iloc[0]['rank1_rate']:.3f} (without) -> "
            f"{sh_on.iloc[0]['rank1_rate']:.3f} (with), delta={delta:+.3f}"
        )
        if delta > 0.3:
            print(
                "  -> Canonicalisation is necessary: without it the metric is blind "
                "to a symmetry the model does not use (shuffle_tokens_within_timepoint). "
                "This comparison goes in the paper as a figure."
            )

    dropped = df[(df.canonicalize) & (df.variant.str.startswith("dropped"))]
    broken = dropped[dropped.rank1_rate < 0.5]
    if len(broken):
        print(
            f"\n  Detection floor: the attack stops recovering from "
            f"{broken.iloc[0]['variant']} corruption onwards."
        )
    else:
        print(
            "\n  Detection floor: the attack recovers up to the highest corruption "
            "level tested. Extend `drop_fractions` to find the limit."
        )
    return df


def _overlap_coefficient(a, b, bins=100):
    # shared area under the two densities, more informative here than a KS
    # statistic that any large sample turns significant
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.minimum(ha, hb).sum() * width)


def compare_distance_distributions(dist_syn_train, dist_syn_holdout, real_real_dists):
    """
    Compare generated-to-real distances with a distinct-patient reference.

    The real-to-real reference distinguishes replication from the similarity
    expected between unrelated patients in the same clinical domain.
    """
    a = np.asarray(dist_syn_train, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(dist_syn_holdout, dtype=float)
    b = b[~np.isnan(b)]
    r = np.asarray(real_real_dists, dtype=float)
    r = r[~np.isnan(r)]

    rows = []
    for name, arr in [
        ("synth_vs_train", a),
        ("synth_vs_holdout", b),
        ("real_vs_real", r),
    ]:
        rows.append(
            {
                "distribution": name,
                "n": len(arr),
                "p05": float(np.percentile(arr, 5)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "min": float(arr.min()),
            }
        )
    df = pd.DataFrame(rows)

    ks_stat, ks_p = ks_2samp(a, r)
    overlap = _overlap_coefficient(a, r)
    shift = float(np.median(a) - np.median(r))

    print(
        "\nMinimum distance distributions (after canonicalising and chunk-0 filtering)"
    )
    print(df.to_string(index=False))
    print(
        f"\n  Synthetic-train vs real-real: KS={ks_stat:.4f} (p={ks_p:.2e}), "
        f"overlap={overlap:.3f}, median shift={shift:+.4f}"
    )
    if shift > 0.05 and overlap < 0.5:
        print(
            "  -> Synthetics are still further from reals than reals are from each other."
        )
        print(
            "     Report as a structural fidelity finding, not as absence of leakage:"
        )
        print("     the metric has no resolution in that range to claim a null.")
    else:
        print("  -> The distributions overlap: the replication null is interpretable.")
    return {
        "table": df,
        "ks": (float(ks_stat), float(ks_p)),
        "overlap": overlap,
        "median_shift": shift,
    }


# worst case attacks


def token_rarity(seqs, reference_seqs):
    """Mean token rarity of each sequence (inverse frequency)."""
    counts = Counter(t for s in reference_seqs for t in s)
    total = sum(counts.values())
    out = []
    for s in seqs:
        if not s:
            out.append(0.0)
            continue
        out.append(float(np.mean([-np.log((counts.get(t, 1)) / total) for t in s])))
    return np.array(out)


def worst_case_attack(
    train_scores,
    holdout_scores,
    train_meta,
    holdout_meta,
    name="attack",
    rarity_train=None,
    rarity_holdout=None,
):
    """
    Repeats the attack on the most identifiable patients. Privacy risk is not
    uniform: an average over the whole cohort can hide a vulnerable subgroup.
    """
    tdf, hdf = meta_to_frame(train_meta), meta_to_frame(holdout_meta)
    tdf["score"] = pd.Series(train_scores)
    hdf["score"] = pd.Series(holdout_scores)
    tdf, hdf = tdf.dropna(subset=["score"]), hdf.dropna(subset=["score"])
    if rarity_train is not None:
        tdf["rarity"] = pd.Series(rarity_train)
        hdf["rarity"] = pd.Series(rarity_holdout)

    # cut points come from train so both groups are split at the same value
    subsets = {
        "all": (tdf, hdf),
        "longest_decile": (
            tdf[tdf.total_len >= tdf.total_len.quantile(0.9)],
            hdf[hdf.total_len >= tdf.total_len.quantile(0.9)],
        ),
        "minority_class": (
            tdf[tdf.cls == tdf.cls.value_counts().idxmin()],
            hdf[hdf.cls == tdf.cls.value_counts().idxmin()],
        ),
    }
    smallest = tdf.center.value_counts().idxmin()
    subsets[f"smallest_center({smallest})"] = (
        tdf[tdf.center == smallest],
        hdf[hdf.center == smallest],
    )
    if "rarity" in tdf:
        cut = tdf.rarity.quantile(0.9)
        subsets["rarest_decile"] = (tdf[tdf.rarity >= cut], hdf[hdf.rarity >= cut])

    rows = []
    for label, (t, h) in subsets.items():
        if len(t) < 30 or len(h) < 30:
            print(
                f"  [{name}/{label}] not enough n (train={len(t)}, holdout={len(h)}), skipped."
            )
            continue
        s = summarize_attack(
            t["score"].to_numpy(),
            h["score"].to_numpy(),
            name=f"{name}/{label}",
            verbose=False,
        )
        rows.append(
            {
                "subset": label,
                "n_train": len(t),
                "n_holdout": len(h),
                "auc": s["auc"],
                "ci_low": s["auc_ci"][0],
                "ci_high": s["auc_ci"][1],
                "tpr@fpr=0.01": s["tpr@fpr=0.01"],
            }
        )
    df = pd.DataFrame(rows)
    print(f"\n  Worst case attack [{name}]:")
    print(df.to_string(index=False))
    return df


# subgroups with stratified CMH
# Replaces "repeat the test in centre 1 only", which confuses absence of effect
# with loss of power from the smaller n. CMH uses all the data and separates the
# effect of the variable from the effect of the centre.


def cochran_mantel_haenszel(flagged, exposure, strata):
    """
    CMH for 2x2xK tables.
      flagged  : bool (in the risk tail)
      exposure : bool (category of interest vs the rest)
      strata   : array of strata (centre)
    Returns the statistic, the p value and the Mantel-Haenszel common odds ratio.
    """
    from scipy.stats import chi2

    flagged = np.asarray(flagged).astype(bool)
    exposure = np.asarray(exposure).astype(bool)
    strata = np.asarray(strata)

    num, den_var, or_num, or_den = 0.0, 0.0, 0.0, 0.0
    obs_total = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = float((flagged[m] & exposure[m]).sum())
        b = float((~flagged[m] & exposure[m]).sum())
        c = float((flagged[m] & ~exposure[m]).sum())
        d = float((~flagged[m] & ~exposure[m]).sum())
        n = a + b + c + d
        if n < 2 or (a + c) == 0 or (a + b) == 0:
            continue
        obs_total += a
        num += (a + b) * (a + c) / n
        den_var += ((a + b) * (c + d) * (a + c) * (b + d)) / (n * n * (n - 1))
        or_num += a * d / n
        or_den += b * c / n

    if den_var <= 0:
        return {"chi2": float("nan"), "p_value": float("nan"), "or_mh": float("nan")}
    # the 0.5 is the continuity correction
    stat = (abs(obs_total - num) - 0.5) ** 2 / den_var
    return {
        "chi2": float(stat),
        "p_value": float(chi2.sf(stat, df=1)),
        "or_mh": float(or_num / or_den) if or_den > 0 else float("inf"),
    }


def run_subgroup_analysis_v2(risks, meta, percentile=99.0, bonferroni=True):
    """
    Subgroup analysis with already decoded statics, in two versions:
      - pooled chi2 (the original, kept for comparability)
      - CMH stratified by centre (the one to report)
    """
    tail = flag_high_risk_patients(risks, percentile=percentile)
    flagged_ids = set(tail["flagged"].keys())
    df = meta_to_frame(meta)
    df = df.loc[[k for k in risks if k in df.index]]
    df["flagged"] = df.index.isin(flagged_ids)

    print(
        f"\nExtreme risk tail: {int(df.flagged.sum())} / {len(df)} "
        f"(risk >= {tail['threshold']:.3f})"
    )

    n_tests = len(DEMOGRAPHIC_ORDER)
    results = {}
    for col in DEMOGRAPHIC_ORDER:
        values_by_id = df[col].to_dict()
        pooled = subgroup_composition_test(
            values_by_id,
            flagged_ids,
            list(df.index),
            is_categorical=True,
        )
        if bonferroni:
            pooled["p_value_bonferroni"] = min(1.0, pooled["p_value"] * n_tests)

        # one CMH per category against the rest, since the variables have more
        # than two levels
        cmh_by_cat = {}
        for cat in sorted(df[col].unique()):
            res = cochran_mantel_haenszel(df["flagged"], df[col] == cat, df["center"])
            if bonferroni:
                res["p_value_bonferroni"] = min(
                    1.0, res["p_value"] * n_tests * len(df[col].unique())
                )
            cmh_by_cat[int(cat)] = res

        results[col] = {"pooled": pooled, "cmh": cmh_by_cat}
        p_pool = pooled.get("p_value_bonferroni", pooled["p_value"])
        best_cmh = min(
            (
                v.get("p_value_bonferroni", v["p_value"])
                for v in cmh_by_cat.values()
                if not np.isnan(v["p_value"])
            ),
            default=float("nan"),
        )
        print(
            f"  {col:<16} pooled p={p_pool:.2e} | CMH (best category) p={best_cmh:.2e}"
        )

    return {"tail": tail, "results": results, "table": df}


def chunk_count_confound_check(risks, meta):
    """
    Correlation between chunk count and risk. If it is positive, the `max`
    aggregation is rewarding patients with more chunks, and the per-centre
    truncation cap mechanistically explains the centre enrichment of the tail.
    """
    df = meta_to_frame(meta)
    df = df.loc[[k for k in risks if k in df.index]]
    df["risk"] = pd.Series(risks)
    rho = df[["n_chunks", "risk"]].corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman correlation (n_chunks, risk) = {rho:.3f}")
    print(df.groupby("center")[["n_chunks", "risk"]].mean().to_string())
    return {
        "spearman": float(rho),
        "by_center": df.groupby("center")[["n_chunks", "risk"]].mean(),
    }


# subgroups, direction and effect size
# `age_static` survives CMH (p=2.53e-05), so the disparity is real and has to be
# reported properly. Three things were missing: the direction of the effect, the
# effect size with CI, and robustness to the percentile chosen for the tail.


def mantel_haenszel_or(flagged, exposure, strata, alpha=0.05):
    """
    Mantel-Haenszel common OR with a CI from the Robins-Breslow-Greenland
    variance. With n=178 in the tail the effect size matters more than the p value.
    """
    flagged = np.asarray(flagged).astype(bool)
    exposure = np.asarray(exposure).astype(bool)
    strata = np.asarray(strata)

    R_sum = S_sum = 0.0
    t1 = t2 = t3 = 0.0
    num = var_num = obs = 0.0

    for s in np.unique(strata):
        m = strata == s
        a = float((flagged[m] & exposure[m]).sum())
        b = float((~flagged[m] & exposure[m]).sum())
        c = float((flagged[m] & ~exposure[m]).sum())
        d = float((~flagged[m] & ~exposure[m]).sum())
        n = a + b + c + d
        if n < 2:
            continue
        R, S = a * d / n, b * c / n
        P, Q = (a + d) / n, (b + c) / n
        R_sum += R
        S_sum += S
        t1 += P * R
        t2 += P * S + Q * R
        t3 += Q * S
        obs += a
        num += (a + b) * (a + c) / n
        if n > 1:
            var_num += ((a + b) * (c + d) * (a + c) * (b + d)) / (n * n * (n - 1))

    if R_sum <= 0 or S_sum <= 0 or var_num <= 0:
        return {
            "or_mh": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "chi2": np.nan,
            "p_value": np.nan,
        }

    or_mh = R_sum / S_sum
    # variance of the log OR, which is where the CI comes from
    var_log = t1 / (2 * R_sum**2) + t2 / (2 * R_sum * S_sum) + t3 / (2 * S_sum**2)
    se = np.sqrt(var_log)
    z = 1.959963985 if abs(alpha - 0.05) < 1e-9 else float(abs(np.sqrt(2) * 0.0))
    stat = (abs(obs - num) - 0.5) ** 2 / var_num

    return {
        "or_mh": float(or_mh),
        "ci_low": float(np.exp(np.log(or_mh) - z * se)),
        "ci_high": float(np.exp(np.log(or_mh) + z * se)),
        "chi2": float(stat),
        "p_value": float(chi2.sf(stat, df=1)),
    }


def subgroup_report_v3(
    risks,
    meta,
    percentiles=(95.0, 99.0, 99.5),
    variables=DEMOGRAPHIC_ORDER,
    bonferroni=True,
):
    """
    Full subgroup report:
      - direction: Pearson residuals and relative change per category
      - effect size: Mantel-Haenszel OR with 95% CI
      - robustness: the same test at several tail percentiles
    """
    df_meta = meta_to_frame(meta)
    keys = [k for k in risks if k in df_meta.index]
    df_meta = df_meta.loc[keys]

    all_rows, direction_rows = [], []
    n_tests = len(variables) * len(percentiles)

    for pct in percentiles:
        tail = flag_high_risk_patients({k: risks[k] for k in keys}, percentile=pct)
        flagged_ids = set(tail["flagged"].keys())
        flagged_mask = df_meta.index.isin(flagged_ids)

        for var in variables:
            values_by_id = df_meta[var].to_dict()
            pooled = subgroup_composition_test(
                values_by_id, flagged_ids, list(df_meta.index), is_categorical=True
            )
            p_pooled = (
                min(1.0, pooled["p_value"] * n_tests)
                if bonferroni
                else pooled["p_value"]
            )

            # keeps the category with the lowest p value, so note that the
            # reported category can change between percentiles
            best = {"p_value": 1.0}
            for cat in sorted(df_meta[var].unique()):
                res = mantel_haenszel_or(
                    flagged_mask, df_meta[var] == cat, df_meta["center"]
                )
                res["category"] = int(cat)
                if not np.isnan(res["p_value"]) and res["p_value"] < best["p_value"]:
                    best = res
                if pct == 99.0:
                    direction_rows.append(
                        {
                            "variable": var,
                            "category": int(cat),
                            "or_mh": res["or_mh"],
                            "ci_low": res["ci_low"],
                            "ci_high": res["ci_high"],
                            "p_cmh": res["p_value"],
                        }
                    )

            p_cmh = (
                min(1.0, best.get("p_value", np.nan) * n_tests)
                if bonferroni
                else best.get("p_value", np.nan)
            )
            all_rows.append(
                {
                    "percentile": pct,
                    "variable": var,
                    "n_flagged": tail["n_flagged"],
                    "p_pooled": p_pooled,
                    "p_cmh_best": p_cmh,
                    "best_category": best.get("category", np.nan),
                    "or_mh": best.get("or_mh", np.nan),
                    "ci": f"[{best.get('ci_low', np.nan):.2f}, {best.get('ci_high', np.nan):.2f}]",
                }
            )

            if pct == 99.0:
                print(f"\n  {var} @P{pct} -- effect direction:")
                for cat, resid, pct_chg in zip(
                    pooled["categories"],
                    pooled["pearson_residuals"],
                    pooled["relative_change_pct"],
                ):
                    print(
                        f"      category {cat}: residual={resid:+.2f}, "
                        f"change={pct_chg:+.1f}%"
                    )

    sens = pd.DataFrame(all_rows)
    print("\nSensitivity to the tail percentile (Bonferroni corrected p values):")
    print(sens.to_string(index=False))

    # an effect that only shows up at some percentiles is not robust
    unstable = []
    for var in variables:
        sub = sens[sens.variable == var]
        sig = (sub["p_cmh_best"] < 0.05).sum()
        if 0 < sig < len(percentiles):
            unstable.append(var)
    if unstable:
        print(
            f"\n  WARNING: {unstable} is only significant at some percentiles. "
            "Fragile effect; report the sensitivity rather than a single percentile."
        )

    return {"sensitivity": sens, "effect_sizes": pd.DataFrame(direction_rows)}


def competing_risk_model(risks, meta, percentile=99.0):
    """
    Logistic `flagged ~ age + female + total_len + C(centre)`.
    Checks whether age survives with all the covariates in.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        print("  statsmodels not available; skipping the competing model.")
        return None

    df = meta_to_frame(meta)
    keys = [k for k in risks if k in df.index]
    df = df.loc[keys].copy()
    tail = flag_high_risk_patients({k: risks[k] for k in keys}, percentile=percentile)
    df["flagged"] = df.index.isin(set(tail["flagged"].keys())).astype(int)
    # log scale because the length distribution is heavily right skewed
    df["log_len"] = np.log1p(df["total_len"])

    formula = (
        "flagged ~ age_static + female_static + weight_static + log_len + C(center)"
    )
    try:
        model = smf.glm(formula, data=df, family=sm.families.Binomial()).fit()
    except Exception as exc:
        print(f"  The logistic model did not converge: {exc}")
        return None

    print(f"\nCompeting model (P{percentile}):")
    print(model.summary2().tables[1].to_string())
    return model


def _is_time_series_token(token):
    # special tokens and demographic bins are left out: a rare demographic bin is
    # a different route to identifiability, already covered by block 7
    return not token.startswith("<") and "_static_" not in token


def patient_token_profile(full_real, rare_threshold=100):
    """
    Time-series token rarity per patient over the full dataset.

    full_real: {"train"|"val"|"test": {"seqs", "ids", "centers"}}, every chunk.

    A token is rare when fewer than `rare_threshold` patients carry it, and a
    patient is atypical when they carry at least one rare token. The threshold
    is a fixed count so it means the same thing at every bin count. Each patient
    keeps the patient count of their rarest token, so the subgroup can be re-cut
    at another threshold from the saved results without rerunning.
    """
    index, is_ts = {}, {}
    patient_tokens, patient_split = {}, {}
    for split, d in full_real.items():
        for seq, sid, cid in zip(d["seqs"], d["ids"], d["centers"]):
            ids = []
            for t in set(seq):
                if t not in is_ts:
                    is_ts[t] = _is_time_series_token(t)
                if is_ts[t]:
                    ids.append(index.setdefault(t, len(index)))
            key = _composite_key(sid, cid)
            arr = np.array(ids, dtype=np.int32)
            prev = patient_tokens.get(key)
            patient_tokens[key] = arr if prev is None else np.union1d(prev, arr)
            patient_split[key] = split

    # every per-patient array holds unique ids, so this counts patients, not chunks
    counts = np.bincount(
        np.concatenate(list(patient_tokens.values())), minlength=len(index)
    )
    token_patients = pd.Series(counts, index=list(index)).sort_values()

    rows = []
    for key, arr in patient_tokens.items():
        c = counts[arr]
        rows.append(
            {
                "key": key,
                "split": patient_split[key],
                "rarest_token_patients": int(c.min()) if len(c) else np.nan,
                "n_rare_tokens": int((c < rare_threshold).sum()),
            }
        )
    patients = pd.DataFrame(rows).set_index("key")
    patients["atypical"] = patients["n_rare_tokens"] > 0

    n_rare = int((token_patients < rare_threshold).sum())
    print(
        f"\nToken profile: {len(token_patients)} time-series tokens, {n_rare} rare "
        f"(< {rare_threshold} patients in the full dataset)"
    )
    by_split = patients.groupby("split")["atypical"].agg(["sum", "count", "mean"])
    by_split.columns = ["n_atypical", "n_patients", "fraction"]
    by_split.loc["all"] = [
        patients["atypical"].sum(),
        len(patients),
        patients["atypical"].mean(),
    ]
    print(by_split.to_string(float_format=lambda x: f"{x:.4g}"))
    return {
        "token_patients": token_patients,
        "patients": patients,
        "rare_threshold": rare_threshold,
        "by_split": by_split,
    }


def combine_risks_explicit(risks, method="max"):
    """
    risks: {attack name: per-patient risk}. The combined risk only exists for
    patients every attack scored, so the sizes are reported instead of the
    non-overlap being dropped silently.
    """
    keysets = {name: set(r) for name, r in risks.items()}
    common = set.intersection(*keysets.values())
    report = {f"n_{name}": len(k) for name, k in keysets.items()}
    report["n_intersection"] = len(common)
    report.update({f"n_only_{name}": len(k - common) for name, k in keysets.items()})
    print(
        "  Risk combination ("
        + method
        + "): "
        + ", ".join(f"{k}={v}" for k, v in report.items())
    )
    return combine_risk_signals(list(risks.values()), method=method), report


def atypical_subgroup_report(
    attacks, atypical_keys, combined_risks, train_meta, percentile=99.0, min_n=30
):
    """
    Are atypical patients more exposed?

    attacks:        {name: (member_scores, nonmember_scores, higher_is_riskier)}
    atypical_keys:  keys of the atypical patients (patient_token_profile)
    combined_risks: {mode: per-patient combined risk of the members}

    Per attack, the membership AUC and TPR at low FPR within the atypical and
    the typical patients. Per mode, the Mantel-Haenszel OR (stratified by
    centre) of being atypical among the members in the combined risk tail.
    """
    rows = []
    for name, (member, nonmember, higher) in attacks.items():
        sign = 1.0 if higher else -1.0
        for group, inside in (("atypical", True), ("typical", False)):
            a = [sign * v for k, v in member.items() if (k in atypical_keys) == inside]
            b = [
                sign * v for k, v in nonmember.items() if (k in atypical_keys) == inside
            ]
            row = {
                "attack": name,
                "group": group,
                "n_members": len(a),
                "n_nonmembers": len(b),
            }
            if len(a) < min_n or len(b) < min_n:
                print(
                    f"  [{name}/{group}] n below {min_n} "
                    f"(members={len(a)}, nonmembers={len(b)}); no AUC reported."
                )
            else:
                s = summarize_attack(a, b, name=f"{name}/{group}", verbose=False)
                row.update(
                    auc=s["auc"],
                    ci_low=s["auc_ci"][0],
                    ci_high=s["auc_ci"][1],
                    tpr_at_1pct_fpr=s["tpr@fpr=0.01"],
                    tpr_at_01pct_fpr=s["tpr@fpr=0.001"],
                )
            rows.append(row)
    by_attack = pd.DataFrame(rows)

    tail = []
    for mode, risks in combined_risks.items():
        keys = list(risks)
        vals = np.array([risks[k] for k in keys])
        flagged = vals >= np.percentile(vals, percentile)
        exposure = np.array([k in atypical_keys for k in keys])
        strata = np.array([train_meta[k]["center"] for k in keys])
        mh = mantel_haenszel_or(flagged, exposure, strata)
        tail.append(
            {
                "mode": mode,
                "n_members": len(keys),
                "n_flagged": int(flagged.sum()),
                "atypical_share_all": float(exposure.mean()),
                "atypical_share_tail": (
                    float(exposure[flagged].mean()) if flagged.any() else np.nan
                ),
                **mh,
            }
        )
    tail = pd.DataFrame(tail)

    print("\n  Atypical vs typical, per attack:")
    print(by_attack.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(
        f"\n  Atypical enrichment in the P{percentile:g} combined risk tail "
        "(MH OR, stratified by centre):"
    )
    print(tail.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return {"by_attack": by_attack, "tail_enrichment": tail}


# multi-seed


def multiseed_summary(score_fn, seeds=(0, 1, 2, 3, 4), name="attack"):
    """
    `score_fn(seed)` must return (member_scores, nonmember_scores).
    Reports the mean and spread of the AUC across seeds, which is what a reviewer
    asks for when the whole pipeline ran with seed=0.
    """
    rows = []
    for seed in seeds:
        a, b = score_fn(seed)
        labels = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
        scores = np.concatenate([np.asarray(a), np.asarray(b)])
        rows.append({"seed": seed, "auc": float(roc_auc_score(labels, scores))})
    df = pd.DataFrame(rows)
    print(
        f"\nCross-seed stability [{name}]: "
        f"mean={df.auc.mean():.4f}, sd={df.auc.std(ddof=1):.4f}, "
        f"range=[{df.auc.min():.4f}, {df.auc.max():.4f}]"
    )
    return df


def multiseed_generation_pool(
    dist_matrices,
    real_index,
    ids,
    centers,
    candidates_per_cell=140,
    seeds=(0, 1, 2, 3, 4),
    chunk_aggregation="min",
):
    """
    Cheap multi-seed for attack C: reuses the distance matrices already computed
    and only redraws which columns (candidates) are in. No recomputation at all.

    dist_matrices: list of (real_idx, matrix) per cell.
    """
    out = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        scores = np.full(len(real_index), np.nan)
        for idx, mat in dist_matrices:
            take = min(candidates_per_cell, mat.shape[1])
            cols = rng.choice(mat.shape[1], take, replace=False)
            scores[idx] = mat[:, cols].min(axis=1)
        out[seed] = _aggregate_by_patient(
            ids, centers, scores, method=chunk_aggregation
        )
    return out


# testbed with injected memorisation, positive control
# The only experiment that separates "we found no leakage" from "the attack does
# not work". Without it, a TPR of 0.0008 at 0.1% FPR is indistinguishable from an
# inert attack, and a reviewer will say so.
#
# Requires training a small model (one centre, ~20-30k stays) with a known set of
# patients duplicated. Ground truth is known by construction.
#
# What it measures:
#   validity      the attack detects the duplicated patients
#   benefit       whether calibration separates memorised from merely seen
#   sensitivity   detection curve against the duplication level
#   failure mode  center_dropout_prob sweep, the condition under which `cal_center`
#                 stops calibrating (see the header note)


def inject_canaries(datamodule, n_canaries=200, dup_factor=50, seed=0):
    """
    Duplicates `n_canaries` patients from the training split `dup_factor` times.

    It operates on the Subset indices rather than the base dataset, so val and
    test stay untouched and the canaries remain legitimate members of the
    training set. Whole patients are duplicated, not individual chunks.

    Must be called after the datamodule has run `_setup()` (which happens in its
    __init__) and before `trainer.fit`.

    Returns the set of canary keys, in the same composite key format the rest of
    the module uses: `_composite_key(stay_id, centre_id)`, with the centre id
    resolved through CENTRE_MAP exactly as ICUDataset.__getitem__ does, and the
    exact int64 stay id, so the keys match the ones the scoring side builds
    through exact_batch_keys.
    """
    from torch.utils.data import Subset
    from syntheticicu.data_classes import CENTRE_MAP

    rng = np.random.default_rng(seed)
    base = datamodule.base_icu_dataset
    train_idx = np.asarray(datamodule.train_indices)

    stay_ids = (
        base.y[:, 0].numpy() if torch.is_tensor(base.y) else np.asarray(base.y)[:, 0]
    )
    sources = np.asarray(base.stay_sources)

    by_patient = {}
    for i in train_idx:
        centre_id = CENTRE_MAP.get(str(sources[i]).lower(), 0)
        key = _composite_key(stay_ids[i], centre_id)
        by_patient.setdefault(key, []).append(int(i))

    patient_keys = sorted(by_patient.keys())
    chosen = rng.choice(
        len(patient_keys), min(n_canaries, len(patient_keys)), replace=False
    )
    canary_keys = {patient_keys[c] for c in chosen}

    # dup_factor - 1 extra copies, since the original is already there
    augmented = list(train_idx)
    for key in canary_keys:
        for _ in range(dup_factor - 1):
            augmented.extend(by_patient[key])

    rng.shuffle(augmented)
    datamodule.train_indices = np.asarray(augmented)
    datamodule.train_icu_dataset = Subset(base, datamodule.train_indices)

    print(f"  Canaries injected: {len(canary_keys)} patients x{dup_factor}")
    print(f"  Train size: {len(train_idx)} -> {len(augmented)} rows")
    return canary_keys


def evaluate_testbed(
    model,
    canary_keys,
    decoder=None,
    k=0.2,
    dup_factor=None,
    batch_size=32,
    num_workers=0,
):
    """
    Three contrasts on the model with injected memorisation:

      canary_vs_holdout  validity: should give a high AUC if the attack works
      normal_vs_holdout  background leakage level
      canary_vs_normal   where calibration could help, telling memorised apart
                         from merely seen
    """
    if decoder is None:
        decoder = StaticsDecoder(model=model)
    audit = run_calibrated_membership_audit(
        model, k=k, decoder=decoder, batch_size=batch_size, num_workers=num_workers
    )

    rows = []
    for stat in ["raw", "cal_center", "cal_class", "cal_both"]:
        train_scores = audit["train_scores"][stat]
        holdout_scores = audit["holdout_scores"][stat]
        canary = [v for kk, v in train_scores.items() if kk in canary_keys]
        normal = [v for kk, v in train_scores.items() if kk not in canary_keys]
        holdout = list(holdout_scores.values())

        if len(canary) < 20:
            print(
                f"  WARNING: only {len(canary)} canaries found in the scores. "
                "Check the composite key format."
            )

        for name, a, b in [
            ("canary_vs_holdout", canary, holdout),
            ("normal_vs_holdout", normal, holdout),
            ("canary_vs_normal", canary, normal),
        ]:
            if len(a) < 10 or len(b) < 10:
                continue
            res = summarize_attack(a, b, name=f"{stat}/{name}", verbose=False)
            rows.append(
                {
                    "statistic": stat,
                    "contrast": name,
                    "auc": res["auc"],
                    "ci_low": res["auc_ci"][0],
                    "ci_high": res["auc_ci"][1],
                    "tpr@1%fpr": res["tpr@fpr=0.01"],
                    "n_pos": len(a),
                    "n_neg": len(b),
                }
            )

    df = pd.DataFrame(rows)
    print("\nTestbed results" + (f" (dup x{dup_factor})" if dup_factor else "") + ":")
    print(df.to_string(index=False))

    valid = df[(df.statistic == "raw") & (df.contrast == "canary_vs_holdout")]
    if len(valid) and valid.iloc[0]["auc"] < 0.7:
        print(
            "\n  Critical: the attack does not detect injected memorisation. The "
            "attack is inert and no null result from the main audit is "
            "interpretable."
        )
    elif len(valid):
        print(
            "\n  The attack detects the injected memorisation, so it has "
            "demonstrated power. The nulls from the main audit are informative."
        )

    # canary_vs_normal is the contrast where a swap calibration could help: telling
    # memorised patients apart from merely seen ones. Reported for each statistic
    # against raw, with no expected direction; the first grid had every calibration
    # below raw.
    cn = df[df.contrast == "canary_vs_normal"].set_index("statistic")
    if "raw" in cn.index:
        raw = cn.loc["raw"]
        print(
            f"\n  canary_vs_normal, calibration vs raw "
            f"(raw = {raw['auc']:.4f} [{raw['ci_low']:.3f}, {raw['ci_high']:.3f}]):"
        )
        for stat in ["cal_center", "cal_class", "cal_both"]:
            if stat not in cn.index:
                continue
            r = cn.loc[stat]
            if r["ci_low"] > raw["ci_high"]:
                verdict = "above raw (CIs do not overlap)"
            elif r["ci_high"] < raw["ci_low"]:
                verdict = "below raw (CIs do not overlap)"
            else:
                verdict = "not distinguishable from raw (CIs overlap)"
            print(
                f"    {stat:<11} {r['auc']:.4f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  "
                f"delta {r['auc'] - raw['auc']:+.4f}  {verdict}"
            )
    return df, audit


def replication_distances(generated, training, holdout, conditioned, cap, seed):
    """Score both synthetic-to-real directions and a distinct-stay baseline."""
    if cap < 1:
        raise ValueError("Positive candidate cap required")
    cells = []
    for group in [generated, training, holdout]:
        if not len(group["seqs"]) == len(group["classes"]) == len(group["centers"]):
            raise ValueError("Replication sequence and conditioning axes differ")
        cells.append(
            set(zip(group["classes"], group["centers"]))
            if conditioned
            else set(group["classes"])
        )
    if not cells[0] == cells[1] == cells[2]:
        raise ValueError("Replication conditioning cells differ; no fallback allowed")

    def score(queries, candidates):
        query_cells = (
            set(zip(queries["classes"], queries["centers"]))
            if conditioned
            else set(queries["classes"])
        )
        candidate_cells = (
            set(zip(candidates["classes"], candidates["centers"]))
            if conditioned
            else set(candidates["classes"])
        )
        if not query_cells.issubset(candidate_cells):
            raise ValueError("Missing attack-B conditioning cell; no fallback allowed")
        distances, _, cells = generation_scores_with_sweep(
            queries["seqs"],
            queries["classes"],
            queries["centers"],
            candidates["seqs"],
            candidates["classes"],
            candidates["centers"],
            use_center=conditioned,
            max_candidates_per_cell=cap,
            pool_sweep=None,
            seed=seed,
        )
        if not np.isfinite(distances).all():
            raise ValueError("Attack B did not cover every query")
        return distances, cells

    generated_train, train_cells = score(generated, training)
    generated_holdout, holdout_cells = score(generated, holdout)
    real_real, baseline_cells = score(holdout, training)
    return dict(
        generated_to_train=generated_train,
        generated_to_holdout=generated_holdout,
        holdout_to_train=real_real,
        train_cells=train_cells,
        holdout_cells=holdout_cells,
        baseline_cells=baseline_cells,
    )
