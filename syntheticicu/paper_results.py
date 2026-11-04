"""Identifier-free result exports for the paper's utility and privacy panels."""

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


BANDS = [0, 100, 500, 2000, 10000, np.inf]


def curve_metrics(roc, pr):
    fpr, tpr = np.asarray(roc[0]), np.asarray(roc[1])
    recall, precision = np.asarray(pr[0]), np.asarray(pr[1])
    return dict(
        auroc=float(np.trapezoid(tpr, fpr)),
        average_precision=float(-np.sum(np.diff(recall) * precision[:-1])),
    )


def prediction_groups(result):
    """Match saved per-stay prediction arrays to the scorer's sorted group order."""
    labels = result["full_test_labels"]
    groups = list(labels.groupby(level="ids", sort=True))
    predictions = result["full_test_predictions"]
    if len(groups) != len(predictions):
        raise ValueError("Saved predictions do not cover each test stay")
    rows = []
    for (_, group), values in zip(groups, predictions):
        values = np.asarray(values).reshape(-1)
        if len(values) != len(group) or not np.isfinite(values).all():
            raise ValueError("Saved prediction/window alignment differs")
        selected = int(np.argmax(values))
        rows.append((int(group.iloc[selected]), float(values[selected])))
    return pd.DataFrame(rows, columns=["label", "prediction"])


def bootstrap_predictions(frame, repeats=2000, seed=0):
    """Resample complete test stays; fitting and window selection are unchanged."""
    if repeats < 1 or len(frame) < 2:
        raise ValueError("Positive bootstrap count and at least two stays required")
    labels = frame.label.to_numpy()
    predictions = frame.prediction.to_numpy()
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(predictions).all():
        raise ValueError("Invalid binary labels or predictions")
    rng = np.random.default_rng(seed)
    auc, ap = [], []
    for _ in range(repeats):
        indices = rng.integers(0, len(frame), len(frame))
        if len(np.unique(labels[indices])) < 2:
            continue
        auc.append(roc_auc_score(labels[indices], predictions[indices]))
        ap.append(average_precision_score(labels[indices], predictions[indices]))
    if not auc:
        raise ValueError("No bootstrap replicate contains both outcome classes")
    return dict(
        auroc_ci=np.quantile(auc, [0.025, 0.975]).tolist(),
        average_precision_ci=np.quantile(ap, [0.025, 0.975]).tolist(),
        bootstrap_requested=repeats,
        bootstrap_valid=len(auc),
    )


def export_utility(path, destination, repeats=2000, seed=0):
    performance, _ = joblib.load(path, mmap_mode="r")
    result = performance["all"]
    metrics = curve_metrics(result["test_roc_curve"], result["test_pr_curve"])
    frame = prediction_groups(result)
    measured = dict(
        auroc=roc_auc_score(frame.label, frame.prediction),
        average_precision=average_precision_score(frame.label, frame.prediction),
    )
    for name in measured:
        if not np.isclose(measured[name], metrics[name], rtol=0, atol=1e-12):
            raise ValueError(
                f"Saved {name} curve does not reproduce grouped predictions"
            )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    frame.sample(frac=1, random_state=seed).to_csv(
        destination / "test_predictions.csv", index=False
    )
    roc, pr = result["test_roc_curve"], result["test_pr_curve"]
    np.savez_compressed(
        destination / "curves.npz",
        fpr=roc[0],
        tpr=roc[1],
        recall=pr[0],
        precision=pr[1],
    )
    return dict(
        **metrics,
        **bootstrap_predictions(frame, repeats, seed),
        test_stays=len(frame),
        test_prevalence=float(frame.label.mean()),
    )


def export_attack(destination, members, nonmembers, summary, negate=False):
    a = np.asarray(list(members.values()), dtype=float)
    b = np.asarray(list(nonmembers.values()), dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite attack scores")
    for key, count in [("n_members", len(a)), ("n_nonmembers", len(b))]:
        if key in summary and summary[key] != count:
            raise ValueError("Saved attack population counts disagree with scores")
    labels = np.r_[np.ones(len(a)), np.zeros(len(b))]
    scores = np.r_[a, b] * (-1 if negate else 1)
    auc = roc_auc_score(labels, scores)
    if not np.isclose(auc, summary["auc"], rtol=0, atol=1e-12):
        raise ValueError("Attack scores do not reproduce saved AUC")
    fpr, tpr, _ = roc_curve(labels, scores)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, fpr=fpr, tpr=tpr)
    return dict(
        {key: value for key, value in summary.items() if key != "name"},
        n_members=len(a),
        n_nonmembers=len(b),
        auc=float(auc),
    )


def rarity_and_groups(audit):
    scores = audit["per_patient"]["attack_a"]
    rarity = audit["token_profile"]["patients"].rarest_token_patients
    reports = []
    channels = [("A", "weights", scores["train"]["raw"], scores["holdout"]["raw"], 1)]
    channels += [
        ("C", mode, value["train_patient_scores"], value["holdout_patient_scores"], -1)
        for mode, value in audit["generation_mia"].items()
    ]
    for attack, mode, member_scores, holdout_scores, sign in channels:
        for lower, upper in zip(BANDS[:-1], BANDS[1:]):
            a = [
                sign * value
                for key, value in member_scores.items()
                if lower <= rarity.loc[key] < upper
            ]
            b = [
                sign * value
                for key, value in holdout_scores.items()
                if lower <= rarity.loc[key] < upper
            ]
            if min(len(a), len(b)) < 30:
                continue
            auc = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
            reports.append(
                dict(
                    attack=attack,
                    mode=mode,
                    lower=lower,
                    upper=upper,
                    auc=float(auc),
                    n_members=len(a),
                    n_nonmembers=len(b),
                )
            )
    meta = audit["per_patient"]["train_meta"]
    profiles = [tuple(meta[key]["statics"]) for key in scores["train"]["raw"]]
    counts = Counter(profiles)
    sizes = np.asarray([counts[profile] for profile in profiles])
    return pd.DataFrame(reports), pd.DataFrame({"members_sharing_profile": sizes})


def closer_to_training(train, holdout):
    train, holdout = np.asarray(train), np.asarray(holdout)
    if train.shape != holdout.shape or train.ndim != 1 or len(train) < 2:
        raise ValueError("Paired generated-to-real distances required")
    scores = (train < holdout).astype(float) + 0.5 * (train == holdout)
    mean = float(scores.mean())
    half = 1.959963984540054 * float(scores.std(ddof=1)) / np.sqrt(len(scores))
    return dict(
        closer_to_training=mean,
        ci_low=max(0, mean - half),
        ci_high=min(1, mean + half),
        n=len(scores),
    )
