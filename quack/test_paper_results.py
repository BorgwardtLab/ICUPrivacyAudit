"""Exercise public exports and stay-level uncertainty without clinical records."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_curve, precision_recall_curve

from syntheticicu.paper_results import (
    prediction_groups,
    bootstrap_predictions,
    export_utility,
    closer_to_training,
)
from syntheticicu.value_distances import pair_distances


def result_fixture():
    index = pd.MultiIndex.from_tuples(
        [(10, 1, 0), (10, 2, 0), (20, 1, 5), (20, 2, 5), (30, 1, 0), (40, 1, 5)],
        names=["ids", "times", "onset"],
    )
    labels = pd.Series([0, 0, 1, 1, 0, 1], index=index)
    scores = [
        np.array([0.1, 0.2]),
        np.array([0.3, 0.9]),
        np.array([0.4]),
        np.array([0.6]),
    ]
    grouped_labels = [0, 1, 0, 1]
    grouped_scores = [0.2, 0.9, 0.4, 0.6]
    precision, recall, thresholds = precision_recall_curve(
        grouped_labels, grouped_scores
    )
    return dict(
        full_test_labels=labels,
        full_test_predictions=scores,
        test_roc_curve=roc_curve(grouped_labels, grouped_scores),
        test_pr_curve=(recall, precision, thresholds),
    )


def test_saved_utility_exports_match_stay_aggregation_and_have_no_ids(tmp_path):
    result = result_fixture()
    frame = prediction_groups(result)
    assert frame.prediction.tolist() == [0.2, 0.9, 0.4, 0.6]
    assert bootstrap_predictions(frame, 20)["bootstrap_valid"] > 0
    path = tmp_path / "results.joblib"
    joblib.dump(({"all": result}, {}), path)
    report = export_utility(path, tmp_path / "export", repeats=20)
    assert report["auroc"] == 1
    assert report["test_stays"] == 4
    assert list(pd.read_csv(tmp_path / "export/test_predictions.csv")) == [
        "label",
        "prediction",
    ]


def test_misaligned_predictions_are_rejected():
    result = result_fixture()
    result["full_test_predictions"] = []
    with pytest.raises(ValueError, match="cover each"):
        prediction_groups(result)


def test_replication_ties_count_as_half():
    summary = closer_to_training([0, 1, 2], [1, 1, 1])
    assert summary["closer_to_training"] == 0.5
    assert summary["ci_low"] <= 0.5 <= summary["ci_high"]


def test_dtw_matches_artificial_reference_cases():
    golden = json.loads(
        (Path(__file__).parent / "fixtures/dtw_l1_golden.json").read_text()
    )
    for case in golden["cases"]:

        def decode(record):
            return {
                key: np.asarray(value, dtype=float if key == "values" else int)
                for key, value in record.items()
            }

        measured = pair_distances(decode(case["left"]), decode(case["right"]))
        assert measured.shape == (1,)
        assert measured[0] == pytest.approx(case["dtw_l1"], abs=1e-14, rel=0)


def test_saved_attack_summary_name_does_not_override_manifest_name(tmp_path):
    from syntheticicu.paper_results import export_attack

    result = export_attack(
        tmp_path / "roc.npz",
        {1: 0.9, 2: 0.7},
        {3: 0.3, 4: 0.1},
        {"name": "attack", "auc": 1.0},
    )
    assert result["auc"] == 1
    assert "name" not in result
