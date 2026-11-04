"""Paper classifier scores preserve the original maximum-per-stay aggregation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from syntheticicu.utils import early_warning_scorer


def test_grouped_curves_match_direct_reference_with_ties_and_unsorted_hours():
    index = pd.MultiIndex.from_tuples(
        [(2, 3, 30), (1, 8, 30), (2, 1, 30), (1, 2, 30)],
        names=["ids", "times", "onset"],
    )
    labels = pd.Series([1, 0, 1, 0], index=index)
    predictions = np.array([0.8, 0.3, 0.8, 0.2])
    expected_labels, expected_predictions = [0, 1], [0.3, 0.8]
    for name, reference in [
        ("roc_auc", roc_auc_score),
        ("average_precision", average_precision_score),
    ]:
        score = early_warning_scorer(
            y=labels, precomputed_predictions=predictions, metric=name, bootstraps=False
        )
        assert score == reference(expected_labels, expected_predictions)
    actual = early_warning_scorer(
        y=labels, precomputed_predictions=predictions, metric="roc_curve"
    )
    for measured, expected in zip(
        actual, roc_curve(expected_labels, expected_predictions)
    ):
        np.testing.assert_array_equal(measured, expected)
    precision, recall, thresholds = precision_recall_curve(
        expected_labels, expected_predictions
    )
    actual = early_warning_scorer(
        y=labels, precomputed_predictions=predictions, metric="pr_curve"
    )
    for measured, expected in zip(actual, (recall, precision, thresholds)):
        np.testing.assert_array_equal(measured, expected)


def test_paper_scorer_rejects_unsupported_metric():
    with pytest.raises(ValueError, match="metric"):
        early_warning_scorer(metric="unavailable")
