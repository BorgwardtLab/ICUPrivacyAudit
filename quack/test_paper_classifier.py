"""Check the real classifier fitting path against its direct scientific specification."""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real

from scripts import train_classifiers
from syntheticicu.tasks_numeric import active_tasks, classifier_arguments
from syntheticicu.utils import early_warning_scorer


def artificial_partition(stays, offset, generator):
    identities = np.repeat(np.arange(stays) + offset, 2)
    labels = np.repeat(np.arange(stays) % 2, 2)
    values = generator.normal(size=(len(labels), 3)) + labels[:, None] * 0.4
    index = pd.MultiIndex.from_arrays(
        [identities, np.tile([1, 2], stays), np.full(len(labels), 40)],
        names=["ids", "times", "onset"],
    )
    return pd.DataFrame(values), pd.Series(labels, index=index)


def test_paper_classifier_retains_fit_and_prediction_contract(tmp_path, monkeypatch):
    generator = np.random.default_rng(13)
    train_X, train_y = artificial_partition(20, 0, generator)
    test_X, test_y = artificial_partition(8, 100, generator)
    args = classifier_arguments(active_tasks()[0], tmp_path, 42, 1)
    args.n_folds, args.hopt_iter = 2, 2
    groups = train_y.index.get_level_values("ids").to_numpy()
    folds = list(
        StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42).split(
            train_X, train_y, groups
        )
    )
    reference = BayesSearchCV(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        penalty="l2",
                        max_iter=1000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        {"classifier__C": Real(0.0001, 1000, prior="log-uniform")},
        cv=folds,
        n_iter=2,
        n_jobs=1,
        scoring=partial(
            early_warning_scorer, metric="roc_auc", threshold=None, bootstraps=False
        ),
        refit=True,
    )
    np.random.seed(42)
    reference.fit(train_X, train_y, groups=groups)

    class Partition:
        def __init__(self, name, size):
            self.partition, self.size = name, size

        def __len__(self):
            return self.size

    module = SimpleNamespace(
        train_dataset=Partition("train", len(train_y)),
        test_dataset=Partition("test", len(test_y)),
        setup=lambda stage: None,
    )

    def extract(options, dataset, force_single_pass=False):
        return (
            (train_X, train_y, 1)
            if dataset.partition == "train"
            else (test_X, test_y, 1)
        )

    monkeypatch.setattr(train_classifiers, "extract_tabular_data", extract)
    np.random.seed(42)
    measured, splits = train_classifiers.main(
        args, datamodule_factory=lambda **options: module, max_parallel_jobs=1
    )
    expected = reference.predict_proba(test_X)[:, 1]
    actual = np.concatenate(measured["all"]["full_test_predictions"])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert (
        measured["all"]["estimator"].get_params()["classifier__C"]
        == reference.best_params_["classifier__C"]
    )
    np.testing.assert_array_equal(
        splits["test_ids"], np.unique(test_y.index.get_level_values("ids"))
    )
