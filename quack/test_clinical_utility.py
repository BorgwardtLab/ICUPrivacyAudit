"""Exercise event-based cutoffs independently of measurement coverage and bins."""

import numpy as np
import pandas as pd
import pytest
import torch
import joblib
from types import SimpleNamespace
from functools import partial

from syntheticicu.clinical_utility import (
    ClinicalTaskDatamodule,
    ClinicalWindows,
    clinical_metadata,
)
from syntheticicu.clinical_sources import discrete_real_stays, load_clinical_source
from quack.test_numerical_task_parity import window_fixture
from syntheticicu.evaluation_numeric import EvaluationData
from syntheticicu.tasks_numeric import active_tasks, run_classifier_task
from syntheticicu import clinical_sources


def windows(stays, schema, endpoints):
    return ClinicalWindows(
        stays,
        schema,
        lookback=24,
        syn_lookback=0,
        window_size=4,
        feature_set="all",
        min_length=6,
        endpoints=endpoints,
        hours_max=9000,
    )


@pytest.mark.parametrize("endpoint", [40.0, 40.5])
def test_real_windows_use_completed_hours_before_clinical_cutoff(endpoint):
    stay, schema = window_fixture(length=80, outcome=True)
    original = stay["values"].clone()
    dataset = windows([stay], schema, {(stay["centre"], stay["stay_id"]): endpoint})
    assert len(dataset) == 13
    np.testing.assert_array_equal(dataset.times, np.arange(4, 17))
    assert dataset.reference_hours.dtype == torch.float64
    assert torch.all(dataset.reference_hours - dataset.times >= 24)
    assert dataset.reference_hours[-1] == endpoint
    assert dataset._values[0].shape[0] == 16
    torch.testing.assert_close(stay["values"], original, equal_nan=True)


def test_short_observation_coverage_and_ordinal_values_share_the_clinical_grid():
    stay, schema = window_fixture(length=10)
    key = (stay["centre"], stay["stay_id"])
    continuous = windows([stay], schema, {key: 40.5})
    ordinal = dict(stay, values=stay["values"].remainder(5))
    discrete = windows([ordinal], schema, {key: 40.5})
    np.testing.assert_array_equal(continuous.times, discrete.times)
    assert continuous._masks[0][10:].all()
    np.testing.assert_array_equal(continuous._values[0][15], continuous._values[0][9])


def test_synthetic_stays_need_no_los_and_keep_the_full_existing_representation():
    stay, schema = window_fixture(length=40)
    stay = dict(
        stay,
        split="synthetic",
        classifier_hours=torch.arange(1, 41),
        classifier_values=stay["values"],
    )
    dataset = windows([stay], schema, {})
    assert len(dataset) == 37
    assert dataset.times[-1] == 40
    assert dataset._values[0].shape[0] == 40


@pytest.mark.parametrize("los", [None, float("inf"), 0, -1])
def test_invalid_clinical_los_fails_without_an_observation_endpoint_fallback(
    tmp_path, los
):
    source = tmp_path / "meta.parquet"
    pd.DataFrame(
        [dict(centre="sic", stay_id=1, patient_uid=1, death_icu=False, los_icu=los)]
    ).to_parquet(source)
    with pytest.raises(ValueError, match="LOS"):
        clinical_metadata(source, 9000)


def test_missing_endpoint_and_oversized_grids_are_rejected():
    stay, schema = window_fixture()
    with pytest.raises(ValueError, match="endpoint"):
        windows([stay], schema, {})
    key = (stay["centre"], stay["stay_id"])
    with pytest.raises(ValueError, match="bound"):
        windows([stay], schema, {key: 9001})


def test_duplicate_clinical_keys_are_rejected(tmp_path):
    source = tmp_path / "meta.parquet"
    row = dict(centre="sic", stay_id=1, patient_uid=1, death_icu=False, los_icu=2.0)
    pd.DataFrame([row, row]).to_parquet(source)
    with pytest.raises(ValueError, match="duplicate"):
        clinical_metadata(source, 9000)


def test_stays_too_short_for_a_complete_pre_event_window_are_counted():
    stay, schema = window_fixture()
    key = (stay["centre"], stay["stay_id"])
    dataset = windows([stay], schema, {key: 27.5})
    assert len(dataset) == 0
    assert dataset.clinical_report["real_excluded_no_window"] == 1


def test_synthetic_windows_remain_uncut_in_every_split():
    module = ClinicalTaskDatamodule.__new__(ClinicalTaskDatamodule)
    module.full_dataset = SimpleNamespace(
        reference_hours=torch.tensor([40.5, 40.0], dtype=torch.float64),
        times=torch.tensor([16, 40]),
        stay_sources=pd.Series(["miiv", "syn"]),
        ids=torch.tensor([1, 2]),
    )
    for partition in ["train", "val", "test"]:
        setattr(module, f"{partition}_indices", np.asarray([0, 1]))
    module._apply_lookback_per_split()
    for partition in ["train", "val", "test"]:
        np.testing.assert_array_equal(getattr(module, f"{partition}_indices"), [0, 1])


def test_discrete_stay_axis_uses_exact_center_keys_and_new_metadata():
    source = SimpleNamespace(
        y=np.asarray([[2**54 + 1], [2**54 + 1]], dtype=np.int64),
        stay_sources=["miiv", "sic"],
        times=[np.arange(2), np.arange(2)],
        X_value_ids=[np.asarray([[0], [5]]), np.asarray([[4], [3]])],
    )
    metadata = pd.DataFrame(
        [
            dict(centre="sic", stay_id=2**54 + 1, patient_uid=8, death_icu=True),
            dict(centre="miiv", stay_id=2**54 + 1, patient_uid=9, death_icu=False),
        ]
    ).set_index(["centre", "stay_id"])
    stays = discrete_real_stays(source, metadata, ["hr_raw"], 5)
    assert stays[0]["patient_uid"] == 9
    assert stays[1]["death_icu"] is True
    assert stays[0]["stay_id"] == 2**54 + 1
    assert torch.isnan(stays[0]["values"][1, 0])


@pytest.mark.parametrize("bins", [2, 5, 15, 25, 50])
def test_discrete_clinical_source_supports_each_requested_bin_resolution(
    tmp_path, bins
):
    path = tmp_path / "real.pkl"
    source = SimpleNamespace(
        features=["hr_raw"],
        num_bins=bins,
        y=np.asarray([[1]], dtype=np.int64),
        stay_sources=["sic"],
        times=[np.arange(2)],
        X_value_ids=[np.asarray([[bins - 1], [bins]])],
    )
    joblib.dump(source, path)
    metadata = pd.DataFrame(
        [dict(centre="sic", stay_id=1, patient_uid=9, death_icu=False)]
    ).set_index(["centre", "stay_id"])
    configuration = SimpleNamespace(
        representation="discrete", num_bins=bins, real_path=str(path)
    )
    data = load_clinical_source(configuration, "null", metadata, False)
    assert len(data.bundle["schema"]["features"]["hr_raw"]["edges"]) == bins + 1
    assert data.bundle["stays"][0]["values"][0, 0] == bins - 1
    assert torch.isnan(data.bundle["stays"][0]["values"][1, 0])


def test_clinical_adapter_fits_and_scores_the_shared_real_only_classifier(tmp_path):
    source, schema = window_fixture(length=30)
    stays = [
        dict(
            source,
            stay_id=2**54 + index,
            patient_uid=2**54 + index,
            death_icu=bool(index % 2),
        )
        for index in range(4000)
    ]
    data = EvaluationData(tmp_path, {"schema": schema, "stays": stays}, [], {})
    endpoints = {(stay["centre"], stay["stay_id"]): 30.0 for stay in stays}
    factory = partial(ClinicalTaskDatamodule, data, endpoints, 9000)
    metrics, splits = run_classifier_task(
        data,
        active_tasks()[0],
        tmp_path,
        42,
        1,
        {"syn_lookback": 0, "n_folds": 2, "hopt_iter": 1, "bootstraps": 0},
        datamodule_factory=factory,
    )
    assert len(splits["test_ids"]) == 2400
    assert metrics["all"]["full_test_input"].shape[0] == 2400 * 3
    assert len(metrics["all"]["full_test_predictions"]) == 2400


def test_numerical_source_allows_quantile_ties(monkeypatch):
    data = SimpleNamespace(
        bundle={
            "schema": {
                "features": {
                    "hr_raw": {"edges": [0, 1, 2, 3, 4, 5]},
                    "crea_raw": {"edges": [0, 1, 2]},
                }
            }
        }
    )
    monkeypatch.setattr(
        clinical_sources, "load_evaluation_data", lambda *args, **kwargs: data
    )
    source = SimpleNamespace(
        representation="numerical", num_bins=5, roots={"null": "unused"}
    )
    assert load_clinical_source(source, "null", None, True) is data
