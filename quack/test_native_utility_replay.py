"""Verify that clinical cohort replay rejects changed samples or clocks."""

from types import SimpleNamespace

import pytest
import torch

from syntheticicu.native_utility_replay import verify_windows


def dataset(times=(3, 4), labels=(0, 0)):
    return SimpleNamespace(
        ids=torch.tensor([10, 10]),
        times=torch.tensor(times),
        reference_hours=torch.tensor([40, 40]),
        y=torch.tensor(labels),
        onset=torch.tensor([0, 0]),
    )


def test_same_windows_pass_in_different_order():
    assert verify_windows(dataset(), dataset(), [0, 1], [1, 0], "test") == 2


@pytest.mark.parametrize("actual", [dataset(times=(3, 5)), dataset(labels=(0, 1))])
def test_changed_clock_or_label_fails(actual):
    with pytest.raises(ValueError, match="windows differ"):
        verify_windows(dataset(), actual, [0, 1], [0, 1], "test")


def test_recover_original_observations_for_excluded_stay(tmp_path, monkeypatch):
    import pandas as pd

    from syntheticicu.native_utility_source import add_missing_reference_stays

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    pd.DataFrame(
        dict(centre=["aumc"], stay_id=[3], stay_time=[2.5], urine_raw=[80.0])
    ).to_parquet("data/core.parquet")
    pd.DataFrame(dict(centre=["aumc"], stay_id=[3], weight_static=[40.0])).to_parquet(
        "data/metadata.parquet"
    )
    stay = dict(
        centre="aumc", stay_id=3, patient_uid=7, split="not_audited", death_icu=False
    )
    reference = SimpleNamespace(bundle=dict(stays=[stay]))
    data = SimpleNamespace(
        bundle=dict(stays=[], schema=dict(features={"urine_raw": {}}))
    )
    recovered = add_missing_reference_stays(data, reference)
    assert len(recovered) == 1
    assert data.bundle["stays"][0]["hours"].tolist() == [2]
    assert data.bundle["stays"][0]["values"].tolist() == [[2.0]]
    assert "values" not in stay
