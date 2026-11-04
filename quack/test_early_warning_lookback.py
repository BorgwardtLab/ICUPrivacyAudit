from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from syntheticicu.data_classes import EarlyWarningDatamodule, EarlyWarningDataset


FEATURES = ["hr_raw", "temp_raw"]
STATICS = ["female_static", "age_static", "height_static", "weight_static"]
NUM_BINS = 5
WINDOW_SIZE = 4
LOOKBACK_HOURS = 24


def make_stays_shard(lengths, labels, onsets=None, label="death_icu", centres=None):
    """Direct FIDDLE shard with one stay per length, hours 0..length-1."""
    rng = np.random.default_rng(0)
    stay_ids = list(range(100, 100 + len(lengths)))
    centres = ["miiv"] * len(lengths) if centres is None else centres
    # by default an onset column that disagrees with the stay end, which death
    # must ignore
    onsets = [3] * len(lengths) if onsets is None else onsets
    data_table = pd.DataFrame(
        {
            "stay_id": stay_ids,
            "centre": centres,
            "los": lengths,
            label: labels,
            f"{label}_onset": onsets,
        }
    )
    values = [
        torch.tensor(rng.integers(0, NUM_BINS, (n, len(FEATURES))), dtype=torch.int32)
        for n in lengths
    ]
    return SimpleNamespace(
        data_table=data_table,
        features=FEATURES,
        statics_set=STATICS,
        num_bins=NUM_BINS,
        X_value_ids=values,
        X_feature_ids=[torch.arange(len(FEATURES)).repeat(n, 1) for n in lengths],
        statics=torch.tensor([[1, 3, 2, 0]] * len(lengths), dtype=torch.int32),
        processed_statics=torch.tensor(
            [[1, 3, 2, 0]] * len(lengths), dtype=torch.int32
        ),
        lengths=torch.tensor(lengths, dtype=torch.long),
        times=[torch.arange(n, dtype=torch.float32) for n in lengths],
        missing_mask=[torch.ones(n, len(FEATURES), dtype=torch.bool) for n in lengths],
        y=torch.tensor(
            [[s, l, o] for s, l, o in zip(stay_ids, labels, onsets)], dtype=torch.long
        ),
        stay_sources=centres,
        X_FIDDLE_bins={f: [1, 2, 3, 4] for f in FEATURES},
        S_FIDDLE_bins={"age_static": [1, 2, 3, 4]},
        static_missing_token_id=NUM_BINS,
        static_pad_token_id=NUM_BINS + 1,
    )


def real_dataset(tmp_path, lengths, labels, lookback, onsets=None, label="death_icu"):
    pkl_path = tmp_path / "stays.pkl"
    joblib.dump(make_stays_shard(lengths, labels, onsets, label), pkl_path)
    return EarlyWarningDataset(
        label=label,
        real_path=str(pkl_path),
        sources=["miiv"],
        feature_set=FEATURES,
        min_length=1,
        min_onset=1,
        max_length=None,
        num_workers=1,
        window_size=WINDOW_SIZE,
        lookback=lookback,
        include_missing_mask=True,
        missingness_threshold=1.0,
        impute_missing="forwardfill",
        include_demographics=True,
    )


def windows_by_stay(dataset):
    ids = dataset.ids.numpy().astype(int)
    return {stay_id: np.flatnonzero(ids == stay_id) for stay_id in np.unique(ids)}


def test_lookback_cuts_real_stays_before_their_last_hour(tmp_path):
    lengths = [40, 50, 35, 60]
    labels = [1, 0, 1, 0]
    dataset = real_dataset(tmp_path, lengths, labels, LOOKBACK_HOURS)

    windows = windows_by_stay(dataset)
    assert len(dataset.missingness_tensor) == len(dataset.window_tensor)
    assert len(dataset.window_tensor) == len(dataset.times)
    for stay_id, length, label in zip(range(100, 104), lengths, labels):
        index = windows[stay_id]
        last_hour = length - 1
        assert len(index) == length - LOOKBACK_HOURS - WINDOW_SIZE + 1
        assert dataset.times[index].max().item() == last_hour - LOOKBACK_HOURS
        expected_onset = last_hour if label else 0
        assert (dataset.onset[index] == expected_onset).all()


def test_lookback_cuts_other_outcomes_before_their_onset_column(tmp_path):
    lengths = [80, 80, 70]
    labels = [1, 1, 0]
    onsets = [50, 60, 0]
    dataset = real_dataset(
        tmp_path, lengths, labels, LOOKBACK_HOURS, onsets, label="aki"
    )

    windows = windows_by_stay(dataset)
    # positives end lookback before onset, the control before its last hour
    for stay_id, last_kept_hour, onset in [
        (100, 50 - LOOKBACK_HOURS, 50),
        (101, 60 - LOOKBACK_HOURS, 60),
        (102, 69 - LOOKBACK_HOURS, 0),
    ]:
        index = windows[stay_id]
        assert dataset.times[index].max().item() == last_kept_hour
        assert (dataset.onset[index] == onset).all()


def test_lookback_without_onset_for_other_outcomes_fails_on_synthetic(tmp_path):
    tokens = [["hr_raw_bin_2", "temp_raw_bin_1", "<GAP>"] * 40 + ["<EOS>"]] * 2
    syn_path = tmp_path / "syn.joblib"
    joblib.dump((tokens, [1, 0], ["syn", "syn"]), syn_path)

    with pytest.raises(ValueError, match="aki_onset"):
        EarlyWarningDataset(
            label="aki",
            syn_path=str(syn_path),
            num_bins=NUM_BINS,
            feature_set=FEATURES,
            min_length=1,
            window_size=WINDOW_SIZE,
            lookback=LOOKBACK_HOURS,
            include_missing_mask=False,
        )


def test_stays_shorter_than_lookback_plus_window_are_dropped(tmp_path):
    short_length = LOOKBACK_HOURS + WINDOW_SIZE - 1
    dataset = real_dataset(tmp_path, [short_length, 40], [1, 1], LOOKBACK_HOURS)

    assert set(windows_by_stay(dataset)) == {101}


def test_zero_lookback_keeps_every_window_and_cut_windows_are_a_prefix(tmp_path):
    lengths = [30, 45]
    labels = [1, 0]
    full = real_dataset(tmp_path, lengths, labels, lookback=0)
    cut = real_dataset(tmp_path, lengths, labels, LOOKBACK_HOURS)

    full_windows = windows_by_stay(full)
    cut_windows = windows_by_stay(cut)
    for stay_id, length in zip((100, 101), lengths):
        full_index = full_windows[stay_id]
        assert len(full_index) == length - WINDOW_SIZE + 1
        assert full.times[full_index].tolist() == list(range(WINDOW_SIZE - 1, length))
        cut_index = cut_windows[stay_id]
        for i_full, i_cut in zip(full_index, cut_index):
            assert torch.equal(full.window_tensor[i_full], cut.window_tensor[i_cut])


def test_lookback_cuts_synthetic_stays_and_sets_their_onset(tmp_path):
    lengths = [40, 30]
    labels = [1, 0]
    tokens = [
        ["hr_raw_bin_2", "temp_raw_bin_1", "<GAP>"] * n + ["<EOS>"] for n in lengths
    ]
    syn_path = tmp_path / "syn.joblib"
    joblib.dump((tokens, labels, ["syn"] * len(lengths)), syn_path)

    dataset = EarlyWarningDataset(
        label="death_icu",
        syn_path=str(syn_path),
        num_bins=NUM_BINS,
        feature_set=FEATURES,
        min_length=1,
        window_size=WINDOW_SIZE,
        lookback=LOOKBACK_HOURS,
        include_missing_mask=False,
    )

    windows = windows_by_stay(dataset)
    for stay_id, length, label in zip(sorted(windows), lengths, labels):
        index = windows[stay_id]
        # synthetic hours run 1..length
        assert len(index) == length - LOOKBACK_HOURS - WINDOW_SIZE + 1
        assert dataset.times[index].max().item() == length - LOOKBACK_HOURS
        assert (dataset.onset[index] == (length if label else 0)).all()


def test_datamodule_applies_lookback_to_train_val_and_test(tmp_path):
    rng = np.random.default_rng(1)
    # centre split: miiv stays all go to train, eicu stays are split into val
    # and test, so a few stays cover all three subsets
    lengths = rng.integers(30, 80, size=24).tolist()
    labels = [i % 2 for i in range(24)]
    centres = ["miiv"] * 8 + ["eicu"] * 16
    pkl_path = tmp_path / "stays.pkl"
    joblib.dump(make_stays_shard(lengths, labels, centres=centres), pkl_path)
    last_hour = {100 + i: n - 1 for i, n in enumerate(lengths)}

    datamodule = EarlyWarningDatamodule(
        label="death_icu",
        real_path=str(pkl_path),
        sources=[["miiv"], ["eicu"]],
        min_length=1,
        max_length=None,
        min_onset=1,
        window_size=WINDOW_SIZE,
        lookback=LOOKBACK_HOURS,
        include_missing_mask=False,
        split_fractions=(0.6, 0.2, 0.2),
        split_mode="centre",
        val_set=True,
        batch_size=8,
        num_workers=1,
        random_state=0,
        feature_set=FEATURES,
    )
    datamodule.setup("fit")

    for subset in (
        datamodule.train_dataset,
        datamodule.val_dataset,
        datamodule.test_dataset,
    ):
        assert len(subset) > 0
        for i in range(len(subset)):
            _, _, _, window_time, _, stay_id = subset[i]
            cutoff = last_hour[int(stay_id.item())] - LOOKBACK_HOURS
            assert window_time.item() <= cutoff


def test_lookback_range_is_rejected_for_sliding_windows():
    with pytest.raises(AssertionError, match="single lookback"):
        EarlyWarningDataset(
            label="death_icu", window_size=WINDOW_SIZE, lookback=(12, 24)
        )


def write_synthetic_corpus(path, lengths, labels):
    tokens = [
        ["hr_raw_bin_2", "temp_raw_bin_1", "<GAP>"] * n + ["<EOS>"] for n in lengths
    ]
    joblib.dump((tokens, labels, ["syn"] * len(lengths)), path)


def split_windows(datamodule):
    dataset = datamodule.full_dataset
    hours_before = (dataset.reference_hours - dataset.times).numpy()
    sources = np.asarray(dataset.stay_sources)
    return {
        name: (hours_before[subset.indices], sources[subset.indices])
        for name, subset in (
            ("train", datamodule.train_dataset),
            ("val", datamodule.val_dataset),
            ("test", datamodule.test_dataset),
        )
    }


def test_syn_lookback_only_applies_to_synthetic_stays_in_train(tmp_path):
    rng = np.random.default_rng(2)
    lengths = rng.integers(40, 80, size=60).tolist()
    syn_path = tmp_path / "syn.joblib"
    write_synthetic_corpus(syn_path, lengths, [i % 2 for i in range(60)])

    datamodule = EarlyWarningDatamodule(
        label="death_icu",
        lookback=LOOKBACK_HOURS,
        syn_lookback=0,
        syn_path=str(syn_path),
        num_bins=NUM_BINS,
        feature_set=FEATURES,
        min_length=1,
        window_size=WINDOW_SIZE,
        include_missing_mask=False,
        split_fractions=(0.6, 0.2, 0.2),
        split_mode="stay_id",
        val_set=True,
        batch_size=8,
        num_workers=1,
        random_state=0,
    )
    datamodule.setup("fit")

    windows = split_windows(datamodule)
    # synthetic training stays keep their last hours; val/test never do
    assert windows["train"][0].min() == 0
    for name in ("val", "test"):
        hours_before, _ = windows[name]
        assert len(hours_before) > 0
        assert hours_before.min() >= LOOKBACK_HOURS


def test_syn_lookback_leaves_real_stays_at_lookback(tmp_path):
    rng = np.random.default_rng(3)
    lengths = rng.integers(40, 80, size=24).tolist()
    labels = [i % 2 for i in range(24)]
    pkl_path = tmp_path / "stays.pkl"
    centres = ["miiv"] * 8 + ["eicu"] * 16
    joblib.dump(make_stays_shard(lengths, labels, centres=centres), pkl_path)
    syn_path = tmp_path / "syn.joblib"
    write_synthetic_corpus(syn_path, [60] * 10, [i % 2 for i in range(10)])

    datamodule = EarlyWarningDatamodule(
        label="death_icu",
        lookback=LOOKBACK_HOURS,
        syn_lookback=0,
        real_path=str(pkl_path),
        syn_path=str(syn_path),
        percentage_synthetic=1.0,
        sources=[["miiv"], ["eicu"]],
        min_length=1,
        max_length=None,
        min_onset=1,
        window_size=WINDOW_SIZE,
        include_missing_mask=False,
        split_fractions=(0.6, 0.2, 0.2),
        split_mode="centre",
        val_set=True,
        batch_size=8,
        num_workers=1,
        random_state=0,
        feature_set=FEATURES,
    )
    datamodule.setup("fit")

    windows = split_windows(datamodule)
    train_hours, train_sources = windows["train"]
    assert train_hours[train_sources == "syn"].min() == 0
    assert train_hours[train_sources != "syn"].min() >= LOOKBACK_HOURS
    for name in ("val", "test"):
        hours_before, sources = windows[name]
        assert (sources != "syn").all()
        assert hours_before.min() >= LOOKBACK_HOURS
