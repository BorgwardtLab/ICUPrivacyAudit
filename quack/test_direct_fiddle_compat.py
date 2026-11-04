from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

from syntheticicu.data_classes import (
    EarlyWarningDataset,
    FIDDLETokenizer,
    ICUDataset,
    ICUDatamodule,
)
from syntheticicu.demo_utils import resolve_demo_model_cfg


FEATURES = ["hr_raw", "temp_raw"]
STATICS = ["female_static", "age_static", "height_static", "weight_static"]


def make_direct_fiddle_shard(stay_id, centre="miiv"):
    data_table = pd.DataFrame(
        {
            "stay_id": [stay_id],
            "centre": [centre],
            "los": [6],
            "death_icu": [1 if stay_id % 2 else 0],
            "death_icu_onset": [4],
        }
    )
    return SimpleNamespace(
        data_table=data_table,
        features=FEATURES,
        statics_set=STATICS,
        num_bins=5,
        X_value_ids=[
            torch.tensor(
                [
                    [2, 5],  # hr observed, temp missing
                    [0, 1],  # both observed
                    [5, 5],  # all missing, should not emit feature tokens
                    [1, 2],
                    [3, 4],
                    [4, 0],
                ],
                dtype=torch.int32,
            )
        ],
        X_feature_ids=[torch.arange(len(FEATURES)).repeat(6, 1)],
        statics=torch.tensor([[1, 3, 5, 0]], dtype=torch.int32),
        processed_statics=torch.tensor([[1, 3, 5, 0]], dtype=torch.int32),
        lengths=torch.tensor([6], dtype=torch.long),
        times=[torch.arange(6, dtype=torch.float32)],
        missing_mask=[torch.ones(6, len(FEATURES), dtype=torch.bool)],
        y=torch.tensor([[stay_id, 1 if stay_id % 2 else 0, 4]], dtype=torch.long),
        stay_sources=[centre],
        X_FIDDLE_bins={"hr_raw": [1, 2, 3, 4], "temp_raw": [1, 2, 3, 4]},
        S_FIDDLE_bins={"age_static": [1, 2, 3, 4]},
        static_missing_token_id=5,
        static_pad_token_id=6,
    )


def dense_timestep(active_bins, num_features=len(FEATURES), num_bins=5):
    row = np.zeros(num_features * num_bins, dtype=np.int64)
    for feature_idx, bin_idx in enumerate(active_bins):
        if bin_idx is not None:
            row[feature_idx * num_bins + bin_idx] = 1
    return row


def dense_demographics(female, age, height, weight):
    values = [female, age, height, weight]
    bins = [2, 5, 5, 5]
    offset = 0
    statics = torch.zeros(sum(bins), dtype=torch.long)
    for value, n_bins in zip(values, bins):
        statics[offset + value] = 1
        offset += n_bins
    return statics


def test_direct_sequence_matches_dense_fiddle_tokens_and_time_positions():
    tokenizer = FIDDLETokenizer(
        features=FEATURES,
        num_bins=5,
        include_demographics=True,
        add_time_point_positional_embeddings=True,
    )
    dense_sequence = [
        dense_timestep([2, None]),
        dense_timestep([0, 1]),
        dense_timestep([None, None]),
        dense_timestep([1, 2]),
    ]
    direct_values = torch.tensor(
        [
            [2, 5],
            [0, 1],
            [5, 5],
            [1, 2],
        ],
        dtype=torch.long,
    )
    dense_statics = dense_demographics(female=1, age=3, height=2, weight=0)
    direct_statics = torch.tensor([1, 3, 2, 0], dtype=torch.long)

    dense_tokens, dense_time_positions = tokenizer.encode_sequence(
        dense_sequence,
        demographics=dense_statics,
    )
    direct_tokens, direct_time_positions = tokenizer.encode_direct_sequence(
        direct_values,
        demographics=direct_statics,
        statics_set=STATICS,
    )

    expected_tokens = [
        tokenizer.vocab["<SOS>"],
        tokenizer.vocab["female_static_bin_1"],
        tokenizer.vocab["age_static_bin_3"],
        tokenizer.vocab["height_static_bin_2"],
        tokenizer.vocab["weight_static_bin_0"],
        tokenizer.vocab["<GAP>"],
        tokenizer.vocab["hr_raw_bin_2"],
        tokenizer.vocab["<GAP>"],
        tokenizer.vocab["hr_raw_bin_0"],
        tokenizer.vocab["temp_raw_bin_1"],
        tokenizer.vocab["<GAP>"],
        tokenizer.vocab["<GAP>"],
        tokenizer.vocab["hr_raw_bin_1"],
        tokenizer.vocab["temp_raw_bin_2"],
        tokenizer.vocab["<EOS>"],
    ]
    expected_time_positions = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4]

    assert direct_tokens == dense_tokens == expected_tokens
    assert direct_time_positions == dense_time_positions == expected_time_positions


def test_direct_demographics_match_dense_observed_bins_and_lock_missing_defaults():
    tokenizer = FIDDLETokenizer(
        features=FEATURES,
        num_bins=5,
        include_demographics=True,
    )

    dense_tokens = tokenizer.encode_demographics(
        dense_demographics(female=1, age=3, height=2, weight=0)
    )
    direct_tokens = tokenizer.encode_direct_demographics(
        torch.tensor([1, 3, 2, 0], dtype=torch.long), statics_set=STATICS
    )
    assert direct_tokens == dense_tokens

    missing_tokens = tokenizer.encode_direct_demographics(
        torch.tensor([5, 5, 5, 5], dtype=torch.long), statics_set=STATICS
    )
    assert missing_tokens == [
        tokenizer.vocab["female_static_bin_1"],
        tokenizer.vocab["age_static_bin_2"],
        tokenizer.vocab["height_static_bin_2"],
        tokenizer.vocab["weight_static_bin_2"],
    ]


def test_icudatamodule_forwards_filters_and_truncates_direct_fiddle(tmp_path):
    for stay_id in [101, 202, 303, 404, 505]:
        joblib.dump(make_direct_fiddle_shard(stay_id), tmp_path / f"{stay_id}.pkl")

    datamodule = ICUDatamodule(
        data_path=str(tmp_path),
        min_length=1,
        max_length=3,
        ids_to_keep=[101, 202, 303],
        val_fraction=1 / 3,
        test_fraction=1 / 3,
        batch_size=1,
        num_workers=1,
        return_label="death_icu",
        add_time_point_positional_embeddings=True,
        include_demographics=True,
    )

    dataset = datamodule.base_icu_dataset
    assert dataset.max_length == 3
    assert dataset.min_length == 1
    assert dataset.y[:, 0].tolist() == [101, 202, 303]
    assert all(len(x) == 3 for x in dataset.X_value_ids)
    assert all(len(t) == 3 for t in dataset.times)
    assert all(len(m) == 3 for m in dataset.missing_mask)
    # Once tokenized, lengths count serialized tokens rather than hourly rows.
    assert dataset.lengths.tolist() == [11, 11, 11]


def test_generate_config_default_demographics_model_is_tvae():
    config_text = Path("config/generate.yaml").read_text()
    assert "gen_demo: tvae" in config_text
    assert (
        resolve_demo_model_cfg("tvae")["_target_"] == "sdv.single_table.TVAESynthesizer"
    )


def test_direct_fiddle_pickle_linearizes_to_lm_tokens(tmp_path):
    pkl_path = tmp_path / "direct.pkl"
    joblib.dump(make_direct_fiddle_shard(101), pkl_path)

    dataset = ICUDataset(
        pkl_path=str(pkl_path),
        feature_set=FEATURES,
        min_length=1,
        num_workers=1,
        return_label="death_icu",
        add_time_point_positional_embeddings=True,
        include_demographics=True,
    )

    vocab = dataset.tokenizer.vocab
    tokens = dataset.X_tokens[0].tolist()
    time_positions = dataset.X_time_positions[0].tolist()

    assert dataset.is_direct_fiddle
    assert tokens[0] == vocab["<SOS>"]
    assert tokens[-1] == vocab["<EOS>"]
    assert len(tokens) == len(time_positions)
    assert vocab["<GAP>"] in tokens
    assert vocab["hr_raw_bin_2"] in tokens
    assert vocab["temp_raw_bin_1"] in tokens
    assert dataset.statics.dtype == torch.float32
    assert dataset.statics.shape == (1, 17)
    assert dataset.statics.sum().item() == len(STATICS)

    item = dataset[0]
    assert len(item) == 5
    assert item[0].shape == item[1].shape


def test_direct_fiddle_directory_shards_are_merged(tmp_path):
    joblib.dump(make_direct_fiddle_shard(101, centre="miiv"), tmp_path / "a.pkl")
    joblib.dump(make_direct_fiddle_shard(202, centre="eicu"), tmp_path / "b.pkl")

    dataset = ICUDataset(
        pkl_path=str(tmp_path),
        feature_set=FEATURES,
        min_length=1,
        num_workers=1,
        return_label="death_icu",
        add_time_point_positional_embeddings=False,
        include_demographics=False,
    )

    assert len(dataset) == 2
    assert len(dataset.X_tokens) == 2
    assert dataset.y[:, 0].tolist() == [101, 202]
    assert set(dataset.stay_sources.tolist()) == {"miiv", "eicu"}


def test_direct_fiddle_shards_preserve_duplicate_stay_ids_across_centers(tmp_path):
    miiv = make_direct_fiddle_shard(101, centre="miiv")
    miiv.data_table["death_icu"] = 0
    miiv.y[:, 1] = 0
    miiv.X_value_ids[0][0, 0] = 0

    eicu = make_direct_fiddle_shard(101, centre="eicu")
    eicu.data_table["death_icu"] = 1
    eicu.y[:, 1] = 1
    eicu.X_value_ids[0][0, 0] = 4

    joblib.dump(miiv, tmp_path / "a_miiv.pkl")
    joblib.dump(eicu, tmp_path / "b_eicu.pkl")

    dataset = ICUDataset(
        pkl_path=str(tmp_path),
        feature_set=FEATURES,
        min_length=1,
        num_workers=1,
        return_label="death_icu",
        add_time_point_positional_embeddings=False,
        include_demographics=False,
    )

    assert len(dataset) == 2
    assert dataset.y[:, 0].tolist() == [101, 101]
    assert dataset.stay_sources.tolist() == ["miiv", "eicu"]

    miiv_item = dataset[0]
    eicu_item = dataset[1]
    assert miiv_item[-1][1:].tolist() == [0.0, 4.0, 0.0]
    assert eicu_item[-1][1:].tolist() == [1.0, 4.0, 1.0]
    assert dataset.tokenizer.vocab["hr_raw_bin_0"] in miiv_item[0]
    assert dataset.tokenizer.vocab["hr_raw_bin_4"] in eicu_item[0]


def test_direct_fiddle_flat_decoder_values_impute_missing_tokens():
    dataset = EarlyWarningDataset.__new__(EarlyWarningDataset)
    dataset.num_bins = 5
    dataset.impute_missing = "forwardfill"

    values = torch.tensor(
        [
            [2, 5],
            [3, 1],
            [5, 5],
        ],
        dtype=torch.int32,
    )

    imputed = dataset._direct_values_for_flat_decoding(values)

    assert imputed.tolist() == [
        [2.0, 2.0],
        [3.0, 1.0],
        [3.0, 1.0],
    ]


def test_direct_fiddle_early_warning_windows_are_flat_decoder_ready(tmp_path):
    pkl_path = tmp_path / "direct.pkl"
    joblib.dump(make_direct_fiddle_shard(101, centre="miiv"), pkl_path)

    dataset = EarlyWarningDataset(
        label="death_icu",
        real_path=str(pkl_path),
        sources=["miiv"],
        feature_set=FEATURES,
        min_length=1,
        min_onset=1,
        num_workers=1,
        window_size=3,
        include_missing_mask=True,
        missingness_threshold=1.0,
        impute_missing="forwardfill",
        include_demographics=True,
    )

    feature_tensor, statics, label, time, onset, stay_id = dataset[0]
    value_rows = feature_tensor[: len(FEATURES)]
    missing_rows = feature_tensor[len(FEATURES) :]

    assert feature_tensor.shape == (len(FEATURES) * 2, 3)
    assert torch.isfinite(value_rows).all()
    assert value_rows.min() >= 0
    assert value_rows.max() < dataset.num_bins
    assert missing_rows.shape == (len(FEATURES), 3)
    assert missing_rows.sum() > 0
    assert statics.dtype == torch.long
    assert statics.shape == (len(STATICS),)
    assert label.ndim == 0
    assert time.ndim == 0
    assert onset.ndim == 0
    assert int(stay_id.item()) == 101
