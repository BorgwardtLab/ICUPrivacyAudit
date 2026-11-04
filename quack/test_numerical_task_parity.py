"""Compare numerical tasks directly with the active binned implementations."""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from syntheticicu.data_classes import EarlyWarningDataset
from syntheticicu.evaluation_numeric import (
    EvaluationData,
    classification_rows_from_tokens,
)
from syntheticicu.tasks_numeric import (
    NumericalWindows,
    active_tasks,
    classifier_arguments,
    run_classifier_task,
)
from syntheticicu.privacy_bridge_numeric import NumericalAuditModel, export_audit_corpus
from syntheticicu.privacy_audit_ext import (
    compute_batch_scores_calibrated,
    prepare_sequences,
)
from syntheticicu.data_numeric import collate_numerical, prepare_numerical_data
from syntheticicu.workflow_numeric import (
    prepare_numerical,
    train_numerical,
    load_numerical_model,
    generate_numerical,
)
from quack.test_numerical_workflow import numerical_workflow
from syntheticicu.conditioned_release_numeric import generate_conditioned_release


def window_fixture(length=40, outcome=False):
    values = (np.arange(length * 2).reshape(length, 2) % 5).astype(np.float32)
    values[0, 0], values[4, 1] = np.nan, np.nan
    stay = dict(
        centre="miiv",
        stay_id=2**54 + 1,
        patient_uid=2**54 + 3,
        split="train",
        death_icu=outcome,
        demographics={},
        hours=torch.arange(length),
        values=torch.from_numpy(values),
    )
    schema = {
        "features": {
            name: {"edges": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]}
            for name in ["crea_raw", "hr_raw"]
        },
        "demographics": {},
    }
    return stay, schema


@pytest.mark.parametrize("outcome", [False, True])
def test_windows_match_binned_cutoff_and_imputation(outcome):
    stay, schema = window_fixture(outcome=outcome)
    dataset = NumericalWindows(
        [stay], schema, lookback=24, window_size=4, feature_set="all", min_length=6
    )
    legacy = EarlyWarningDataset.__new__(EarlyWarningDataset)
    legacy.X = [np.where(np.isnan(stay["values"].numpy()), 5, stay["values"].numpy())]
    legacy.times, legacy.lengths = [torch.arange(40)], [40]
    legacy.y_full = torch.tensor([[1, int(outcome), 0, 0]])
    legacy.lookback, legacy.syn_lookback, legacy.window_size = 24, 24, 4
    legacy.synthetic, legacy.num_bins, legacy.impute_missing = False, 5, "forwardfill"
    legacy.return_label = "death_icu"
    legacy.is_direct_fiddle = True
    legacy._cut_stays_before_end()
    legacy._create_sliding_windows()
    assert len(dataset) == len(legacy.window_tensor) == 13
    assert dataset.times.max().item() == 15
    for index, expected in enumerate(legacy.window_tensor):
        np.testing.assert_allclose(dataset[index][0][:2].numpy().T, expected.numpy())
    changed = copy.deepcopy(stay)
    changed["values"][16:] = 10000
    other = NumericalWindows(
        [changed], schema, lookback=24, window_size=4, feature_set="all", min_length=6
    )
    for index in range(len(dataset)):
        torch.testing.assert_close(dataset[index][0], other[index][0])


def test_short_stays_and_training_only_synthetic_lookback():
    short, schema = window_fixture(length=27)
    dataset = NumericalWindows(
        [short], schema, lookback=24, window_size=4, feature_set="all", min_length=6
    )
    assert len(dataset) == 0
    synthetic = {
        **short,
        "split": "synthetic",
        "centre": "__null__",
        "classifier_hours": short["hours"] + 1,
        "classifier_values": short["values"],
    }
    dataset = NumericalWindows(
        [synthetic],
        schema,
        lookback=24,
        syn_lookback=0,
        window_size=4,
        feature_set="all",
        min_length=6,
    )
    assert len(dataset) == 24
    assert dataset.stay_sources.iloc[0] == "syn"


def test_active_task_matrix_matches_main_binned_launcher():
    tasks = active_tasks()
    assert [task["index"] for task in tasks] == list(range(11))
    assert [task["mix"] for task in tasks[:11]] == [i / 10 for i in range(11)]
    args = classifier_arguments(tasks[0], "outputs/parity", 42, 1)
    assert (args.window_size, args.lookback, args.max_length) == (4, 24, None)
    assert args.extracted_features == "median-min-max"
    assert args.include_missing_mask and not args.include_statics
    assert args.n_folds == 5 and args.hopt_iter == 50


def test_privacy_bridge_uses_residuals_and_shared_conditioning_swaps(
    numerical_workflow,
):
    cfg = numerical_workflow
    for field in ["core_path", "meta_path", "split_manifest"]:
        path = cfg.data[field]
        frame = pd.read_parquet(path)
        frame["centre"] = frame.centre.replace({"a": "miiv", "b": "eicu"})
        frame.to_parquet(path, index=False)
    cfg.data.centers = ["miiv", "eicu"]
    prepare_numerical(cfg)
    path = train_numerical(cfg)
    core, checkpoint = load_numerical_model(path, "cpu")
    bundle = torch.load(path.parent / "prepared.pt", weights_only=True)
    data = EvaluationData(path.parent, bundle, [], {})
    model = NumericalAuditModel(core, data)
    batch = model._datamodule.collate_fn(bundle["records"][:2])
    exact = [r["stay_id"] for r in bundle["records"][:2]]
    centers = [model.center_code[r["centre"]] for r in bundle["records"][:2]]
    scores = compute_batch_scores_calibrated(model, batch, exact, centers, k=0.2)
    assert len(scores["s_true"]) == 2
    torch.testing.assert_close(
        batch.residuals, collate_numerical(bundle["records"][:2])["residuals"]
    )
    expected = model.score_privacy_batch(batch, exact, centers, k=0.2)
    np.testing.assert_allclose(scores["s_flip"], expected["s_flip"])
    assert model.core is core

    def forward(input_ids, class_labels, center_ids, time_positions, attention_mask):
        centers = center_ids.long().clone()
        centers[centers == 6] = core.num_centers
        return core(
            input_ids=input_ids,
            residuals=batch.residuals,
            class_labels=class_labels.long(),
            center_ids=centers,
            time_positions=time_positions,
        ).logits

    proxy = SimpleNamespace(
        vocabulary=model.vocabulary,
        device=torch.device("cpu"),
        actual_prefix_len=0,
        forward=forward,
    )
    reference = compute_batch_scores_calibrated(proxy, batch, exact, centers, k=0.2)
    for name in ["s_true", "s_nullc", "s_flip", "mean_true", "lengths"]:
        np.testing.assert_allclose(scores[name], reference[name], rtol=1e-5)
    sequences, ids = model.collect_privacy_sequences(
        model._datamodule.train_icu_dataset, collect_statics=False
    )
    prepared = prepare_sequences(sequences, canonicalize=True, chunk_zero_only=True)
    assert len(prepared) == 16


def test_shared_classifier_fits_and_scores_windows(tmp_path):
    source, schema = window_fixture(length=30)
    stays = [
        {
            **source,
            "stay_id": 2**54 + index,
            "patient_uid": 2**54 + index,
            "death_icu": bool(index % 2),
        }
        for index in range(4000)
    ]
    data = EvaluationData(tmp_path, {"schema": schema, "stays": stays}, [], {})
    metrics, splits = run_classifier_task(
        data,
        active_tasks()[0],
        tmp_path,
        42,
        1,
        {"n_folds": 2, "hopt_iter": 1, "bootstraps": 0},
    )
    assert len(splits["test_ids"]) == 2400
    assert len(splits["train_ids"]) == 391
    assert metrics["all"]["full_test_input"].shape[0] == 2400 * 3
    assert len(metrics["all"]["full_test_predictions"]) == 2400
    identities = pd.read_parquet(tmp_path / "identities.parquet")
    assert identities.stay_id.iloc[1] == 2**54 + 1
    assert not identities.alias_id.duplicated().any()


def test_conditioned_release_preserves_null_and_recovers(numerical_workflow):
    cfg = numerical_workflow
    prepare_numerical(cfg)
    train_numerical(cfg)
    null = generate_numerical(cfg)
    before = (null / "generation.json").read_bytes()
    destination = generate_conditioned_release(cfg)
    meta = pd.read_parquet(destination / "meta.parquet")
    assert len(meta) == 8
    assert not meta.stay_id.duplicated().any()
    assert set(meta.centre) == {"a", "b"}
    assert (null / "generation.json").read_bytes() == before
    modification = (destination / "core.parquet").stat().st_mtime_ns
    assert generate_conditioned_release(cfg) == destination
    assert (destination / "core.parquet").stat().st_mtime_ns == modification


def test_synthetic_token_row_alignment_matches_legacy(tmp_path):
    stay, schema = window_fixture(length=30)
    vocabulary = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "<GAP>": 3,
        "female_static_bin_0": 4,
    }
    for name, spec in schema["features"].items():
        spec["tokens"] = [f"{name}_bin_{i}" for i in range(5)]
        for token in spec["tokens"]:
            vocabulary[token] = len(vocabulary)
    schema["vocabulary"] = vocabulary
    ids, values = [1, 4, 3], [float("nan")] * 3
    for hour in range(30):
        ids.extend([vocabulary["crea_raw_bin_1"], vocabulary["hr_raw_bin_2"], 3])
        values.extend([1.0, 2.0, float("nan")])
        if hour == 10:
            ids.append(3)
            values.append(float("nan"))
    ids.append(2)
    values.append(float("nan"))
    generated = tmp_path / "generated"
    generated.mkdir()
    torch.save(
        [dict(input_ids=torch.tensor([ids]), values=torch.tensor([values]))],
        generated / "tokens.pt",
    )
    rows = classification_rows_from_tokens(tmp_path, schema)[0]
    assert len(rows) == 31
    legacy = EarlyWarningDataset.__new__(EarlyWarningDataset)
    legacy.features, legacy.num_bins = list(schema["features"]), 5
    inverse = {value: name for name, value in vocabulary.items()}
    matrices, lengths = legacy._map_tok_to_idx(
        [[inverse[token] for token in ids if token != 1]]
    )
    binned = matrices[0].reshape(31, 2, 5)
    legacy.X = [np.where(binned.sum(-1) == 0, 5, binned.argmax(-1))]
    legacy.times, legacy.lengths = [torch.arange(1, 32)], [31]
    legacy.y_full = torch.tensor([[1, 0, 0, 0]])
    legacy.lookback = legacy.syn_lookback = 24
    legacy.synthetic, legacy.window_size = True, 4
    legacy.return_label, legacy.impute_missing = "death_icu", "forwardfill"
    legacy.is_direct_fiddle = True
    legacy._cut_stays_before_end()
    legacy._create_sliding_windows()
    source = {
        **stay,
        "split": "synthetic",
        "classifier_values": rows,
        "classifier_hours": torch.arange(1, 32),
    }
    numeric = NumericalWindows([source], schema, 24, 4, "all", 6)
    assert len(numeric) == len(legacy.window_tensor) == 4
    for index, expected in enumerate(legacy.window_tensor):
        np.testing.assert_allclose(numeric[index][0][:2].numpy().T, expected.numpy())


def test_synthetic_token_rows_reject_corruption(numerical_workflow, tmp_path):
    schema = prepare_numerical_data(numerical_workflow.data)["schema"]
    generated = tmp_path / "generated"
    generated.mkdir()
    batch = dict(input_ids=torch.tensor([[1, 99999, 2]]), values=torch.zeros(1, 3))
    torch.save([batch], generated / "tokens.pt")
    with pytest.raises(ValueError, match="Unknown generated token"):
        classification_rows_from_tokens(tmp_path, schema)
    batch["input_ids"] = batch["input_ids"].float()
    torch.save([batch], generated / "tokens.pt")
    with pytest.raises(ValueError, match="integer batch matrix"):
        classification_rows_from_tokens(tmp_path, schema)
