"""End-to-end contracts for the independent numerical workflow."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from syntheticicu.data_numeric import (
    collate_numerical,
    prepare_numerical_data,
    repeated_training_indices,
)
from syntheticicu.workflow_numeric import (
    generate_numerical,
    train_numerical,
)


@pytest.fixture
def numerical_workflow(tmp_path):
    core, meta, splits = [], [], []
    for patient in range(12):
        split = "train" if patient < 8 else "val" if patient < 10 else "test"
        for visit in range(2):
            center = "a" if visit == 0 else "b"
            stay = patient
            meta.append(
                dict(
                    centre=center,
                    stay_id=stay,
                    patient_uid=2**54 + patient,
                    death_icu=bool(patient % 2),
                    female_static=patient % 2,
                    age_static=30.0 + patient,
                )
            )
            splits.append(dict(centre=center, stay_id=stay, split=split))
            for hour in range(3):
                core.append(
                    dict(
                        centre=center,
                        stay_id=stay,
                        stay_time=float(hour),
                        hr_raw=60.0 + patient + hour if patient < 8 else 300.0,
                        temp_raw=36.0 + patient / 10 + hour / 10,
                    )
                )
    pd.DataFrame(core).to_parquet(tmp_path / "core.parquet", index=False)
    pd.DataFrame(meta).to_parquet(tmp_path / "meta.parquet", index=False)
    pd.DataFrame(splits).to_parquet(tmp_path / "splits.parquet", index=False)
    cfg = OmegaConf.load(Path(__file__).parents[1] / "config/numerical_workflow.yaml")
    model_cfg = OmegaConf.load(
        Path(__file__).parents[1] / "config/numerical_model.yaml"
    )
    del cfg.defaults
    cfg.model = model_cfg
    cfg.output_dir = str(tmp_path / "run")
    cfg.data.core_path = str(tmp_path / "core.parquet")
    cfg.data.meta_path = str(tmp_path / "meta.parquet")
    cfg.data.split_manifest = str(tmp_path / "splits.parquet")
    cfg.data.features = ["hr_raw", "temp_raw"]
    cfg.data.demographics = ["female_static", "age_static"]
    cfg.data.centers = ["a", "b"]
    cfg.data.max_patients = None
    cfg.data.max_hours = 3
    cfg.data.max_tokens = 12
    cfg.data.num_bins = 2
    cfg.model.config.hidden_size = 16
    cfg.model.config.intermediate_size = 32
    cfg.model.config.num_hidden_layers = 1
    cfg.model.config.num_attention_heads = 2
    cfg.model.config.num_key_value_heads = 2
    cfg.train.max_steps = 1
    cfg.train.batch_size = 2
    cfg.train.accelerator = "cpu"
    cfg.train.limit_val_batches = 1
    cfg.generate.num_samples = 8
    cfg.generate.batch_size = 4
    cfg.generate.max_new_tokens = 16
    cfg.generate.device = "cpu"
    cfg.evaluate.max_hours = 3
    return cfg


def test_parquet_identity_train_only_fitting_and_chunks(numerical_workflow):
    bundle = prepare_numerical_data(numerical_workflow.data)
    assert bundle["schema"]["features"]["hr_raw"]["edges"][-1] == 69.0
    assert bundle["report"]["clipped"]["test"] > 0
    by_patient = {}
    for stay in bundle["stays"]:
        by_patient.setdefault(stay["patient_uid"], set()).add(stay["split"])
    assert all(len(splits) == 1 for splits in by_patient.values())
    assert min(by_patient) == 2**54
    assert len(bundle["stays"]) == 24
    assert len(bundle["records"]) > len(bundle["stays"])
    for record in bundle["records"]:
        assert len(record["input_ids"]) <= 12
        assert record["input_ids"].shape == record["residuals"].shape
        assert record["input_ids"].shape == record["time_positions"].shape


def test_joint_shuffle_preserves_token_value_time_alignment(numerical_workflow):
    bundle = prepare_numerical_data(numerical_workflow.data)
    records = bundle["records"][:2]
    torch.manual_seed(4)
    batch = collate_numerical(records, shuffle=True)
    for row, original in enumerate(records):
        original_pairs = sorted(
            zip(
                original["input_ids"].tolist(),
                original["residuals"].tolist(),
                original["time_positions"].tolist(),
            )
        )
        length = len(original["input_ids"])
        actual_pairs = sorted(
            zip(
                batch["input_ids"][row, :length].tolist(),
                batch["residuals"][row, :length].tolist(),
                batch["time_positions"][row, :length].tolist(),
            )
        )
        assert actual_pairs == original_pairs


def test_split_manifest_rejects_patient_overlap(numerical_workflow):
    path = numerical_workflow.data.split_manifest
    splits = pd.read_parquet(path)
    splits.loc[1, "split"] = "test"
    splits.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="patient"):
        prepare_numerical_data(numerical_workflow.data)


def test_canaries_repeat_only_training_patients(numerical_workflow):
    bundle = prepare_numerical_data(numerical_workflow.data)
    indices, canaries = repeated_training_indices(bundle, 2, 3, seed=5)
    assert len(canaries) == 2
    for index in indices:
        assert bundle["records"][index]["split"] == "train"
    for index, record in enumerate(bundle["records"]):
        if record["split"] == "train":
            assert indices.count(index) == (
                3 if record["patient_uid"] in canaries else 1
            )


def test_automatic_splits_are_reproducible_and_patient_disjoint(numerical_workflow):
    cfg = numerical_workflow.data
    cfg.split_manifest = None
    first = prepare_numerical_data(cfg)
    second = prepare_numerical_data(cfg)
    partitions = {}
    for stay in first["stays"]:
        partitions.setdefault(stay["patient_uid"], set()).add(stay["split"])
    assert all(len(splits) == 1 for splits in partitions.values())
    assert {stay["split"] for stay in first["stays"]} == {"train", "val", "test"}
    assert [
        (stay["centre"], stay["stay_id"], stay["split"]) for stay in first["stays"]
    ] == [(stay["centre"], stay["stay_id"], stay["split"]) for stay in second["stays"]]


def test_out_of_range_error_policy_rejects_heldout_extremes(numerical_workflow):
    numerical_workflow.data.out_of_range = "error"
    with pytest.raises(ValueError, match="outside training"):
        prepare_numerical_data(numerical_workflow.data)


def test_repeated_quantiles_collapse_to_nonempty_bins(numerical_workflow):
    cfg = numerical_workflow.data
    core = pd.read_parquet(cfg.core_path)
    core.loc[core.stay_id < 8, "hr_raw"] = core.loc[core.stay_id < 8, "stay_id"] % 2
    core.to_parquet(cfg.core_path, index=False)
    cfg.num_bins = 5
    bundle = prepare_numerical_data(cfg)
    assert bundle["schema"]["features"]["hr_raw"]["edges"] == [0.0, 1.0]


def test_categorical_raw_columns_cannot_be_used_as_continuous(numerical_workflow):
    cfg = numerical_workflow.data
    core = pd.read_parquet(cfg.core_path)
    core["hr_raw"] = "high"
    core.to_parquet(cfg.core_path, index=False)
    with pytest.raises(ValueError, match="continuous numerical"):
        prepare_numerical_data(cfg)


def test_tiny_parquet_training_generation_and_mortality(numerical_workflow):
    cfg = numerical_workflow
    bundle = prepare_numerical_data(cfg.data)
    output = Path(cfg.output_dir)
    output.mkdir()
    torch.save(bundle, output / "prepared.pt")
    checkpoint = train_numerical(cfg)
    assert checkpoint.is_file()
    generated = generate_numerical(cfg)
    core = pd.read_parquet(generated / "core.parquet")
    meta = pd.read_parquet(generated / "meta.parquet")
    assert len(meta) == 8
    assert set(meta.death_icu) == {False, True}
    assert set(core.stay_id) == set(meta.stay_id)
    assert "hr_raw" in core
    assert "death_icu" not in core
    assert np.isfinite(core.hr_raw.dropna()).all()
