"""Full-scale numerical settings and resumable training contracts."""

from pathlib import Path

import pytest
import pandas as pd
import torch
from hydra import compose, initialize_config_dir

from syntheticicu.data_numeric import collate_numerical, prepare_numerical_data
from syntheticicu.workflow_numeric import train_numerical
from quack.test_numerical_workflow import numerical_workflow


def test_full_config_matches_recorded_binned_backbone_and_optimizer():
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"
    ):
        cfg = compose(config_name="numerical_full")
    assert cfg.data.max_patients is None
    assert cfg.data.max_hours is None
    assert len(cfg.data.features) == 58
    assert cfg.model.config.hidden_size == 512
    assert cfg.model.config.num_hidden_layers == 6
    assert cfg.model.config.num_attention_heads == 8
    assert (
        cfg.train.batch_size * cfg.train.devices * cfg.train.accumulate_grad_batches
        == 512
    )
    assert cfg.train.max_epochs == 100
    assert cfg.train.max_steps == 200000
    assert cfg.train.learning_rate == pytest.approx(0.0021649915375620173)
    assert cfg.train.weight_decay == pytest.approx(0.04739107925821598)
    assert list(cfg.train.betas) == [0.9, 0.95]
    assert cfg.train.warmup_steps == 2000
    assert cfg.train.shuffle_within_hour is True


def test_paper_continuous_config_preserves_effective_batch_and_order():
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[1] / "config"), version_base="1.3"
    ):
        cfg = compose(config_name="numerical_matched")
    assert cfg.train.shuffle_within_hour is False
    assert (
        cfg.train.devices * cfg.train.accumulate_grad_batches * cfg.train.batch_size
        == 512
    )
    assert cfg.train.val_check_interval // cfg.train.accumulate_grad_batches == 125
    assert cfg.data.split_unit == "reference_stay"
    assert cfg.generate.positive_fraction == 0.07


def test_disabled_shuffle_keeps_stored_token_value_order(numerical_workflow):
    records = prepare_numerical_data(numerical_workflow.data)["records"][:2]
    batch = collate_numerical(records, shuffle=False)
    for index, record in enumerate(records):
        for key in ["input_ids", "residuals", "time_positions"]:
            assert torch.equal(batch[key][index, : len(record[key])], record[key])


def test_no_hour_cap_preserves_long_stays(numerical_workflow):
    cfg = numerical_workflow
    core = pd.read_parquet(cfg.data.core_path)
    late = core.iloc[[0]].copy()
    late["stay_time"] = 30.0
    pd.concat([core, late], ignore_index=True).to_parquet(
        cfg.data.core_path, index=False
    )
    cfg.data.max_hours = None
    bundle = prepare_numerical_data(cfg.data)
    assert bundle["schema"]["max_hours"] == 31
    assert max(len(stay["hours"]) for stay in bundle["stays"]) == 31


def test_full_preprocessing_normalizes_urine_and_keeps_missing_weights_missing(
    numerical_workflow,
):
    cfg = numerical_workflow.data
    core = pd.read_parquet(cfg.core_path)
    core["urine_raw"] = core.hr_raw
    core.to_parquet(cfg.core_path, index=False)
    meta = pd.read_parquet(cfg.meta_path)
    meta["weight_static"] = 70.0
    meta.loc[meta.stay_id == 1, "weight_static"] = float("nan")
    meta.to_parquet(cfg.meta_path, index=False)
    cfg.features = ["urine_raw", "temp_raw"]
    cfg.demographics = ["female_static", "age_static", "weight_static"]
    cfg.normalize_urine_by_weight = True
    bundle = prepare_numerical_data(cfg)
    first = next(stay for stay in bundle["stays"] if stay["stay_id"] == 0)
    missing = next(stay for stay in bundle["stays"] if stay["stay_id"] == 1)
    assert first["values"][0, 0].item() == pytest.approx(60 / 70)
    assert torch.isnan(missing["values"][:, 0]).all()


@pytest.mark.parametrize("shuffle_within_hour", [True, False])
def test_accumulation_scheduler_and_resume(numerical_workflow, shuffle_within_hour):
    cfg = numerical_workflow
    cfg.train.shuffle_within_hour = shuffle_within_hour
    cfg.train.accumulate_grad_batches = 2
    cfg.train.warmup_steps = 1
    cfg.train.scheduler_steps = 10
    cfg.train.checkpoint_every_n_steps = 1
    cfg.train.max_steps = 1
    output = Path(cfg.output_dir)
    output.mkdir()
    torch.save(prepare_numerical_data(cfg.data), output / "prepared.pt")
    train_numerical(cfg)
    checkpoint = torch.load(
        output / "checkpoints/last.ckpt", map_location="cpu", weights_only=False
    )
    assert checkpoint["global_step"] == 1
    assert checkpoint["lr_schedulers"][0]["last_epoch"] == 1
    cfg.train.max_steps = 2
    cfg.overwrite = True
    cfg.train.resume = True
    train_numerical(cfg)
    exported = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    assert exported["training"]["steps"] == 2
    assert exported["training"]["effective_batch"] == 4
    assert (
        exported["training"]["settings"]["shuffle_within_hour"] == shuffle_within_hour
    )
    cfg.train.shuffle_within_hour = not shuffle_within_hour
    with pytest.raises(ValueError, match="Resume checkpoint"):
        train_numerical(cfg)
