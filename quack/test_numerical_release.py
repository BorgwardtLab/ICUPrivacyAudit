"""Recovery and identity contracts for long numerical generation jobs."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

import syntheticicu.generation_numeric as generation
from syntheticicu.conditioned_release_numeric import (
    generate_conditioned_release,
    prepare_conditioned_release,
    generate_conditioned_center,
    merge_conditioned_release,
)
from syntheticicu.data_numeric import prepare_numerical_data
from syntheticicu.workflow_numeric import train_numerical
from quack.test_numerical_workflow import numerical_workflow


def assert_releases_equal(first, second):
    for name in ["core", "meta"]:
        pd.testing.assert_frame_equal(
            pd.read_parquet(first / f"{name}.parquet"),
            pd.read_parquet(second / f"{name}.parquet"),
            check_exact=True,
        )
    a = torch.load(first / "tokens.pt", weights_only=True)
    b = torch.load(second / "tokens.pt", weights_only=True)
    assert len(a) == len(b)
    for batch_a, batch_b in zip(a, b):
        assert batch_a.keys() == batch_b.keys()
        for key in batch_a:
            torch.testing.assert_close(
                batch_a[key], batch_b[key], rtol=0, atol=0, equal_nan=True
            )
    assert (first / "generation.json").read_text() == (
        second / "generation.json"
    ).read_text()


def test_parallel_batches_preserve_sequential_seeds_labels_ids_and_values(
    trained_release,
):
    cfg = trained_release
    cfg.generate.num_samples = 30
    cfg.generate.positive_fraction = 0.07
    sequential = generation.generate_release(cfg)
    settings = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    settings.output_dir = str(Path(cfg.output_dir).parent / "parallel")
    root = Path(settings.output_dir)
    root.mkdir()
    (root / "model.pt").symlink_to(Path(cfg.output_dir) / "model.pt")
    generation.prepare_release(settings)
    for shard in [2, 0, 1]:
        generation.generate_release(
            settings, shard_index=shard, shard_count=3, merge=False
        )
    parallel = generation.merge_release(settings)
    assert_releases_equal(sequential, parallel)
    assert pd.read_parquet(parallel / "meta.parquet").death_icu.sum() == 2


def test_merge_rejects_missing_batches_and_invalid_shards(trained_release):
    generation.prepare_release(trained_release)
    with pytest.raises(FileNotFoundError, match="batch"):
        generation.merge_release(trained_release)
    with pytest.raises(ValueError, match="shard"):
        generation.generate_release(trained_release, shard_index=2, shard_count=2)


def test_parallel_centers_match_sequential_generation(trained_release):
    cfg = trained_release
    cfg.generate.num_samples = 30
    cfg.generate.positive_fraction = 0.07
    sequential = generate_conditioned_release(cfg)
    settings = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    root = Path(cfg.output_dir).parent / "parallel_centers"
    root.mkdir()
    (root / "model.pt").symlink_to(Path(cfg.output_dir) / "model.pt")
    settings.output_dir = str(root)
    prepare_conditioned_release(settings)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        merge_conditioned_release(settings)
    for center in [1, 0]:
        generate_conditioned_center(settings, center)
    parallel = merge_conditioned_release(settings)
    assert_releases_equal(sequential, parallel)
    assert pd.read_parquet(parallel / "meta.parquet").death_icu.sum() == 2


@pytest.fixture
def trained_release(numerical_workflow):
    cfg = numerical_workflow
    root = Path(cfg.output_dir)
    root.mkdir()
    torch.save(prepare_numerical_data(cfg.data), root / "prepared.pt")
    train_numerical(cfg)
    return cfg


def test_interrupted_generation_resumes_without_repeating_completed_batches(
    trained_release, monkeypatch
):
    cfg = trained_release
    original = generation.generate_numerical
    calls = []

    def interrupted(batch_cfg):
        calls.append(batch_cfg.output_dir)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(batch_cfg)

    monkeypatch.setattr(generation, "generate_numerical", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generation.generate_release(cfg)
    root = Path(cfg.output_dir) / "generated"
    completed = root / "batches/00000000/generated/meta.parquet"
    modified = completed.stat().st_mtime_ns
    monkeypatch.setattr(generation, "generate_numerical", original)
    generation.generate_release(cfg)
    assert completed.stat().st_mtime_ns == modified
    meta = pd.read_parquet(root / "meta.parquet")
    core = pd.read_parquet(root / "core.parquet")
    assert len(meta) == cfg.generate.num_samples
    assert meta.stay_id.tolist() == list(range(cfg.generate.num_samples))
    assert meta.patient_uid.is_unique
    assert set(core.stay_id) == set(meta.stay_id)
    assert meta.death_icu.sum() == round(
        cfg.generate.num_samples * cfg.generate.positive_fraction
    )


def test_generation_rejects_changed_settings_on_resume(trained_release):
    cfg = trained_release
    generation.generate_release(cfg)
    cfg.generate.temperature = 0.5
    with pytest.raises(ValueError, match="settings"):
        generation.generate_release(cfg)


def test_generation_rejects_mixing_model_checkpoints(trained_release):
    cfg = trained_release
    generation.generate_release(cfg)
    path = Path(cfg.output_dir) / "model.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["state_dict"]["lm_head.weight"][0, 0] += 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="checkpoint"):
        generation.generate_release(cfg)
