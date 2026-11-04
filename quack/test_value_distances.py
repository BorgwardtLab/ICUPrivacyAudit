"""Value distances retain numerical detail, missingness, clocks, and deletion pairing."""

import numpy as np
import pytest

from syntheticicu.value_distances import (
    deletion_indices,
    pair_distances,
    robust_scaling,
    scale_values,
    recovery_summary,
)


def trajectory(values, hours=None, demo=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return dict(
        values=values,
        hours=(
            np.arange(len(values), dtype=np.int64)
            if hours is None
            else np.asarray(hours, dtype=np.int64)
        ),
        demographics=np.array([0, 1] if demo is None else demo, dtype=np.int64),
    )


def test_identical_values_have_zero_distance_for_dtw_l1():
    record = trajectory([1.0, np.nan, 3.0])
    np.testing.assert_array_equal(pair_distances(record, record), np.zeros(1))


def test_sub_bin_differences_are_not_rounded_away():
    source, changed = trajectory([1.0, 2.0]), trajectory([1.0 + 1e-9, 2.0])
    assert np.all(pair_distances(source, changed) > 0)
    np.testing.assert_allclose(
        pair_distances(source, changed), pair_distances(changed, source)
    )


def test_missingness_and_elapsed_hours_affect_every_distance():
    source = trajectory([1.0, 2.0])
    assert np.all(pair_distances(source, trajectory([1.0, np.nan])) > 0)
    assert np.all(pair_distances(source, trajectory([1.0, 2.0], hours=[0, 3])) > 0)
    assert np.all(pair_distances(source, trajectory([1.0, 2.0], demo=[1, 1])) > 0)


def test_alignment_can_recover_a_deleted_hour():
    source = trajectory([1.0, 100.0, 2.0])
    deleted = trajectory([1.0, 2.0], hours=[0, 2])
    distances = pair_distances(source, deleted)
    assert np.isfinite(distances).all()
    assert (distances >= 0).all()
    assert distances.shape == (1,)


def test_scaling_preserves_extreme_values_and_is_invertible():
    values = np.array([[0.0, np.nan], [1.0, 2.0], [2.0, 3.0], [1e9, 4.0]])
    location, scale = robust_scaling(values)
    transformed = scale_values(values, location, scale)
    np.testing.assert_allclose(
        np.sinh(transformed) * scale + location, values, rtol=1e-12, equal_nan=True
    )
    assert transformed[-1, 0] > transformed[-2, 0]


def test_deletions_are_reproducible_nested_and_keep_original_hours():
    first = deletion_indices(100, 0.3, "0_123", 0)
    second = deletion_indices(100, 0.8, "0_123", 0)
    assert len(first) == 70 and len(second) == 20
    assert set(second) <= set(first)
    np.testing.assert_array_equal(first, deletion_indices(100, 0.3, "0_123", 0))
    np.testing.assert_array_equal(deletion_indices(100, 0, "0_123", 0), np.arange(100))


def test_recovery_handles_ties_conservatively_and_reports_wilson_interval():
    summary = recovery_summary(np.array([0.0, 0.5, 0.2]), np.array([1.0, 0.5, 0.1]))
    assert summary["rank1_rate"] == pytest.approx(1 / 3)
    assert summary["ties"] == 1
    assert summary["ci_low"] < summary["rank1_rate"] < summary["ci_high"]


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_invalid_deletion_fraction_fails(fraction):
    with pytest.raises(ValueError):
        deletion_indices(10, fraction, "0_1", 0)


def test_invalid_pair_fails_before_jit_work():
    with pytest.raises(ValueError):
        pair_distances(trajectory([1, 2], hours=[1, 1]), trajectory([1, 2]))
    with pytest.raises(ValueError):
        pair_distances(trajectory([1, np.inf]), trajectory([1, 2]))
    with pytest.raises(ValueError, match="cell budget"):
        pair_distances(trajectory([1, 2]), trajectory([1, 2]), max_pair_cells=3)


def test_packed_parallel_search_matches_direct_distances():
    from types import SimpleNamespace
    from syntheticicu.value_corpus import nearest, pack_stays, trajectory, pools

    schema = dict(
        features={"hr": {"edges": [0.0, 10.0], "tokens": ["hr0"]}},
        demographics={
            "sex": {
                "categories": [0.0, 1.0],
                "tokens": ["female0", "female1", "unknown"],
            }
        },
        vocabulary={"hr0": 4, "female0": 5, "female1": 6, "unknown": 7},
    )
    stays = [
        dict(
            values=np.array([[1.0], [2.0]]),
            hours=np.array([0, 2]),
            demographics={"sex": 0.0},
            centre="miiv",
            stay_id=index,
            death_icu=0,
            split="train",
        )
        for index in range(2)
    ]
    stays[1]["values"] = np.array([[2.0], [3.0]])
    packed = pack_stays(stays, schema, np.array([0.0]), np.array([1.0]))
    assert packed["offsets"].tolist() == [0, 3, 6]
    assert np.isnan(packed["values"][1]).all()  # Internal empty hour retained.
    options = SimpleNamespace(max_pair_cells=100, time_weight=0.1, gap_cost=1.0)
    distances, neighbors = nearest(
        packed, np.array([0]), packed, np.array([1]), options
    )
    np.testing.assert_allclose(
        distances[0], pair_distances(trajectory(packed, 0), trajectory(packed, 1))
    )
    assert np.all(neighbors == 1)
    candidate_pools = pools(packed, np.array([0, 1]), False, 140, 0)
    assert candidate_pools[(0, -1)].tolist() == [0, 1]


def test_small_corpus_pipeline_restarts_and_collects_all_attacks(tmp_path, monkeypatch):
    import joblib
    from omegaconf import OmegaConf
    from numba import set_num_threads
    from scripts import value_corpus_audit as pipeline
    from syntheticicu.value_corpus import pack_stays, pools, save_pack
    from syntheticicu.privacy_audit_ext import summarize_attack

    set_num_threads(1)
    schema = dict(
        features={"hr": {"edges": [0.0, 10.0], "tokens": ["hr0"]}},
        demographics={
            "sex": {"categories": [0.0, 1.0], "tokens": ["f0", "f1", "missing"]}
        },
        vocabulary={"hr0": 4, "f0": 5, "f1": 6, "missing": 7},
    )
    stays = [
        dict(
            values=np.array([[index + 1.0], [index + 1.5]]),
            hours=np.array([0, 1]),
            demographics={"sex": float(index % 2)},
            centre="miiv",
            stay_id=index,
            death_icu=index % 2,
            split=["train", "val", "test"][index // 2],
        )
        for index in range(6)
    ]
    generated = [
        dict(
            stays[index],
            values=stays[index]["values"] + 0.1,
            centre="__null__",
            stay_id=100 + index,
            split="synthetic",
        )
        for index in range(2)
    ]
    real = pack_stays(stays, schema, np.zeros(1), np.ones(1))
    synthetic = pack_stays(generated, schema, np.zeros(1), np.ones(1))
    save_pack(tmp_path / "real.joblib", real)
    save_pack(tmp_path / "null/synthetic.joblib", synthetic)
    directory = tmp_path / "null"
    selected_pools = {
        split: pools(real, np.flatnonzero(real["splits"] == split), False, 140, 0)
        for split in ["train", "val", "test"]
    }
    selected_pools["synthetic"] = pools(synthetic, np.arange(2), False, 140, 0)
    joblib.dump(selected_pools, directory / "pools.joblib")
    joblib.dump(dict(canaries=real["keys"][:2]), tmp_path / "parameters.joblib")
    cfg = OmegaConf.create(
        dict(
            generation_mode="null",
            shards=2,
            shard_index=0,
            query_batch_size=2,
            max_pair_cells=100,
            time_weight=0.1,
            gap_cost=1.0,
            score_epsilon=1e-12,
            n_canaries=2,
            drop_fractions=[value / 10 for value in range(9)],
            seed=0,
        )
    )
    for shard in range(2):
        cfg.shard_index = shard
        pipeline.membership(cfg, tmp_path)
        pipeline.replication(cfg, tmp_path)
        pipeline.floor(cfg, tmp_path)
    first = (directory / "membership/shard0/batch0.joblib").read_bytes()
    cfg.shard_index = 0
    pipeline.membership(cfg, tmp_path)
    assert (directory / "membership/shard0/batch0.joblib").read_bytes() == first
    monkeypatch.setattr(
        pipeline,
        "summarize_attack",
        lambda *args, **kwargs: summarize_attack(*args, n_boot=10, **kwargs),
    )
    pipeline.collect(cfg, tmp_path)
    import pandas as pd

    assert len(pd.read_csv(directory / "membership_metrics.csv")) == 1
    recovery = pd.read_csv(directory / "recovery.csv")
    assert len(recovery) == 9
    assert recovery.fraction.unique().tolist() == cfg.drop_fractions
    values = joblib.load(directory / "replication_results.joblib")["arrays"]
    assert values["generated_to_train"].shape == (2, 1)
    assert values["baseline_to_train"].shape == (2, 1)


def test_null_mode_is_a_literal_string_in_paper_config():
    from pathlib import Path
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).parents[1] / "config")
    ):
        cfg = compose(config_name="value_corpus_audit")
        assert cfg.generation_mode == "null"
