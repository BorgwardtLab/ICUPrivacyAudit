"""Check restart equivalence, atomic completion, and incompatible-cache rejection."""

import random

import numpy as np
import pytest
import torch

from syntheticicu.privacy_stage_cache import AuditStageCache


def test_restart_reuses_result_and_restores_rng(tmp_path):
    calls = []

    def calculate():
        calls.append(True)
        return random.random(), np.random.random(), torch.rand(3)

    cache = AuditStageCache(tmp_path, {"model": "a"})
    result = cache.run("scores", calculate)
    expected_next = calculate()
    restarted = AuditStageCache(tmp_path, {"model": "a"})
    recovered = restarted.run("scores", calculate)
    assert len(calls) == 2
    assert result[:2] == recovered[:2]
    assert torch.equal(result[2], recovered[2])
    actual_next = calculate()
    assert expected_next[:2] == actual_next[:2]
    assert torch.equal(expected_next[2], actual_next[2])


def test_failed_stage_is_not_complete_and_changed_contract_rejected(tmp_path):
    cache = AuditStageCache(tmp_path, {"model": "a"})

    def fail():
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        cache.run("scores", fail)
    assert not (tmp_path / "scores.joblib").exists()
    assert cache.run("scores", lambda: 42) == 42
    with pytest.raises(ValueError, match="contract"):
        AuditStageCache(tmp_path, {"model": "b"})


def test_uncached_path_returns_same_value_and_propagates_errors():
    assert AuditStageCache.run(None, "scores", lambda value: value + 1, 2) == 3
    with pytest.raises(ZeroDivisionError):
        AuditStageCache.run(None, "scores", lambda: 1 / 0)
