"""Attack B keeps first chunks and aligns complete synthetic rows with labels."""

import numpy as np
import pytest

from scripts.discrete_replication_audit import prepare_real_group, prepare_generated


def first_chunk(value="hr_raw_0"):
    return ["female_static_0", "age_static_1", "<GAP>", value, "<EOS>"]


def test_real_group_preserves_exact_ids_and_filters_continuations():
    group = prepare_real_group(
        [first_chunk(), ["hr_raw_1", "<EOS>"], first_chunk()],
        [100000001, 100000001, 100000002],
        [0, 0, 1],
        [0, 0, 0],
    )
    assert group["keys"] == ["0_100000001", "0_100000002"]
    assert group["classes"].tolist() == [0, 1]


def test_real_group_rejects_duplicate_initial_chunks():
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_real_group([first_chunk(), first_chunk()], [1, 1], [0, 0], [0, 0])


def test_complete_synthetic_filter_keeps_conditioning_aligned():
    source = [first_chunk(), first_chunk()[:-1], first_chunk("hr_raw_1")]
    group, mask = prepare_generated(source, np.array([0, 1, 1]), np.array([6, 6, 6]))
    assert mask.tolist() == [True, False, True]
    assert group["classes"].tolist() == [0, 1]
    assert group["centers"].tolist() == [6, 6]
    assert len(group["seqs"]) == 2


def test_synthetic_axes_must_match():
    with pytest.raises(ValueError, match="axes"):
        prepare_generated([first_chunk()], np.array([0, 1]), np.array([6]))


def test_attack_b_distances_follow_synthetic_to_real_direction():
    from syntheticicu.privacy_audit_ext import replication_distances

    training = dict(
        seqs=[["a"], ["b"]], classes=np.array([0, 1]), centers=np.array([0, 0])
    )
    holdout = dict(
        seqs=[["a", "a"], ["b", "b"]],
        classes=np.array([0, 1]),
        centers=np.array([0, 0]),
    )
    generated = dict(
        seqs=[["a"], ["b"]], classes=np.array([0, 1]), centers=np.array([6, 6])
    )
    result = replication_distances(generated, training, holdout, False, 1, 0)
    np.testing.assert_array_equal(result["generated_to_train"], [0.0, 0.0])
    np.testing.assert_array_equal(result["generated_to_holdout"], [0.5, 0.5])
    np.testing.assert_array_equal(result["holdout_to_train"], [0.5, 0.5])
