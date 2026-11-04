"""Check that corpus histograms count clinical hours and measurements consistently."""

import numpy as np

from scripts.summarize_tokenization_corpora import sequence_hour_counts


def test_demographics_special_tokens_and_trailing_gap_do_not_add_measurements():
    counts, complete = sequence_hour_counts(
        [
            "<SOS>",
            "age_static_2",
            "female_static_0",
            "<GAP>",
            "hr_2",
            "temp_3",
            "<GAP>",
            "hr_1",
            "<EOS>",
            "<PAD>",
        ]
    )
    np.testing.assert_array_equal(counts, [2, 1])
    assert complete
    trailing, complete = sequence_hour_counts(
        [
            "age_static_2",
            "<GAP>",
            "hr_2",
            "<GAP>",
            "<EOS>",
        ]
    )
    np.testing.assert_array_equal(trailing, [1])
    assert complete


def test_empty_hours_and_truncated_last_hour_are_retained():
    counts, complete = sequence_hour_counts(
        [
            "age_static_bin_2",
            "<GAP>",
            "hr_raw_bin_1",
            "<GAP>",
            "<GAP>",
            "hr_raw_bin_2",
        ]
    )
    np.testing.assert_array_equal(counts, [1, 0, 1])
    assert not complete
