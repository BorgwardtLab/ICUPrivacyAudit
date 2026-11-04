"""Check complete real-hour accounting without exporting patient-level records."""

import numpy as np
import pytest

from scripts.prepare_clinical_figure import real_hour_histogram


def test_real_reference_counts_empty_hours_and_excludes_post_exit_rows():
    values = np.asarray([[0, 5], [5, 3], [1, 2]])
    times = np.asarray([1, 2, 7])
    counts = real_hour_histogram(values, times, 5.5, 5)
    assert counts.sum() == 6
    assert counts[0] == 4
    assert counts[1] == 2
    assert counts[2] == 0


def test_real_reference_rejects_duplicate_hours():
    with pytest.raises(ValueError, match="increasing"):
        real_hour_histogram(np.asarray([[0], [1]]), np.asarray([1, 1]), 5.0, 5)
