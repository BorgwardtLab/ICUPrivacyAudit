"""The distributable example contains artificial observations and exact IDs only."""

from pathlib import Path
import json
import pandas as pd
from syntheticicu.paper_example import make_data


def test_artificial_fixture_is_deterministic_and_has_clinical_endpoints(tmp_path):
    specification = json.loads(Path("examples/toy_spec.json").read_text())
    first = make_data(tmp_path / "a", specification)
    second = make_data(tmp_path / "b", specification)
    pd.testing.assert_frame_equal(
        pd.read_parquet(first / "core.parquet"),
        pd.read_parquet(second / "core.parquet"),
    )
    metadata = pd.read_parquet(first / "meta.parquet")
    assert len(metadata) == 40
    assert metadata.stay_id.is_unique
    assert set(metadata.death_icu) == {0, 1}
    assert (metadata.los_icu * 24 == 48).all()


def test_observation_signature_ignores_bin_labels_but_preserves_missingness(tmp_path):
    import joblib
    from scripts.summarize_paper_quality import observation_signature

    specification = json.loads(
        (Path(__file__).parents[1] / "examples/toy_spec.json").read_text()
    )
    source = make_data(tmp_path / "source", specification) / "discrete.pkl"
    original = observation_signature(source)
    dataset = joblib.load(source)
    for values in dataset.X_value_ids:
        observed = values < dataset.num_bins
        values[observed] = (values[observed] + 1) % dataset.num_bins
    joblib.dump(dataset, source)
    assert observation_signature(source) == original
    dataset.X_value_ids[0][1, 0] = dataset.num_bins
    joblib.dump(dataset, source)
    assert observation_signature(source) != original
