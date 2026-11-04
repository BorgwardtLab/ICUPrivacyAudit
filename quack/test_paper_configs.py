"""The public configurations compose without private cluster path groups."""

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest


@pytest.mark.parametrize(
    "name",
    [
        "train",
        "generate",
        "numerical_matched",
        "clinical_pipeline",
        "clinical_native_replay",
        "canary_curves",
        "value_corpus_audit",
        "discrete_replication",
        "paper_example",
        "paper_inputs",
        "paper_results",
        "paper_quality",
        "paper_cohort",
        "paper_figures",
    ],
)
def test_public_configuration_composes(name):
    directory = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        configuration = compose(config_name=name)
    assert len(configuration) > 0
