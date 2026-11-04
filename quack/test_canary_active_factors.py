"""The paper schedules only the three nonredundant duplication doses."""

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_paper_canary_grid_has_no_singleton_retraining():
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).parents[1] / "config")
    ):
        cfg = compose(config_name="canary_curves")
    assert list(cfg.factors) == [5, 20, 50]
    assert cfg.count == 500
