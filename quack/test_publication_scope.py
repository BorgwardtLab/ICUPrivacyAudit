"""The paper interface exposes only its reported generator and utility tasks."""

import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from syntheticicu.demo_utils import DEMO_MODEL_REGISTRY
from syntheticicu.tasks_numeric import active_tasks, classifier_arguments


def test_demographic_generator_registry_contains_only_the_reported_model():
    assert set(DEMO_MODEL_REGISTRY) == {"tvae"}
    assert DEMO_MODEL_REGISTRY["tvae"]["_target_"] == "sdv.single_table.TVAESynthesizer"


def test_paper_configuration_has_only_its_training_components():
    directory = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(directory)):
        cfg = compose(config_name="train")
    assert set(cfg) == {
        "seed",
        "wandb",
        "trainer",
        "train",
        "model",
        "ckpt_name",
        "resume_every_n_epochs",
        "logs",
        "datamodule",
        "paths",
    }


def test_utility_matrix_has_fixed_clinical_windows_and_single_model_fit(tmp_path):
    tasks = active_tasks()
    assert [task["mix"] for task in tasks] == [index / 10 for index in range(11)]
    assert all(task["kind"] == "classifier" for task in tasks)
    args = classifier_arguments(tasks[0], tmp_path, 42, 1)
    assert args.lookback == 24
    assert args.n_augmentations == 1


def test_distributable_fixture_declares_its_artificial_origin():
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((root / "quack/fixtures/dtw_l1_golden.json").read_text())
    assert fixture["artificial"] is True
    assert fixture["seed"] == 18
