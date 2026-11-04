"""Run the separate numerical workflow using Hydra configuration."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from syntheticicu.workflow_numeric import (
    generate_numerical,
    prepare_numerical,
    train_numerical,
)


@hydra.main(
    version_base="1.3", config_path="../config", config_name="numerical_workflow"
)
def main(cfg: DictConfig):
    """Prepare, train, generate, or evaluate inside the repository."""
    root = Path(hydra.utils.get_original_cwd()).resolve()
    if not Path(cfg.output_dir).resolve().is_relative_to(root):
        raise ValueError("output_dir must stay inside the repository")
    actions = {
        "prepare": prepare_numerical,
        "train": train_numerical,
        "generate": generate_numerical,
    }
    print(actions[cfg.action](cfg), flush=True)


if __name__ == "__main__":
    main()
