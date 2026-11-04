"""Train the matched continuous model with explicit distributed resources."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from syntheticicu.train_multinode_numeric import train_multinode


@hydra.main(
    version_base="1.3", config_path="../config", config_name="numerical_matched"
)
def main(cfg: DictConfig):
    if (
        not Path(cfg.output_dir)
        .resolve()
        .is_relative_to(Path(hydra.utils.get_original_cwd()).resolve())
    ):
        raise ValueError("Distributed outputs must remain inside the repository")
    print(train_multinode(cfg), flush=True)


if __name__ == "__main__":
    main()
