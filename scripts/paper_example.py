"""Run the entirely artificial CPU example using the standard Hydra interface."""

import hydra
from omegaconf import DictConfig
from syntheticicu.paper_example import run_example


@hydra.main(version_base="1.3", config_path="../config", config_name="paper_example")
def main(cfg: DictConfig):
    print(run_example(cfg), flush=True)


if __name__ == "__main__":
    main()
