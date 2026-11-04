"""Generate the full numerical release with resumable batch outputs."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from syntheticicu.generation_numeric import generate_release


@hydra.main(
    version_base="1.3", config_path="../config", config_name="numerical_matched"
)
def main(cfg: DictConfig):
    """Generate within the repository using the completed full-model checkpoint."""
    root = Path(hydra.utils.get_original_cwd()).resolve()
    if not Path(cfg.output_dir).resolve().is_relative_to(root):
        raise ValueError("output_dir must stay inside the repository")
    print(generate_release(cfg), flush=True)


if __name__ == "__main__":
    main()
