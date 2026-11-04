"""Generate the center-conditioned companion release for full privacy auditing."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from syntheticicu.conditioned_release_numeric import generate_conditioned_release


@hydra.main(
    version_base="1.3", config_path="../config", config_name="numerical_matched"
)
def main(cfg: DictConfig):
    """Keep conditioned release artifacts in a separate repository directory."""
    repository = Path(hydra.utils.get_original_cwd()).resolve()
    if not Path(cfg.output_dir).resolve().is_relative_to(repository):
        raise ValueError("output_dir must stay inside the repository")
    print(generate_conditioned_release(cfg), flush=True)


if __name__ == "__main__":
    main()
