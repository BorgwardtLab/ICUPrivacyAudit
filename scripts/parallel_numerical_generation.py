"""Prepare, dispatch, and merge unchanged generation batches in Slurm arrays."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from syntheticicu.conditioned_release_numeric import (
    prepare_conditioned_release,
    generate_conditioned_center,
    merge_conditioned_release,
)
from syntheticicu.generation_numeric import (
    prepare_release,
    generate_release,
    merge_release,
)


def link_input(source, target):
    """Preserve existing artifacts and reject links to different input data."""
    if target.exists():
        if target.resolve() != source.resolve():
            raise ValueError(f"Input link points elsewhere: {target}")
    else:
        target.symlink_to(source.resolve())


@hydra.main(
    version_base="1.3", config_path="../config", config_name="numerical_prevalence07"
)
def main(cfg: DictConfig):
    """Run one bounded phase, with final assembly owned by a separate CPU job."""
    root = Path(cfg.output_dir).resolve()
    repository = Path(hydra.utils.get_original_cwd()).resolve()
    if not root.is_relative_to(repository):
        raise ValueError("All generation artifacts must stay inside the repository")
    if cfg.action == "prepare_parallel":
        prepare_release(cfg)
        prepare_conditioned_release(cfg)
        for name in ["model.pt", "prepared.pt"]:
            link_input(root / name, root / "conditioned" / name)
    elif cfg.action == "generate_null_shard":
        generate_release(
            cfg, cfg.parallel.shard_index, cfg.parallel.shard_count, merge=False
        )
    elif cfg.action == "generate_center":
        generate_conditioned_center(cfg, cfg.parallel.shard_index)
    elif cfg.action == "merge_null":
        merge_release(cfg)
    elif cfg.action == "merge_conditioned":
        merge_conditioned_release(cfg)
    else:
        raise ValueError("Unknown parallel generation phase")


if __name__ == "__main__":
    main()
