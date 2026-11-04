"""Arithmetic interpolation for the paper's trusted Hydra configurations."""

from omegaconf import OmegaConf


def register_resolvers():
    """Register once so the scientific entry points can share one interpreter."""
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


register_resolvers()
