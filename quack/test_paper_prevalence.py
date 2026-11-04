"""Paper releases use the same outcome count in every representation."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import torch

from syntheticicu.base_model import ICULlamaBase


def test_paper_prevalence_overrides_the_training_cohort_fraction():
    model = SimpleNamespace(
        cfg=OmegaConf.create({"sampling": {"positive_fraction": 0.07}}),
        _datamodule=SimpleNamespace(
            train_icu_dataset=SimpleNamespace(
                indices=[0, 1],
                dataset=SimpleNamespace(y_full=torch.tensor([[1, 0], [2, 1]])),
            )
        ),
    )
    positive, negative, fraction = ICULlamaBase.compute_samples_by_prevalence(
        model, "death_icu", 12000
    )
    assert (positive, negative) == (840, 11160)
    assert fraction == 0.07
