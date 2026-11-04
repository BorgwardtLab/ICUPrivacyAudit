"""
Resuming an interrupted training run from scripts/train.py resume checkpoints.
"""

import importlib.util
from pathlib import Path

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
CKPT_NAME = "tiny"
STEPS_PER_EPOCH = 4


@pytest.fixture(scope="module")
def train_script():
    spec = importlib.util.spec_from_file_location("train_script", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyModule(L.LightningModule):
    """Stores an OmegaConf config as hyperparameters, like ICULlamaBase does."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), betas=tuple(self.hparams.cfg.betas))


class TinyDataModule(L.LightningDataModule):
    def train_dataloader(self):
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(STEPS_PER_EPOCH * 2, 2, generator=generator)
        return DataLoader(TensorDataset(x, x.sum(dim=1, keepdim=True)), batch_size=2)


def make_trainer(tmp_path, max_epochs):
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def make_model():
    return TinyModule(OmegaConf.create({"betas": [0.9, 0.95]}))


def test_resumes_from_checkpoint_with_omegaconf_hparams(tmp_path, train_script):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()

    first = make_trainer(tmp_path, max_epochs=1)
    first.fit(make_model(), TinyDataModule())
    first.save_checkpoint(resume_dir / f"{CKPT_NAME}-resume-epoch000.ckpt")

    second = make_trainer(tmp_path, max_epochs=2)
    train_script.fit_with_resume(
        second, make_model(), TinyDataModule(), str(resume_dir), CKPT_NAME
    )

    assert second.current_epoch == 2
    assert second.global_step == 2 * STEPS_PER_EPOCH


def test_trains_from_scratch_without_resume_checkpoint(tmp_path, train_script):
    trainer = make_trainer(tmp_path, max_epochs=1)
    train_script.fit_with_resume(
        trainer, make_model(), TinyDataModule(), str(tmp_path / "resume"), CKPT_NAME
    )

    assert trainer.global_step == STEPS_PER_EPOCH
