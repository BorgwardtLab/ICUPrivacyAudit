import glob
import logging
import os
import re
import zipfile
import hydra
from hydra import initialize, compose
import torch
from omegaconf import OmegaConf
import lightning as L
from syntheticicu.resolvers import register_resolvers
import sys

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

# Register arithmetic interpolation for the training schedule.
register_resolvers()

log = logging.getLogger(__name__)


def get_model_class(cfg):
    """Get the appropriate model class based on config."""
    model_target = getattr(
        cfg.model, "_target_", "syntheticicu.model_emb.get_llama_model"
    )

    if "model_gap" in model_target:
        from syntheticicu.model_gap import ICULlamaGap

        return ICULlamaGap
    else:
        from syntheticicu.model_emb import ICULlama

        return ICULlama


def find_resume_checkpoint(resume_dir, ckpt_name):
    """Latest complete resume checkpoint in resume_dir, or None."""
    pattern = re.compile(rf"^{re.escape(ckpt_name)}-resume-epoch(\d+)\.ckpt$")
    candidates = []
    for path in glob.glob(os.path.join(resume_dir, f"{ckpt_name}-resume-epoch*.ckpt")):
        match = pattern.match(os.path.basename(path))
        if match:
            candidates.append((int(match.group(1)), path))
    # A job killed while writing leaves a truncated file; skip it for an older one
    for _, path in sorted(candidates, reverse=True):
        if zipfile.is_zipfile(path):
            return path
        log.warning(f"Skipping incomplete resume checkpoint: {path}")
    return None


def fit_with_resume(trainer, model, datamodule, resume_dir, ckpt_name):
    """Fit, continuing from the latest resume checkpoint of an interrupted run, if any."""
    resume_ckpt = find_resume_checkpoint(resume_dir, ckpt_name)
    if resume_ckpt:
        log.info(f"Resuming training from {resume_ckpt}")
        print(f"[TRAIN] Resuming from {resume_ckpt}", flush=True)
    else:
        print("[TRAIN] No resume checkpoint, training from scratch", flush=True)

    # Our checkpoints hold the Hydra config (OmegaConf objects) as hyperparameters,
    # which torch>=2.6's default weights_only=True load rejects. They are our own files.
    trainer.fit(model, datamodule, ckpt_path=resume_ckpt, weights_only=False)


def build_wandb_tags(cfg) -> list[str]:
    """Build a list of WandB tags from the configuration."""
    tags = []

    # Model size
    model_name = getattr(cfg.model, "model_name", "unknown")
    tags.append(f"model:{model_name}")

    # Model type (base, CFG, Gap)
    model_target = getattr(cfg.model, "_target_", "")
    if "model_emb" in model_target:
        tags.append("type:CFG")
    elif "model_gap" in model_target:
        tags.append("type:Gap")
    else:
        tags.append("type:base")

    # Time embeddings
    time_emb = getattr(cfg.datamodule, "add_time_point_positional_embeddings", False)
    tags.append(f"time_emb:{time_emb}")

    # Demographics
    demographics = getattr(cfg.datamodule, "include_demographics", False)
    tags.append(f"demographics:{demographics}")

    # Token shuffling
    shuffle = getattr(cfg.datamodule, "shuffle_tokens_within_timepoint", False)
    tags.append(f"shuffle:{shuffle}")

    # Dataset
    dataset = getattr(cfg.datamodule, "dataset_name", "unknown")
    tags.append(f"dataset:{dataset}")

    # Label/conditioning (if using CFG)
    label = getattr(cfg.datamodule, "return_label", None)
    if label:
        tags.append(f"label:{label}")

    # Precision
    precision = getattr(cfg.trainer, "precision", "32")
    tags.append(f"precision:{precision}")

    return tags


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "train"
    overrides = sys.argv[2:]  # skip script name and config_name

    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
        print(OmegaConf.to_yaml(cfg), flush=True)

    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    L.seed_everything(cfg.seed, workers=True)

    # Instantiate the model and datamodule
    ModelClass = get_model_class(cfg)
    log.info(f"Using model class: {ModelClass.__name__}")
    model = ModelClass(cfg)

    # if torch.cuda.is_available():
    #     model = torch.compile(model)

    datamodule = model._datamodule

    logger = []
    if cfg.wandb:
        tags = build_wandb_tags(cfg)
        log.info(f"WandB tags: {tags}")
        wandb_logger = L.pytorch.loggers.WandbLogger(project="ICULlama", tags=tags)
        logger.append(wandb_logger)
    logger.append(L.pytorch.loggers.CSVLogger(cfg.logs.path, name="csv_logs"))

    # We log learning rates during training, and the best model ckpt
    # enable_version_counter=False: a restarted run overwrites the files of the
    # attempt it replaces instead of writing <name>-v1.ckpt next to them
    resume_dir = f"{cfg.logs.path}/resume"
    callbacks = [
        L.pytorch.callbacks.LearningRateMonitor(),
        L.pytorch.callbacks.ModelCheckpoint(
            monitor=f"val_loss",
            dirpath=cfg.logs.path,
            filename=cfg.ckpt_name,
            mode="min",
            enable_version_counter=False,
        ),
        L.pytorch.callbacks.ModelCheckpoint(
            every_n_epochs=10,
            dirpath=f"{cfg.logs.path}/probe",
            filename=f"{cfg.ckpt_name}-epoch{{epoch:03d}}",
            save_top_k=-1,  # save cheackpoint every 10 epochs, regardless of val_loss for probing purposes
            enable_version_counter=False,
        ),
        # Resume point: full trainer state at the end of every
        # resume_every_n_epochs epochs, only the latest one is kept. Saved at
        # epoch end so a resumed run sees the same data order.
        L.pytorch.callbacks.ModelCheckpoint(
            every_n_epochs=cfg.resume_every_n_epochs,
            save_on_train_epoch_end=True,
            dirpath=resume_dir,
            filename=f"{cfg.ckpt_name}-resume-epoch{{epoch:03d}}",
            auto_insert_metric_name=False,
            save_top_k=1,
            enable_version_counter=False,
        ),
    ]

    # Instantiate the Trainer
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    fit_with_resume(trainer, model, datamodule, resume_dir, cfg.ckpt_name)

    trainer.save_checkpoint(f"{cfg.logs.path}/{cfg.ckpt_name}-last.ckpt")

    trainer.validate(model, datamodule)


if __name__ == "__main__":
    main()
