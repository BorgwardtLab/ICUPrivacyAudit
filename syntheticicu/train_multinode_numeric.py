"""Distributed training with explicit node count and effective-batch validation."""

from functools import partial
import json
import os
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from syntheticicu.data_numeric import (
    collate_numerical,
    numerical_bins,
    repeated_training_indices,
)
from syntheticicu.workflow_numeric import (
    NumericalTrainingTask,
    _new_output,
    _resolved_model_config,
    data_signature,
)


def effective_batch(cfg):
    """Devices are per node; accumulation multiplies the global microbatch."""
    factors = [
        cfg.train.batch_size,
        cfg.train.devices,
        cfg.distributed.num_nodes,
        cfg.train.accumulate_grad_batches,
    ]
    if any(not isinstance(value, int) or value < 1 for value in factors):
        raise ValueError(
            "Positive integer batch, GPU, node and accumulation counts required"
        )
    batch = 1
    for value in factors:
        batch *= value
    if batch != cfg.distributed.expected_effective_batch:
        raise ValueError("Distributed effective batch differs from the reviewed target")
    return batch


def contract(cfg, signature, model_cfg):
    return dict(
        data_signature=signature,
        model_cfg=OmegaConf.to_container(model_cfg, resolve=True),
        optimizer={
            key: OmegaConf.to_container(cfg.train, resolve=True)[key]
            for key in [
                "learning_rate",
                "weight_decay",
                "betas",
                "warmup_steps",
                "scheduler_steps",
            ]
        },
        effective_batch=effective_batch(cfg),
        seed=int(cfg.seed),
        shuffle_within_hour=cfg.train.shuffle_within_hour,
        canaries=dict(
            count=int(cfg.train.n_canaries),
            duplication_factor=int(cfg.train.duplication_factor),
        ),
        training_entrypoint="matched_multinode_v1",
    )


def loaders(cfg, bundle):
    indices, canaries = repeated_training_indices(
        bundle, cfg.train.n_canaries, cfg.train.duplication_factor, cfg.seed
    )
    validation = [
        i for i, record in enumerate(bundle["records"]) if record["split"] == "val"
    ]
    options = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    training = DataLoader(
        Subset(bundle["records"], indices),
        shuffle=True,
        collate_fn=partial(collate_numerical, shuffle=cfg.train.shuffle_within_hour),
        **options,
    )
    validation = DataLoader(
        Subset(bundle["records"], validation), collate_fn=collate_numerical, **options
    )
    return training, validation, canaries


def trainer(cfg, root):
    checkpoint = ModelCheckpoint(
        dirpath=root / "checkpoints",
        filename="step-{step}",
        every_n_train_steps=cfg.train.checkpoint_every_n_steps,
        save_last=True,
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    return L.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        num_nodes=cfg.distributed.num_nodes,
        strategy=cfg.train.strategy,
        precision=cfg.train.precision,
        max_steps=cfg.train.max_steps,
        max_epochs=cfg.train.max_epochs,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        logger=CSVLogger(save_dir=str(root), name="csv_logs"),
        callbacks=[checkpoint],
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=cfg.train.limit_val_batches,
        val_check_interval=cfg.train.val_check_interval,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        default_root_dir=str(root),
        deterministic=cfg.train.deterministic,
    )


def train_multinode(cfg):
    """Start fresh after archival, or resume only this entry point's checkpoints."""
    root = Path(cfg.output_dir).resolve()
    target = root / "model.pt"
    _new_output(target, cfg.overwrite)
    if cfg.train.max_steps < 1:
        raise ValueError("Positive training step budget required")
    if (
        cfg.distributed.require_fresh_marker
        and not (root / "multinode_fresh_start.json").is_file()
    ):
        raise ValueError(
            "Archive the stopped two-GPU run before launching fresh training"
        )
    batch = effective_batch(cfg)
    expected_world = cfg.distributed.num_nodes * cfg.train.devices
    if "SLURM_NTASKS" in os.environ and cfg.train.accelerator == "gpu":
        if int(os.environ["SLURM_NTASKS"]) != expected_world:
            raise ValueError(
                "Slurm task count differs from the configured GPU world size"
            )
        if int(os.environ["SLURM_NNODES"]) != cfg.distributed.num_nodes:
            raise ValueError("Slurm node count differs from configuration")
    bundle = torch.load(root / "prepared.pt", map_location="cpu", weights_only=True)
    L.seed_everything(cfg.seed, workers=True)
    model_cfg = _resolved_model_config(cfg, bundle["schema"])
    if cfg.data.max_tokens + 2 > model_cfg.config.max_position_embeddings:
        raise ValueError(
            "Model context cannot accommodate conditioning and chunk length"
        )
    model = hydra.utils.instantiate(model_cfg, bins=numerical_bins(bundle["schema"]))
    signature = data_signature(bundle)
    task = NumericalTrainingTask(
        model,
        cfg.train.learning_rate,
        cfg.train.weight_decay,
        cfg.train.betas,
        cfg.train.warmup_steps,
        cfg.train.scheduler_steps,
        contract(cfg, signature, model_cfg),
    )
    training, validation, canaries = loaders(cfg, bundle)
    last = root / "checkpoints/last.ckpt"
    if last.exists() and not cfg.train.resume:
        raise FileExistsError("Existing checkpoint requires explicit resume")
    runner = trainer(cfg, root)
    print(
        f"Distributed training: {cfg.distributed.num_nodes} nodes, {expected_world} GPUs, effective batch {batch}",
        flush=True,
    )
    runner.fit(
        task,
        training,
        validation,
        ckpt_path=str(last) if cfg.train.resume and last.exists() else None,
    )
    if not runner.is_global_zero:
        return target
    report = dict(
        steps=int(runner.global_step),
        canaries=canaries,
        duplication_factor=int(cfg.train.duplication_factor),
        seed=int(cfg.seed),
        effective_batch=batch,
        num_nodes=int(cfg.distributed.num_nodes),
        world_size=int(expected_world),
        training_entrypoint="matched_multinode_v1",
        settings=OmegaConf.to_container(cfg.train, resolve=True),
    )
    return export_model(root, model, model_cfg, bundle["schema"], signature, report)


def export_model(root, model, model_cfg, schema, signature, report):
    """Publish the standard model artifact only after all optimizer updates finish."""
    target = root / "model.pt"
    payload = dict(
        state_dict={
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        model_cfg=OmegaConf.to_container(model_cfg, resolve=True),
        schema=schema,
        data_signature=signature,
        training=report,
    )
    temporary = target.with_suffix(".pt.partial")
    torch.save(payload, temporary)
    temporary.replace(target)
    (root / "training.json").write_text(
        json.dumps(dict(data_signature=signature, **report), indent=2) + "\n"
    )
    return target
