#!/usr/bin/env python
"""
Injected-memorisation testbed at the scale of the audited model.

Same architecture, same six centres and stays, same training configuration as
the audited checkpoint: the configuration is read from that run's own
hparams.yaml, not from train.yaml. The only differences are the injected
canaries, the output paths, wandb off, and the world-size bookkeeping
(devices / accumulate_grad_batches / val_check_interval are rescaled so the
effective batch and the validation cadence in optimizer steps stay those of the
audited run). `train --data-path` swaps the dataset (e.g. another binning of the
same stays); the path is kept in the manifest so `score` loads the same data.

Two stages, run as separate jobs:

    srun python scripts/run_testbed_full.py train --out OUT --dup 20 --n-canaries 500
    python scripts/run_testbed_full.py score --out OUT

train   builds the model as scripts/train.py does, injects the canaries before
        fit, trains under DDP to the audited budget (max_epochs, early stop on
        nothing) and keeps the best-val_loss checkpoint, like the audited
        llama-xs.ckpt. Resumes from OUT/last.ckpt if present.
score   loads that best checkpoint the way the main audit loads the deployed
        model (fresh datamodule, so every training row is scored once), runs
        evaluate_testbed, and reports cal_center on normal members vs holdout
        restricted to the deployed audit's stratified cohort, the number
        comparable with the deployed model's cal_center.

Run each stage in a GPU or CPU allocation as appropriate.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import torch
import lightning as L
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# importing it registers the OmegaConf resolvers, as every entry point does
from run_privacy_audit_ext import (  # noqa: E402
    banner,
    get_model_class,
    testbed_headline,
    print_testbed_headline,
)
from syntheticicu.privacy_audit import get_stratified_patient_keys  # noqa: E402
from syntheticicu.privacy_audit_ext import (  # noqa: E402
    inject_canaries,
    evaluate_testbed,
    summarize_attack,
)
from syntheticicu.data_classes import CENTRE_MAP  # noqa: E402

AUDITED_HPARAMS = "inputs/discrete_hparams.yaml"

# Reference training budget: 44,900 optimizer steps over 100 epochs.
AUDITED_STEPS_PER_EPOCH = 449
# deployed model, main audit, stratified cohort (n_per_center=3000, seed 0)
N_PER_CENTER = 3000
STATS = ["raw", "cal_center", "cal_class", "cal_both"]


def rank0():
    return int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0))) == 0


def say(*a):
    if rank0():
        print(*a, flush=True)


def load_cfg(hparams, out, world_size, data_path=None):
    """
    The audited configuration, with only what has to change. Returns the config
    and the reference values it was rescaled from.
    """
    cfg = OmegaConf.load(hparams).cfg
    OmegaConf.set_struct(cfg, False)
    if data_path is not None:
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)
        # hparams.yaml is saved resolved, so the path is set in both places
        cfg.datamodule.data_path = data_path
        if "paths" in cfg and "data" in cfg.paths and "icuall" in cfg.paths.data:
            cfg.paths.data.icuall = data_path
    t = cfg.trainer
    ref = {
        "devices": int(t.devices),
        "accumulate_grad_batches": int(t.accumulate_grad_batches),
        "val_check_interval": int(t.val_check_interval),
        "strategy": str(t.strategy),
        "batch_size": int(cfg.datamodule.batch_size),
    }
    effective_batch = (
        ref["batch_size"] * ref["devices"] * ref["accumulate_grad_batches"]
    )
    val_every_steps = ref["val_check_interval"] / ref["accumulate_grad_batches"]

    per_step = ref["batch_size"] * world_size
    if effective_batch % per_step:
        raise ValueError(
            f"{world_size} GPUs cannot reproduce the effective batch of "
            f"{effective_batch} with batch_size {ref['batch_size']}"
        )
    accumulate = effective_batch // per_step
    if val_every_steps != int(val_every_steps):
        raise ValueError(
            "audited val_check_interval is not a whole number of optimizer steps"
        )

    t.devices = world_size
    t.accumulate_grad_batches = accumulate
    # val_check_interval counts batches per rank: keep validation every
    # val_every_steps optimizer steps, as in the audited run
    t.val_check_interval = int(val_every_steps) * accumulate
    t.strategy = "ddp" if world_size > 1 else "auto"
    t.default_root_dir = out
    cfg.logs.path = out
    cfg.wandb = False
    ref.update(effective_batch=effective_batch, val_every_steps=int(val_every_steps))
    return cfg, ref


def train(args):
    import hydra

    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    cfg, ref = load_cfg(args.hparams, args.out, world, data_path=args.data_path)
    say(f"Audited config: {args.hparams}")
    say(f"  data: {cfg.datamodule.data_path}")
    say(
        f"  effective batch {ref['effective_batch']} = batch {ref['batch_size']} x "
        f"{ref['devices']} GPUs x accumulate {ref['accumulate_grad_batches']}"
    )
    say(
        f"  this run: {world} GPUs x accumulate {cfg.trainer.accumulate_grad_batches}, "
        f"val_check_interval {cfg.trainer.val_check_interval} "
        f"(= every {ref['val_every_steps']} optimizer steps, as audited)"
    )
    say(
        f"  budget: max_epochs {cfg.trainer.max_epochs}, max_steps {cfg.trainer.max_steps}"
    )

    # as in train.py
    L.seed_everything(cfg.seed, workers=True)
    ModelClass = get_model_class(cfg)
    model = ModelClass(cfg)
    dm = model._datamodule

    n_rows = len(dm.train_indices)
    base_steps = n_rows / ref["effective_batch"]
    say(
        f"  train rows before canaries: {n_rows} -> ~{base_steps:.0f} steps/epoch "
        f"(audited run: {AUDITED_STEPS_PER_EPOCH})"
    )
    if abs(base_steps - AUDITED_STEPS_PER_EPOCH) / AUDITED_STEPS_PER_EPOCH > 0.02:
        raise ValueError(
            "training split differs from the audited run's; check data_path / filters"
        )

    canary_keys = inject_canaries(dm, args.n_canaries, args.dup, cfg.seed)
    extra = len(dm.train_indices) - n_rows
    steps = len(dm.train_indices) / ref["effective_batch"]
    say(
        f"  canary rows: +{extra} ({extra / n_rows:.1%} of the training split) -> "
        f"~{steps:.0f} steps/epoch, ~{steps * cfg.trainer.max_epochs:.0f} steps in total"
    )
    if steps * cfg.trainer.max_epochs > cfg.trainer.max_steps:
        raise ValueError(
            "max_steps would bind before max_epochs: not the audited budget"
        )

    # silently ignored settings would invalidate the comparison
    emb = model.label_embedder
    loaded = sorted(
        {str(s).lower() for s in np.asarray(dm.base_icu_dataset.stay_sources)}
    )
    say(
        f"  effective center_dropout_prob {emb.center_dropout_prob}, label_dropout {emb.dropout_prob}"
    )
    say(f"  centres loaded: {loaded}")
    if loaded != sorted(CENTRE_MAP):
        raise ValueError(f"expected all six centres, loaded {loaded}")
    if (
        abs(emb.center_dropout_prob - cfg.model.center_dropout_prob) > 1e-9
        or abs(emb.dropout_prob - cfg.model.label_dropout) > 1e-9
    ):
        raise ValueError("dropout settings differ from the audited configuration")

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    canary_path = os.path.join(args.out, "canaries.json")
    last = os.path.join(args.out, "last.ckpt")
    resume = last if os.path.isfile(last) else None
    if resume and os.path.isfile(canary_path):
        # injection is deterministic; a resume must see the same canaries
        if set(json.load(open(canary_path))) != set(canary_keys):
            raise ValueError(
                "canaries differ from the ones the interrupted run trained on"
            )

    logger = [L.pytorch.loggers.CSVLogger(args.out, name="csv_logs")]
    ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.out,
        filename=cfg.model.model_name,
        mode="min",
        save_last=True,
    )
    callbacks = [L.pytorch.callbacks.LearningRateMonitor(), ckpt_cb]
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    eb = (
        cfg.datamodule.batch_size * trainer.world_size * trainer.accumulate_grad_batches
    )
    say(
        f"  trainer: world_size {trainer.world_size}, accumulate {trainer.accumulate_grad_batches} "
        f"-> effective batch {eb}"
    )
    if eb != ref["effective_batch"]:
        raise ValueError(f"effective batch {eb} != audited {ref['effective_batch']}")

    manifest = {
        "dup_factor": args.dup,
        "n_canaries": args.n_canaries,
        "seed": int(cfg.seed),
        "hparams": args.hparams,
        "data_path": str(cfg.datamodule.data_path),
        "world_size": trainer.world_size,
        "accumulate_grad_batches": trainer.accumulate_grad_batches,
        "val_check_interval": int(cfg.trainer.val_check_interval),
        "effective_batch": eb,
        "train_rows": n_rows,
        "canary_rows": extra,
        "resumed_from": resume,
        "finished": False,
    }
    if trainer.is_global_zero:
        json.dump(sorted(canary_keys), open(canary_path, "w"))
        json.dump(manifest, open(manifest_path, "w"), indent=1)

    trainer.fit(model, dm, ckpt_path=resume)
    # as in train.py
    trainer.save_checkpoint(os.path.join(args.out, f"{cfg.model.model_name}-last.ckpt"))

    if trainer.is_global_zero:
        best = (
            float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None
            else None
        )
        manifest.update(
            finished=True,
            best_model_path=ckpt_cb.best_model_path,
            best_val_loss=best,
            global_step=int(trainer.global_step),
            epochs=int(trainer.current_epoch),
        )
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        print(f"\nBest checkpoint: {ckpt_cb.best_model_path} (val_loss {best})")
        print(f"Stopped at step {trainer.global_step}, epoch {trainer.current_epoch}")


def score(args):
    manifest = json.load(open(os.path.join(args.out, "manifest.json")))
    if not manifest.get("finished"):
        raise ValueError("training has not finished (manifest.json: finished=false)")
    canary_keys = set(json.load(open(os.path.join(args.out, "canaries.json"))))
    # older manifests have no data_path: they trained on the audited one
    cfg, _ = load_cfg(
        manifest["hparams"], args.out, world_size=1, data_path=manifest.get("data_path")
    )
    audited_data = OmegaConf.load(manifest["hparams"]).cfg.datamodule.data_path
    same_data = os.path.realpath(cfg.datamodule.data_path) == os.path.realpath(
        audited_data
    )
    print(f"Data: {cfg.datamodule.data_path}", flush=True)
    dup = manifest["dup_factor"]

    # as the main audit loads the deployed model: a fresh datamodule, so the
    # canaries are back to one copy and every training row is scored once
    print(
        f"Loading {manifest['best_model_path']} (best val_loss {manifest['best_val_loss']}, "
        f"training stopped at step {manifest['global_step']})",
        flush=True,
    )
    ModelClass = get_model_class(cfg)
    model = ModelClass.load_from_checkpoint(
        manifest["best_model_path"], new_cfg=cfg, weights_only=False
    )
    if torch.cuda.is_available():
        model.to("cuda")
    # fp32 by default, as the deployed audit was scored: cal_center is a
    # difference of two close log-probs, so bf16 would not be comparable
    if args.precision == "bf16":
        model.to(torch.bfloat16)
    model.eval()
    dm = model._datamodule

    df, audit = evaluate_testbed(
        model,
        canary_keys,
        dup_factor=dup,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    headline = testbed_headline(df, audit, canary_keys, manifest["n_canaries"])

    banner("cal_center on the deployed audit's cohort")
    # same procedure and seed as run_audit's shared stratified cohort, on the
    # same split, so these are the patients behind the deployed number
    fixed = {
        "train": get_stratified_patient_keys(
            model, dm.train_icu_dataset, N_PER_CENTER, seed=0
        ),
        "holdout": get_stratified_patient_keys(
            model, dm.test_icu_dataset, N_PER_CENTER, seed=0
        ),
    }
    in_cohort = len(fixed["train"] & canary_keys)
    stratified = {}
    for stat in STATS:
        # the deployed model has no canaries, so they are left out of the members
        tr = [
            v
            for k, v in audit["train_scores"][stat].items()
            if k in fixed["train"] and k not in canary_keys
        ]
        ho = [
            v for k, v in audit["holdout_scores"][stat].items() if k in fixed["holdout"]
        ]
        stratified[stat] = summarize_attack(
            tr, ho, name=f"cohort normal_vs_holdout/{stat}"
        )
    print(
        f"  cohort: {len(fixed['train'])} train patients ({in_cohort} canaries left out), "
        f"{len(fixed['holdout'])} holdout"
    )
    s = stratified["cal_center"]
    print(f"\n  cal_center, normal members vs holdout, deployed cohort:")
    print(
        f"    testbed (canaries x{dup}) : {s['auc']:.4f} [{s['auc_ci'][0]:.4f}, {s['auc_ci'][1]:.4f}]"
    )
    r = stratified["raw"]
    print(
        f"  raw on the same cohort    : {r['auc']:.4f} [{r['auc_ci'][0]:.4f}, {r['auc_ci'][1]:.4f}]"
    )

    print_testbed_headline(
        {f"full_dup{dup}": headline},
        title=f"full-scale testbed headline, canaries x{dup}",
    )

    out = os.path.join(args.out, "testbed_full_results.joblib")
    joblib.dump(
        {
            "table": df,
            "summaries": audit["summaries"],
            "headline": headline,
            "stratified_normal_vs_holdout": stratified,
            "cohort_sizes": {k: len(v) for k, v in fixed.items()},
            "canaries_in_cohort": in_cohort,
            "manifest": manifest,
            "precision": args.precision,
            "canary_keys": sorted(canary_keys),
            # per-patient scores, for per-centre breakdowns later
            "train_scores": audit["train_scores"],
            "holdout_scores": audit["holdout_scores"],
        },
        out,
    )
    print(f"\nResults saved to: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--dup", type=int, required=True)
    t.add_argument("--n-canaries", type=int, default=500)
    t.add_argument("--hparams", default=AUDITED_HPARAMS)
    t.add_argument(
        "--data-path",
        default=None,
        help="dataset to train on instead of the audited one",
    )
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
    s.add_argument("--batch-size", type=int, default=128)
    s.add_argument("--num-workers", type=int, default=12)
    args = p.parse_args()
    train(args) if args.cmd == "train" else score(args)


if __name__ == "__main__":
    main()
