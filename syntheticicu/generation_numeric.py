"""Resumable batch generation of a numerical synthetic corpus."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import torch

from syntheticicu.workflow_numeric import generate_numerical


def _checkpoint_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete(directory):
    return all(
        (directory / name).is_file()
        for name in ["core.parquet", "meta.parquet", "tokens.pt", "generation.json"]
    )


def prepare_release(cfg):
    """Bind the release plan before sequential or concurrent batch generation."""
    root = Path(cfg.output_dir).resolve()
    checkpoint = root / "model.pt"
    destination = root / "generated"
    count, batch_size = int(cfg.generate.num_samples), int(cfg.generate.batch_size)
    fraction = float(cfg.generate.positive_fraction)
    if count < 1 or batch_size < 1 or not 0 <= fraction <= 1:
        raise ValueError(
            "Positive sample/batch counts and valid class fraction required"
        )
    device = torch.device(cfg.generate.device)
    plan = {
        "checkpoint_sha256": _checkpoint_hash(checkpoint),
        "seed": int(cfg.seed),
        "settings": OmegaConf.to_container(cfg.generate, resolve=True),
        "precision": "bf16-mixed" if device.type == "cuda" else "float32",
    }
    plan_path = destination / "generation-plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError(
                "Existing release checkpoint or generation settings differ"
            )
    else:
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                "Existing generation output has no recovery plan; use a new output directory"
            )
        destination.mkdir(parents=True, exist_ok=True)
        temporary = plan_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(plan, indent=2) + "\n")
        temporary.replace(plan_path)
    return plan


def _release_batches(cfg):
    count, batch_size = int(cfg.generate.num_samples), int(cfg.generate.batch_size)
    fraction = float(cfg.generate.positive_fraction)
    positive = int(round(count * fraction))
    labels = np.concatenate(
        [np.zeros(count - positive, dtype=int), np.ones(positive, dtype=int)]
    )
    labels = np.random.default_rng(cfg.seed).permutation(labels)
    destination = Path(cfg.output_dir).resolve() / "generated"
    batches = [
        (start, destination / "batches" / f"{start:08d}" / "generated")
        for start in range(0, count, batch_size)
    ]
    return labels, batches


def _generate_release_batch(cfg, start, batch_output, labels):
    count, batch_size = int(cfg.generate.num_samples), int(cfg.generate.batch_size)
    size, batch_seed = min(batch_size, count - start), int(cfg.seed) + start
    if _complete(batch_output):
        manifest = json.loads((batch_output / "generation.json").read_text())
        if manifest["num_samples"] != size or manifest["seed"] != batch_seed:
            raise ValueError(
                "Completed batch manifest does not match the recovery plan"
            )
        print(f"Reusing samples {start}:{start + size}", flush=True)
        return
    batch_root = batch_output.parent
    batch_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(cfg.output_dir).resolve() / "model.pt"
    model_link = batch_root / "model.pt"
    if not model_link.exists():
        model_link.symlink_to(checkpoint)
    elif model_link.resolve() != checkpoint.resolve():
        raise ValueError("Batch model link points to another checkpoint")
    batch_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    batch_cfg.output_dir, batch_cfg.seed = str(batch_root), batch_seed
    batch_cfg.overwrite = True
    batch_cfg.generate.num_samples = size
    batch_cfg.generate.positive_fraction = (
        float(labels[start : start + size].sum()) / size
    )
    print(f"Generating samples {start}:{start + size} of {count}", flush=True)
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.device(cfg.generate.device).type == "cuda"
        else nullcontext()
    )
    with context:
        generate_numerical(batch_cfg)


def generate_release(cfg, shard_index=0, shard_count=1, merge=True):
    """Generate unchanged batches; parallel workers own disjoint batch indexes."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid generation shard index/count")
    destination = Path(cfg.output_dir).resolve() / "generated"
    if shard_count > 1 and not (destination / "generation-plan.json").exists():
        raise FileNotFoundError("Call prepare_release before starting parallel shards")
    prepare_release(cfg)
    if _complete(destination):
        print(f"Release already complete: {destination}", flush=True)
        return destination
    if torch.device(cfg.generate.device).type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA release generation requires BF16 support")
    labels, batches = _release_batches(cfg)
    for index, (start, batch_output) in enumerate(batches):
        if index % shard_count == shard_index:
            _generate_release_batch(cfg, start, batch_output, labels)
    return merge_release(cfg) if merge else destination


def merge_release(cfg):
    """Assemble complete batches in the original order, with unchanged IDs."""
    plan = prepare_release(cfg)
    destination = Path(cfg.output_dir).resolve() / "generated"
    if _complete(destination):
        return destination
    count, batch_size = int(cfg.generate.num_samples), int(cfg.generate.batch_size)
    fraction = float(cfg.generate.positive_fraction)
    _, batches = _release_batches(cfg)
    core_frames, meta_frames, tokens, manifests = [], [], [], []
    for start, batch_output in batches:
        if not _complete(batch_output):
            raise FileNotFoundError(f"Generation batch is incomplete: {batch_output}")
        core = pd.read_parquet(batch_output / "core.parquet")
        meta = pd.read_parquet(batch_output / "meta.parquet")
        core["stay_id"] += start
        meta["stay_id"] += start
        meta["patient_uid"] += start
        core_frames.append(core)
        meta_frames.append(meta)
        tokens.extend(
            torch.load(
                batch_output / "tokens.pt", map_location="cpu", weights_only=True
            )
        )
        manifests.append(json.loads((batch_output / "generation.json").read_text()))
    core = pd.concat(core_frames, ignore_index=True)
    meta = pd.concat(meta_frames, ignore_index=True)
    if (
        len(meta) != count
        or not meta.stay_id.is_unique
        or not meta.patient_uid.is_unique
    ):
        raise ValueError("Release count or synthetic identifiers are inconsistent")
    if set(core.stay_id) != set(meta.stay_id):
        raise ValueError("Synthetic core and meta identifiers do not align")
    signatures = {manifest["data_signature"] for manifest in manifests}
    if len(signatures) != 1:
        raise ValueError("Batches come from different prepared populations")
    for name, frame in [("core", core), ("meta", meta)]:
        temporary = destination / f"{name}.parquet.partial"
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination / f"{name}.parquet")
    temporary = destination / "tokens.pt.partial"
    torch.save(tokens, temporary)
    temporary.replace(destination / "tokens.pt")
    _write_release_manifest(cfg, destination, plan, signatures.pop(), meta)
    print(f"Completed numerical release: {count} samples at {destination}", flush=True)
    return destination


def _write_release_manifest(cfg, destination, plan, signature, meta):
    """Write the same completion metadata after all payloads are durable."""
    manifest = {
        "data_signature": signature,
        "model_sha256": plan["checkpoint_sha256"],
        "num_samples": int(cfg.generate.num_samples),
        "center": cfg.generate.center,
        "positive_fraction": float(cfg.generate.positive_fraction),
        "seed": int(cfg.seed),
        "terminated": int(meta.terminated.sum()),
        "batch_size": int(cfg.generate.batch_size),
        "precision": plan["precision"],
        "settings": plan["settings"],
    }
    temporary = destination / "generation.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "generation.json")
