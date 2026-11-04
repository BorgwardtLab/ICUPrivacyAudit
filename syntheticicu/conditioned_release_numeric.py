"""Build the second release mode required by the existing full privacy protocol."""

import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
import torch

from syntheticicu.generation_numeric import (
    generate_release,
    prepare_release,
    _checkpoint_hash,
    _complete,
)


def prepare_conditioned_release(cfg):
    """Prepare immutable per-center plans before launching independent workers."""
    root = Path(cfg.output_dir).resolve()
    destination = root / "conditioned"
    checkpoint = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    centers = checkpoint["schema"]["centers"]
    if cfg.generate.num_samples % len(centers):
        raise ValueError("Conditioned sample count must divide evenly across centers")
    plan = {
        "seed": int(cfg.seed),
        "generate": OmegaConf.to_container(cfg.generate, resolve=True),
        "signature": checkpoint["data_signature"],
        "centers": centers,
        "model_sha256": _checkpoint_hash(root / "model.pt"),
    }
    plan_path = destination / "plan.json"
    destination.mkdir(parents=True, exist_ok=True)
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Conditioned release recovery plan differs")
    if not plan_path.exists():
        temporary = plan_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(plan, indent=2) + "\n")
        temporary.replace(plan_path)
    count = cfg.generate.num_samples // len(centers)
    for index, center in enumerate(centers):
        shard = destination / "shards" / center
        shard.mkdir(parents=True, exist_ok=True)
        link = shard / "model.pt"
        if link.exists() and link.resolve() != (root / "model.pt").resolve():
            raise ValueError("Conditioned shard uses another checkpoint")
        if not link.exists():
            link.symlink_to(root / "model.pt")
        prepare_release(conditioned_shard_settings(cfg, plan, index))
    return plan


def conditioned_shard_settings(cfg, plan, index):
    """Retain the sequential workflow's center order and seed offsets."""
    centers = plan["centers"]
    if not 0 <= index < len(centers):
        raise ValueError("Invalid conditioned center shard")
    count = cfg.generate.num_samples // len(centers)
    settings = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    settings.output_dir = str(
        Path(cfg.output_dir).resolve() / "conditioned/shards" / centers[index]
    )
    settings.seed = int(cfg.seed) + index * count
    settings.generate.center, settings.generate.num_samples = centers[index], count
    return settings


def generate_conditioned_center(cfg, index):
    """Generate one already prepared center without touching other centers."""
    root = Path(cfg.output_dir).resolve()
    plan = json.loads((root / "conditioned/plan.json").read_text())
    if plan["generate"] != OmegaConf.to_container(cfg.generate, resolve=True):
        raise ValueError("Conditioned release recovery plan differs")
    if plan["seed"] != int(cfg.seed):
        raise ValueError("Conditioned release seed differs")
    return generate_release(conditioned_shard_settings(cfg, plan, index))


def merge_conditioned_release(cfg):
    """Merge prepared, complete centers using the original ordering."""
    root = Path(cfg.output_dir).resolve()
    plan = json.loads((root / "conditioned/plan.json").read_text())
    shards = [
        root / "conditioned/shards" / center / "generated" for center in plan["centers"]
    ]
    if not all(_complete(shard) for shard in shards):
        raise FileNotFoundError("Conditioned generation has incomplete center shards")
    return merge_conditioned_shards(shards, root / "conditioned/generated", plan)


def generate_conditioned_release(cfg):
    """Generate all centers sequentially, preserving the existing entry point."""
    plan = prepare_conditioned_release(cfg)
    marker = Path(cfg.output_dir).resolve() / "conditioned/generated/generation.json"
    if marker.exists():
        return marker.parent
    for index in range(len(plan["centers"])):
        generate_conditioned_center(cfg, index)
    return merge_conditioned_release(cfg)


def merge_conditioned_shards(shards, destination, plan):
    """Merge already validated shards with unique IDs and a completion marker last."""
    core, meta, tokens, manifests, offset = [], [], [], [], 0
    for shard in shards:
        manifest = json.loads((shard / "generation.json").read_text())
        if manifest["data_signature"] != plan["signature"]:
            raise ValueError("Conditioned shards have different prepared populations")
        frame, info = pd.read_parquet(shard / "core.parquet"), pd.read_parquet(
            shard / "meta.parquet"
        )
        frame["stay_id"] += offset
        info["stay_id"] += offset
        info["patient_uid"] += offset
        offset += len(info)
        core.append(frame)
        meta.append(info)
        tokens.extend(
            torch.load(shard / "tokens.pt", map_location="cpu", weights_only=True)
        )
        manifests.append(manifest)
    if len({m["model_sha256"] for m in manifests}) != 1:
        raise ValueError("Conditioned shards use different model weights")
    info = pd.concat(meta, ignore_index=True)
    if info.stay_id.duplicated().any() or offset != plan["generate"]["num_samples"]:
        raise ValueError("Conditioned release IDs or sample count are invalid")
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in [("core", pd.concat(core, ignore_index=True)), ("meta", info)]:
        temporary = destination / f"{name}.parquet.partial"
        value.to_parquet(temporary, index=False)
        temporary.replace(destination / f"{name}.parquet")
    temporary = destination / "tokens.pt.partial"
    torch.save(tokens, temporary)
    temporary.replace(destination / "tokens.pt")
    manifest = {
        "data_signature": plan["signature"],
        "model_sha256": manifests[0]["model_sha256"],
        "center": "mixed_conditioned",
        "num_samples": offset,
        "terminated": int(info.terminated.sum()),
        "plan": plan,
    }
    temporary = destination / "generation.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "generation.json")
    return destination
