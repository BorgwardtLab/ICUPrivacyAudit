"""Independent training, generation, and mortality evaluation for numerical EHRs."""

from functools import partial
import hashlib
import json
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import get_cosine_schedule_with_warmup

from syntheticicu.data_numeric import (
    EOS,
    GAP,
    PAD,
    SOS,
    category_index,
    collate_numerical,
    numerical_bins,
    prepare_numerical_data,
    repeated_training_indices,
)


def output_directory(cfg):
    """Resolve the configured workflow artifact directory."""
    return Path(cfg.output_dir).resolve()


def _new_output(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists; choose a new output_dir or set overwrite=true"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def data_signature(bundle):
    """Fingerprint the schema, patient partitions, labels, and numerical data."""
    digest = hashlib.sha256(json.dumps(bundle["schema"], sort_keys=True).encode())
    if "preparation_contract" in bundle:
        digest.update(
            json.dumps(bundle["preparation_contract"], sort_keys=True).encode()
        )
    for stay in bundle["stays"]:
        identity = {
            name: stay[name]
            for name in [
                "centre",
                "stay_id",
                "patient_uid",
                "split",
                "death_icu",
                "demographics",
            ]
        }
        digest.update(json.dumps(identity, sort_keys=True).encode())
        digest.update(stay["values"].numpy().tobytes())
        digest.update(stay["hours"].numpy().tobytes())
    return digest.hexdigest()


def prepare_numerical(cfg):
    """Prepare and save reusable data and patient split artifacts."""
    root = output_directory(cfg)
    target = root / "prepared.pt"
    _new_output(target, cfg.overwrite)
    bundle = prepare_numerical_data(cfg.data)
    bundle["signature"] = data_signature(bundle)
    torch.save(bundle, target)
    manifest = [
        {name: stay[name] for name in ["centre", "stay_id", "patient_uid", "split"]}
        for stay in bundle["stays"]
    ]
    pd.DataFrame(manifest).to_parquet(root / "splits.parquet", index=False)
    (root / "preparation.json").write_text(
        json.dumps(bundle["report"], indent=2) + "\n"
    )
    (root / "schema.json").write_text(json.dumps(bundle["schema"], indent=2) + "\n")
    return target


def _resolved_model_config(cfg, schema):
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=False))
    model_cfg.config.vocab_size = len(schema["vocabulary"])
    model_cfg.config.pad_token_id = PAD
    model_cfg.config.bos_token_id = SOS
    model_cfg.config.eos_token_id = EOS
    model_cfg.bins = None
    model_cfg.gap_token_id = GAP
    model_cfg.num_centers = len(schema["centers"])
    return model_cfg


class NumericalTrainingTask(L.LightningModule):
    """Lightning training wrapper around the standalone numerical model."""

    def __init__(
        self,
        model,
        learning_rate,
        weight_decay,
        betas,
        warmup_steps,
        scheduler_steps,
        resume_contract,
    ):
        super().__init__()
        self.model = model
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.betas = tuple(betas)
        self.warmup_steps, self.scheduler_steps = warmup_steps, scheduler_steps
        self.resume_contract = resume_contract

    def on_save_checkpoint(self, checkpoint):
        """Bind restart state to the exact data, model, and optimizer settings."""
        checkpoint["numerical_contract"] = self.resume_contract

    def on_load_checkpoint(self, checkpoint):
        """Reject resuming a checkpoint under incompatible training settings."""
        saved_contract = dict(checkpoint["numerical_contract"])
        # Checkpoints written before this option always shuffled within the hour.
        saved_contract.setdefault("shuffle_within_hour", True)
        if saved_contract != self.resume_contract:
            raise ValueError(
                "Resume checkpoint does not match data/model/optimizer settings"
            )

    def _step(self, batch, phase):
        output = self.model(**batch, compute_loss=True)
        self.log(
            f"{phase}_loss",
            output.loss,
            on_step=phase == "train",
            on_epoch=True,
            batch_size=len(batch["input_ids"]),
            sync_dist=True,
        )
        self.log(
            f"{phase}_token_loss",
            output.token_loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["input_ids"]),
            sync_dist=True,
        )
        self.log(
            f"{phase}_numerical_loss",
            output.numerical_loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["input_ids"]),
            sync_dist=True,
        )
        return output.loss

    def training_step(self, batch, batch_idx):
        """Optimize token CE and numerical NLL."""
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Measure the same objective on disjoint validation patients."""
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Use AdamW with optional step-based cosine decay and warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        if self.scheduler_steps is None:
            return optimizer
        if not 0 <= self.warmup_steps < self.scheduler_steps:
            raise ValueError(
                "Warmup must be shorter than the positive scheduler budget"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.warmup_steps, self.scheduler_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def train_numerical(cfg):
    """Train to the configured epoch/step budget and export the final model."""
    root = output_directory(cfg)
    target = root / "model.pt"
    _new_output(target, cfg.overwrite)
    if cfg.train.max_steps < 1:
        raise ValueError("train.max_steps must be positive")
    bundle = torch.load(root / "prepared.pt", map_location="cpu", weights_only=True)
    schema = bundle["schema"]
    L.seed_everything(cfg.seed, workers=True)
    model_cfg = _resolved_model_config(cfg, schema)
    if cfg.data.max_tokens + 2 > model_cfg.config.max_position_embeddings:
        raise ValueError(
            "Model context must accommodate chunk length plus conditioning"
        )
    model = hydra.utils.instantiate(model_cfg, bins=numerical_bins(schema))
    signature = data_signature(bundle)
    effective_batch = (
        cfg.train.batch_size * cfg.train.devices * cfg.train.accumulate_grad_batches
    )
    contract = {
        "data_signature": signature,
        "model_cfg": OmegaConf.to_container(model_cfg, resolve=True),
        "optimizer": {
            key: OmegaConf.to_container(cfg.train, resolve=True)[key]
            for key in [
                "learning_rate",
                "weight_decay",
                "betas",
                "warmup_steps",
                "scheduler_steps",
            ]
        },
        "effective_batch": effective_batch,
        "seed": int(cfg.seed),
        "shuffle_within_hour": cfg.train.shuffle_within_hour,
        "canaries": {
            "count": int(cfg.train.n_canaries),
            "duplication_factor": int(cfg.train.duplication_factor),
        },
    }
    task = NumericalTrainingTask(
        model,
        cfg.train.learning_rate,
        cfg.train.weight_decay,
        cfg.train.betas,
        cfg.train.warmup_steps,
        cfg.train.scheduler_steps,
        contract,
    )
    indices, canaries = repeated_training_indices(
        bundle, cfg.train.n_canaries, cfg.train.duplication_factor, cfg.seed
    )
    val_indices = [
        index
        for index, record in enumerate(bundle["records"])
        if record["split"] == "val"
    ]
    loader_options = dict(
        batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
    )
    training = DataLoader(
        Subset(bundle["records"], indices),
        shuffle=True,
        collate_fn=partial(collate_numerical, shuffle=cfg.train.shuffle_within_hour),
        **loader_options,
    )
    validation = DataLoader(
        Subset(bundle["records"], val_indices),
        collate_fn=collate_numerical,
        **loader_options,
    )
    last_checkpoint = root / "checkpoints/last.ckpt"
    if last_checkpoint.exists() and not cfg.train.resume:
        raise FileExistsError(
            "A training checkpoint exists; set train.resume=true or use a new directory"
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=root / "checkpoints",
        filename="step-{step}",
        every_n_train_steps=cfg.train.checkpoint_every_n_steps,
        save_last=True,
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    trainer = L.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        precision=cfg.train.precision,
        max_steps=cfg.train.max_steps,
        max_epochs=cfg.train.max_epochs,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        logger=CSVLogger(save_dir=str(root), name="csv_logs"),
        callbacks=[checkpoint_callback],
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
    trainer.fit(
        task,
        training,
        validation,
        ckpt_path=(
            str(last_checkpoint)
            if cfg.train.resume and last_checkpoint.exists()
            else None
        ),
    )
    if not trainer.is_global_zero:
        return target
    training_report = {
        "steps": int(trainer.global_step),
        "canaries": canaries,
        "duplication_factor": int(cfg.train.duplication_factor),
        "seed": int(cfg.seed),
        "effective_batch": int(effective_batch),
        "settings": OmegaConf.to_container(cfg.train, resolve=True),
    }
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "model_cfg": OmegaConf.to_container(model_cfg, resolve=True),
            "schema": schema,
            "data_signature": signature,
            "training": training_report,
        },
        target.with_suffix(".pt.partial"),
    )
    target.with_suffix(".pt.partial").replace(target)
    (root / "training.json").write_text(
        json.dumps({"data_signature": signature, **training_report}, indent=2) + "\n"
    )
    return target


def load_numerical_model(path, device):
    """Restore an exported model without loading the original Parquet files."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = hydra.utils.instantiate(
        OmegaConf.create(checkpoint["model_cfg"]),
        bins=numerical_bins(checkpoint["schema"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


class NumericalGrammar:
    """Constrain demographics, per-hour feature uniqueness, and the time horizon."""

    def __init__(self, schema, min_hours):
        self.schema = schema
        self.vocabulary = schema["vocabulary"]
        self.demographics = [
            [self.vocabulary[token] for token in spec["tokens"]]
            for spec in schema["demographics"].values()
        ]
        self.feature_groups = [
            [self.vocabulary[token] for token in spec["tokens"]]
            for spec in schema["features"].values()
        ]
        if not 1 <= min_hours <= schema["max_hours"]:
            raise ValueError("min_hours must lie within the prepared horizon")
        self.min_hours = min_hours

    def __call__(self, input_ids, scores):
        constrained = torch.full_like(scores, -torch.inf)
        demo_count = len(self.demographics)
        for row, sequence in enumerate(input_ids.tolist()):
            position = len(sequence) - 1
            if position < demo_count:
                allowed = self.demographics[position]
            elif position == demo_count:
                allowed = [GAP]
            else:
                gaps = [index for index, token in enumerate(sequence) if token == GAP]
                if len(gaps) > self.schema["max_hours"]:
                    allowed = [EOS]
                else:
                    this_hour = set(sequence[gaps[-1] + 1 :])
                    allowed = [
                        token
                        for group in self.feature_groups
                        if not this_hour.intersection(group)
                        for token in group
                    ]
                    any_observation = any(
                        token > GAP for token in sequence[demo_count + 2 :]
                    )
                    if any_observation:
                        allowed.append(GAP)
                        if len(gaps) >= self.min_hours:
                            allowed.append(EOS)
            constrained[row, allowed] = scores[row, allowed]
        return constrained


def _representative(specification, index):
    if index == len(specification["tokens"]) - 1:
        return float("nan")
    if "categories" in specification:
        return specification["categories"][index]
    return (specification["edges"][index] + specification["edges"][index + 1]) / 2


def decode_generated(result, schema, labels, centers, offset):
    """Convert token-aligned outputs to separate synthetic core/meta tables."""
    vocabulary = schema["vocabulary"]
    features = {
        vocabulary[token]: name
        for name, spec in schema["features"].items()
        for token in spec["tokens"]
    }
    demographics = {
        vocabulary[token]: (name, _representative(spec, index))
        for name, spec in schema["demographics"].items()
        for index, token in enumerate(spec["tokens"])
    }
    core_rows, meta_rows = [], []
    for row, sequence in enumerate(result["input_ids"].cpu().tolist()):
        identifier = offset + row
        center = (
            "__null__"
            if centers[row] == len(schema["centers"])
            else schema["centers"][centers[row]]
        )
        metadata = {
            "stay_id": identifier,
            "patient_uid": identifier,
            "centre": center,
            "death_icu": bool(labels[row]),
            "terminated": EOS in sequence,
        }
        events = {}
        hour = -1
        for position, token in enumerate(sequence):
            if token == EOS or token == PAD:
                break
            if token == GAP:
                hour += 1
            elif token in demographics:
                name, value = demographics[token]
                metadata[name] = value
            elif token in features:
                if hour < 0:
                    raise ValueError(
                        "Physiology generated before the demographic separator"
                    )
                event = events.setdefault(
                    hour,
                    {
                        "stay_id": identifier,
                        "centre": center,
                        "stay_time": float(hour),
                        **{name: float("nan") for name in schema["features"]},
                    },
                )
                event[features[token]] = float(result["values"][row, position])
        if not events:
            raise ValueError("Token budget too short to generate any physiology")
        if set(schema["demographics"]) - set(metadata):
            raise ValueError("Token budget too short to generate demographics")
        core_rows.extend(events.values())
        meta_rows.append(metadata)
    return core_rows, meta_rows


def trim_generated_trails(result, numerical):
    """Remove only GAP suffixes; preserve truncation, values, and internal hours."""
    shape = result["input_ids"].shape
    if len(shape) != 2 or any(value.shape != shape for value in result.values()):
        raise ValueError("Generated token, value and time matrices must align")
    cleaned = {
        name: torch.full_like(value, float("nan") if name == "values" else 0)
        for name, value in result.items()
    }
    for row, ids in enumerate(result["input_ids"]):
        stops = ((ids == EOS) | (ids == PAD)).nonzero().flatten()
        stop = int(stops[0]) if len(stops) else len(ids)
        observed = numerical[ids[:stop]].nonzero().flatten()
        if not len(observed):
            raise ValueError("Generated stay contains no physiology")
        end = int(observed[-1]) + 1
        if not torch.all(ids[end:stop] == GAP):
            raise ValueError("Unexpected non-GAP token after final physiology")
        positions = torch.arange(end, device=ids.device)
        if stop < len(ids) and ids[stop] == EOS:
            positions = torch.cat([positions, positions.new_tensor([stop])])
        count = len(positions)
        for name in cleaned:
            cleaned[name][row, :count] = result[name][row, positions]
        if "time_positions" in cleaned:
            cleaned["time_positions"][row, :count] = torch.cat(
                [
                    ids.new_zeros(1),
                    (cleaned["input_ids"][row, : count - 1] == GAP).long(),
                ]
            ).cumsum(0)
        assert cleaned["input_ids"][row, end - 1] != GAP
        assert count <= shape[1]
    return cleaned


def generate_numerical(cfg):
    """Generate a labelled numerical corpus with no real-patient prompts."""
    root = output_directory(cfg)
    destination = root / "generated"
    for name in ["core.parquet", "meta.parquet", "tokens.pt", "generation.json"]:
        _new_output(destination / name, cfg.overwrite)
    L.seed_everything(cfg.seed, workers=True)
    model, checkpoint = load_numerical_model(root / "model.pt", cfg.generate.device)
    schema = checkpoint["schema"]
    count = cfg.generate.num_samples
    if (
        count < 1
        or cfg.generate.batch_size < 1
        or not 0 <= cfg.generate.positive_fraction <= 1
    ):
        raise ValueError(
            "Positive sample/batch counts and a valid positive fraction required"
        )
    if cfg.generate.max_new_tokens + 3 > model.config.max_position_embeddings:
        raise ValueError("Generation budget exceeds the configured context")
    positive = int(round(count * cfg.generate.positive_fraction))
    labels = np.concatenate(
        [np.zeros(count - positive, dtype=np.int64), np.ones(positive, dtype=np.int64)]
    )
    labels = np.random.default_rng(cfg.seed).permutation(labels)
    center_id = (
        len(schema["centers"])
        if cfg.generate.center is None
        else schema["centers"].index(cfg.generate.center)
    )
    grammar = NumericalGrammar(schema, cfg.generate.min_hours)
    core_rows, meta_rows, tokens = [], [], []
    for start in range(0, count, cfg.generate.batch_size):
        batch_labels = labels[start : start + cfg.generate.batch_size]
        size = len(batch_labels)
        device = cfg.generate.device
        result = model.generate(
            torch.full((size, 1), SOS, dtype=torch.long, device=device),
            torch.zeros(size, 1, device=device),
            torch.tensor(batch_labels, dtype=torch.long, device=device),
            torch.full((size,), center_id, dtype=torch.long, device=device),
            max_new_tokens=cfg.generate.max_new_tokens,
            time_positions=torch.zeros(size, 1, dtype=torch.long, device=device),
            temperature=cfg.generate.temperature,
            logits_processors=[grammar],
        )
        if cfg.generate.get("trim_empty_trails", False):
            result = trim_generated_trails(result, model.bins.numerical)
        rows, metadata = decode_generated(
            result, schema, batch_labels, [center_id] * size, start
        )
        core_rows.extend(rows)
        meta_rows.extend(metadata)
        tokens.append({name: value.cpu() for name, value in result.items()})
    pd.DataFrame(core_rows).to_parquet(destination / "core.parquet", index=False)
    pd.DataFrame(meta_rows).to_parquet(destination / "meta.parquet", index=False)
    torch.save(tokens, destination / "tokens.pt")
    (destination / "generation.json").write_text(
        json.dumps(
            {
                "data_signature": checkpoint["data_signature"],
                "num_samples": count,
                "center": cfg.generate.center,
                "positive_fraction": cfg.generate.positive_fraction,
                "terminated": sum(row["terminated"] for row in meta_rows),
                "seed": int(cfg.seed),
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def _midpoint_values(values, schema):
    result = np.asarray(values, dtype=float).copy()
    for index, spec in enumerate(schema["features"].values()):
        observed = np.isfinite(result[:, index])
        bins = np.searchsorted(
            spec["edges"][1:-1], result[observed, index], side="right"
        )
        edges = np.asarray(spec["edges"])
        result[observed, index] = (edges[bins] + edges[bins + 1]) / 2
    return result


def _mortality_features(stays, schema, max_hours, midpoints=False):
    features, labels = [], []
    for stay in stays:
        values = stay["values"].numpy()[stay["hours"].numpy() < max_hours].astype(float)
        if midpoints:
            values = _midpoint_values(values, schema)
        row = []
        for column in values.T:
            observed = column[np.isfinite(column)]
            row.extend(
                [
                    float(observed.mean()),
                    float(observed.min()),
                    float(observed.max()),
                    float(observed[-1]),
                    float(len(observed)),
                ]
                if len(observed)
                else [float("nan")] * 4 + [0.0]
            )
        for name, spec in schema["demographics"].items():
            value = stay["demographics"][name]
            index = category_index(value, spec, "clip")
            row.append(_representative(spec, index))
        features.append(row)
        labels.append(int(stay["death_icu"]))
    return np.asarray(features), np.asarray(labels)
