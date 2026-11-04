"""Residual-aware bridge into the existing full binned privacy audit."""

from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from syntheticicu.audit_numeric import attach_feature_ranges, batch_likelihood_scores
from syntheticicu.data_classes import CENTRE_MAP
from syntheticicu.data_numeric import PAD, SOS, collate_numerical
from syntheticicu.privacy_audit_ext import DEMOGRAPHIC_ORDER


class ResidualAuditBatch(list):
    """The legacy five-tensor layout plus explicitly aligned numerical inputs."""

    def __init__(self, values, numerical):
        super().__init__(values)
        self.numerical = numerical
        self.residuals = numerical["residuals"]


class NumericalAuditRecords(Dataset):
    """Keep integer identity metadata outside collated conditioning tensors."""

    def __init__(self, records, schema, center_code):
        self.records = records
        self.y = np.asarray(
            [[r["stay_id"], r["class_labels"], 0] for r in records], dtype=np.int64
        )
        self.stay_sources = np.asarray([r["centre"] for r in records])
        self.static_names = self.statics_set = DEMOGRAPHIC_ORDER
        self.schema, self.center_code = schema, center_code

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class NumericalAuditDatamodule:
    """Expose the same ordered train/validation/holdout loaders as the full audit."""

    def __init__(self, data, center_code):
        self.schema, self.center_code = data.bundle["schema"], center_code
        self.base_icu_dataset = NumericalAuditRecords(
            data.bundle["records"], self.schema, center_code
        )
        for name, split in [("train", "train"), ("val", "val"), ("test", "test")]:
            indices = [
                i for i, r in enumerate(data.bundle["records"]) if r["split"] == split
            ]
            setattr(self, f"{name}_icu_dataset", Subset(self.base_icu_dataset, indices))
            setattr(self, f"{name}_indices", np.asarray(indices, dtype=np.int64))
        self.tokenizer = SimpleNamespace(
            vocab=self.schema["vocabulary"],
            demographic_bins={
                name: (
                    len(self.schema["demographics"][name]["tokens"])
                    if name in self.schema["demographics"]
                    else 1
                )
                for name in DEMOGRAPHIC_ORDER
            },
        )

    def collate_fn(self, records):
        numerical = collate_numerical(records)
        ids = torch.tensor(
            [
                [r["stay_id"], r["class_labels"], 0, self.center_code[r["centre"]]]
                for r in records
            ],
            dtype=torch.float32,
        )
        # The shared audit recovers exact keys from base.y, never this tensor.
        statics = torch.tensor([self._statics(r) for r in records], dtype=torch.long)
        return ResidualAuditBatch(
            [
                numerical["input_ids"],
                numerical["time_positions"],
                numerical["input_ids"] != PAD,
                statics,
                ids,
            ],
            numerical,
        )

    def _statics(self, record):
        present = set(record["input_ids"].tolist())
        return [
            (
                next(
                    (
                        i
                        for i, token in enumerate(
                            self.schema["demographics"][name]["tokens"]
                        )
                        if self.schema["vocabulary"][token] in present
                    ),
                    0,
                )
                if name in self.schema["demographics"]
                else 0
            )
            for name in DEMOGRAPHIC_ORDER
        ]

    def val_dataloader(self):
        return DataLoader(
            self.val_icu_dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


class NumericalAuditModel(nn.Module):
    """Preserve full-audit statistics while conditioning on observed residuals."""

    def __init__(self, core, data, score_kind="categorical"):
        super().__init__()
        if score_kind not in {"categorical", "numerical"}:
            raise ValueError("score_kind must be categorical or numerical")
        self.core, self.score_kind = core, score_kind
        self.center_code = _center_codes(data.bundle["schema"]["centers"])
        self._datamodule = NumericalAuditDatamodule(data, self.center_code)
        self.vocabulary = data.bundle["schema"]["vocabulary"]
        self.label_embedder = SimpleNamespace(
            num_classes=core.num_classes,
            center_dropout_prob=core.center_dropout,
            dropout_prob=core.outcome_dropout,
        )
        attach_feature_ranges(core, data.bundle["schema"])
        self.eval()

    @property
    def device(self):
        return next(self.core.parameters()).device

    def score_privacy_batch(
        self,
        batch,
        exact_ids,
        exact_centers,
        k=0.2,
        exclude_static_tokens=False,
        **kwargs,
    ):
        numerical = {
            name: value.to(self.device) for name, value in batch.numerical.items()
        }
        settings = [
            numerical,
            {
                **numerical,
                "center_ids": torch.full_like(
                    numerical["center_ids"], self.core.num_centers
                ),
            },
            {**numerical, "class_labels": 1 - numerical["class_labels"]},
        ]
        results = [batch_likelihood_scores(self.core, inputs, k) for inputs in settings]
        targets = numerical["input_ids"][:, 1:].cpu().numpy()
        mask = (targets != PAD) & (targets != self.core.config.eos_token_id)
        if exclude_static_tokens:
            statics = [
                value for name, value in self.vocabulary.items() if "_static_" in name
            ]
            mask &= ~np.isin(targets, statics)
        length_mask = mask.copy()
        if self.score_kind == "numerical":
            mask &= results[0]["numeric_mask"]
        arrays = [self._logp(result, targets) for result in results]
        kept = np.flatnonzero(mask.any(axis=1))
        output = {}
        for name, values in zip(["s_true", "s_nullc", "s_flip"], arrays):
            output[name] = np.asarray([_min_k(values[i][mask[i]], k) for i in kept])
        output.update(
            mean_true=np.asarray([arrays[0][i][mask[i]].mean() for i in kept]),
            lengths=length_mask.sum(axis=1)[kept],
            stay_ids=np.asarray(exact_ids)[kept],
            center_ids=np.asarray(exact_centers)[kept],
            class_labels=numerical["class_labels"].cpu().numpy()[kept],
            statics=batch[-2].numpy()[kept],
        )
        return output

    def _logp(self, result, targets):
        if self.score_kind == "categorical":
            return result["token_logp"]
        bins = self.core.bins
        ranges = self.core.audit_feature_ranges.detach().cpu().numpy()
        widths = (bins.upper - bins.lower).detach().cpu().numpy()
        values = result["numeric_logp"].copy()
        mask = result["numeric_mask"]
        values[mask] += np.log(ranges[targets[mask]] / widths[targets[mask]])
        return values

    def collect_privacy_sequences(
        self,
        subset,
        batch_size=32,
        max_batches=None,
        collect_statics=True,
        collect_labels=False,
        collect_centers=False,
        shuffle=None,
    ):
        if shuffle:
            raise ValueError("Ordered collection required for exact privacy keys")
        selected = list(subset.indices)
        if max_batches is not None:
            selected = selected[: batch_size * max_batches]
        inverse = {value: name for name, value in self.vocabulary.items()}
        seen, sequences, ids, labels, centers, statics = set(), [], [], [], [], []
        for index in selected:
            record = subset.dataset.records[index]
            key = (record["centre"], record["stay_id"])
            tokens = record["input_ids"].tolist()
            if key in seen:
                tokens = tokens[len(self._datamodule.schema["demographics"]) + 2 :]
            seen.add(key)
            sequences.append(
                [inverse[token] for token in tokens if token not in {PAD, SOS}]
            )
            ids.append(record["stay_id"])
            labels.append(record["class_labels"])
            centers.append(self.center_code[record["centre"]])
            statics.append(self._datamodule._statics(record))
        result = [sequences, ids]
        for enabled, values in [
            (collect_statics, statics),
            (collect_labels, labels),
            (collect_centers, centers),
        ]:
            if enabled:
                result.append(values)
        return tuple(result)


def _min_k(values, fraction):
    count = max(1, int(len(values) * fraction))
    return float(np.sort(values)[:count].mean())


def export_audit_corpus(root, schema, destination):
    """Export generated token strings without re-binning or discarding EOS status."""
    inverse = {value: name for name, value in schema["vocabulary"].items()}
    batches = torch.load(
        root / "generated/tokens.pt", map_location="cpu", weights_only=True
    )
    meta = pd.read_parquet(root / "generated/meta.parquet").sort_values("stay_id")
    sequences = [
        [inverse[token] for token in row if token not in {PAD, SOS}]
        for batch in batches
        for row in batch["input_ids"].tolist()
    ]
    if len(meta) != len(sequences):
        raise ValueError("Generated token and metadata counts do not match")
    centers = [
        6 if center == "__null__" else CENTRE_MAP[center] for center in meta.centre
    ]
    joblib.dump(
        (sequences, meta.death_icu.to_numpy(dtype=int), np.asarray(centers)),
        destination,
    )
    return str(destination)


def _center_codes(centers):
    if not set(centers).issubset(CENTRE_MAP):
        raise ValueError("Full audit requires the binned pipeline's known center names")
    return {name: CENTRE_MAP[name] for name in centers}
