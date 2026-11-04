"""Continuous data source for the existing binned classifier task implementations."""

from bisect import bisect_right
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from syntheticicu.data_classes import EarlyWarningDatamodule, EarlyWarningSubset
from syntheticicu.utils import get_feature_set


class NumericalWindows:
    """Lazy four-hour windows with the binned onset/end cutoff and missingness.

    Raw measurements are forward-filled using past observations only. Initial
    missing values use the middle training bin's midpoint, the continuous analogue
    of the binned loader's middle-bin fill. IDs are collision-free integer aliases;
    identity_map retains the exact original center, stay, and patient identifiers.
    """

    reference_dtype = torch.long

    def __init__(
        self,
        stays,
        schema,
        lookback,
        window_size,
        feature_set,
        min_length,
        syn_lookback=None,
        include_missing_mask=True,
        missingness_threshold="vitals",
        impute_missing="forwardfill",
        identity_map=None,
    ):
        if lookback < 0 or window_size < 1 or min_length < 0:
            raise ValueError(
                "Nonnegative lookback/minimum length and positive window required"
            )
        self.lookback = lookback
        self.syn_lookback = lookback if syn_lookback is None else syn_lookback
        if self.syn_lookback < 0 or impute_missing != "forwardfill":
            raise ValueError("Nonnegative synthetic lookback and forwardfill required")
        self.window_size, self.feature_set = window_size, feature_set
        self.features = sorted(
            set(schema["features"]) & set(get_feature_set(feature_set))
        )
        if not self.features:
            raise ValueError("Requested feature set is absent from numerical schema")
        self.include_missing_mask, self.use_sliding_windows = include_missing_mask, True
        self.label, self.statics_set = ["death_icu"], list(schema["demographics"])
        self.num_bins = max(len(s["edges"]) - 1 for s in schema["features"].values())
        self.identity_map = (
            identity_map
            if identity_map is not None
            else {_identity(stay): index for index, stay in enumerate(stays)}
        )
        self._values, self._masks, metadata = [], [], []
        for stay in stays:
            self._add_stay(stay, schema, min_length, metadata)
        self._finish(metadata, missingness_threshold)

    def _add_stay(self, stay, schema, min_length, metadata):
        synthetic = stay["split"] == "synthetic"
        hours = stay["classifier_hours"] if synthetic else stay["hours"]
        values = stay["classifier_values"] if synthetic else stay["values"]
        origin = int(synthetic)
        length = int(hours[-1]) + 1 - origin
        if length < min_length or (synthetic and length == min_length):
            return
        cut = min(self.lookback, self.syn_lookback) if synthetic else self.lookback
        kept = length - cut
        if kept < self.window_size:
            return
        columns = [list(schema["features"]).index(name) for name in self.features]
        raw = np.full((kept, len(columns)), np.nan, dtype=np.float32)
        selected = hours.numpy() - origin < kept
        indices = hours.numpy()[selected] - origin
        raw[indices] = values.numpy()[selected][:, columns]
        self._append_values(stay, schema, raw, metadata, length - 1 + origin, origin)

    def _append_values(self, stay, schema, raw, metadata, reference, origin):
        """Impute the already bounded grid and retain its original missingness."""
        if np.isinf(raw).any():
            raise ValueError("Infinite numerical classification measurements")
        missing = np.isnan(raw)
        filled = pd.DataFrame(raw).ffill().to_numpy()
        for column, name in enumerate(self.features):
            edges = schema["features"][name]["edges"]
            middle = (len(edges) - 1) // 2
            filled[np.isnan(filled[:, column]), column] = (
                edges[middle] + edges[middle + 1]
            ) / 2
        self._values.append(filled)
        self._masks.append(missing)
        metadata.append((stay, len(raw) - self.window_size + 1, reference, origin))

    def _finish(self, metadata, threshold):
        excluded = (
            set(get_feature_set(threshold)) if isinstance(threshold, str) else set()
        )
        if not isinstance(threshold, str):
            raise ValueError(
                "Numerical parity requires the named feature-set missingness policy"
            )
        self._mask_columns = [
            i for i, name in enumerate(self.features) if name not in excluded
        ]
        self.feature_names_with_missingness = self.features + [
            f"{self.features[i]}__mask" for i in self._mask_columns
        ]
        self.missingness_tensor = self._masks if self.include_missing_mask else None
        counts = [entry[1] for entry in metadata]
        self._ends = np.cumsum(counts).tolist()
        ids, times, onset, references, labels, sources = [], [], [], [], [], []
        for stay, count, reference, origin in metadata:
            ids.extend([self.identity_map[_identity(stay)]] * count)
            times.extend(
                range(
                    self.window_size - 1 + origin, self.window_size - 1 + origin + count
                )
            )
            onset.extend([reference if stay["death_icu"] else 0] * count)
            references.extend([reference] * count)
            labels.extend([int(stay["death_icu"])] * count)
            sources.extend(
                ["syn" if stay["split"] == "synthetic" else stay["centre"]] * count
            )
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.times = torch.tensor(times, dtype=torch.long)
        self.onset = torch.tensor(onset, dtype=self.reference_dtype)
        self.reference_hours = torch.tensor(references, dtype=self.reference_dtype)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.stay_sources = pd.Series(sources, dtype="string")

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        stay = bisect_right(self._ends, index)
        start = index - (self._ends[stay - 1] if stay else 0)
        values = self._values[stay][start : start + self.window_size]
        if self.include_missing_mask:
            masks = self._masks[stay][
                start : start + self.window_size, self._mask_columns
            ]
            values = np.concatenate([values, masks], axis=1)
        return (
            torch.tensor(values.T, dtype=torch.float32),
            torch.zeros(len(self.statics_set)),
            self.y[index],
            self.times[index],
            self.onset[index],
            self.ids[index],
        )


def _identity(stay):
    return (
        stay["split"] == "synthetic",
        stay["centre"],
        stay["stay_id"],
        stay["patient_uid"],
    )


class NumericalTaskDatamodule(EarlyWarningDatamodule):
    """Reuse the main binned sampling, split, and subset behavior verbatim."""

    window_class = NumericalWindows

    def __init__(self, data, impute_missing="forwardfill", syn_lookback=None, **kwargs):
        super().__init__(
            impute_missing=impute_missing,
            syn_lookback=kwargs["lookback"] if syn_lookback is None else syn_lookback,
            **kwargs,
        )
        self.data = data

    def setup(self, stage=None):
        if self.full_dataset is not None:
            return
        all_stays = self.data.bundle["stays"] + self.data.synthetic
        identities = getattr(self.data, "utility_aliases", None)
        if identities is None:
            identities = {_identity(s): index for index, s in enumerate(all_stays)}
        stays = []
        if self.real_path is not None:
            stays.extend(self.data.bundle["stays"])
        if self.syn_path is not None:
            stays.extend(self.data.synthetic)
        self.full_dataset = self.window_class(
            stays,
            self.data.bundle["schema"],
            lookback=self.lookback,
            window_size=self.window_size,
            feature_set=self.feature_set,
            min_length=self.min_length,
            syn_lookback=self.syn_lookback,
            include_missing_mask=self.include_missing_mask,
            missingness_threshold=self.missingness_threshold,
            impute_missing=self.impute_missing,
            identity_map=identities,
        )
        if not len(self.full_dataset):
            raise ValueError(
                "No numerical stays remain after the binned cutoff/filter rules"
            )
        self.split()
        self._apply_lookback_per_split()
        self.train_dataset = EarlyWarningSubset(self.full_dataset, self.train_indices)
        self.test_dataset = EarlyWarningSubset(self.full_dataset, self.test_indices)
        self.val_dataset = (
            EarlyWarningSubset(self.full_dataset, self.val_indices)
            if self.val_set
            else None
        )


def active_tasks():
    """The eleven reported real/synthetic mixture classifiers."""
    return [
        dict(
            index=i,
            name=f"mix_{i / 10:.1f}",
            kind="classifier",
            mix=i / 10,
            n_real=None,
            augment=False,
            features="all",
        )
        for i in range(11)
    ]


def classifier_arguments(task, destination, seed, threads):
    """Main binned classifier defaults, with the full-pipeline command overrides."""
    synthetic = task["mix"] > 0 or task["augment"]
    centers = "miiv-eicu-aumc-mimic-hirid-sic"
    return SimpleNamespace(
        real_path="numerical",
        syn_path="numerical" if synthetic else None,
        label="death_icu",
        lookback=24,
        str_lookback="24",
        syn_lookback=24,
        window_size=4,
        max_length=None,
        min_length=6,
        max_gap=12,
        extracted_features="median-min-max",
        impute_missing="forwardfill",
        include_missing_mask=True,
        missingness_threshold="vitals",
        include_statics=False,
        data_size=task["features"],
        test_fraction=0.2,
        val_fraction=0.1,
        val_set=True,
        percentage_synthetic=task["mix"],
        n_real_train=task["n_real"],
        augment_with_all_syn=task["augment"],
        train_dataset=centers + ("-syn" if synthetic else ""),
        test_dataset=centers,
        test_on_syn=False,
        sources=None,
        split_mode="stay_id",
        train_ids=None,
        match_controls=False,
        FIDDLE_encode=False,
        linear=True,
        n_folds=5,
        hopt_iter=50,
        bootstraps=1000,
        seed=seed,
        num_workers=threads,
        n_augmentations=1,
        debug=False,
        output_path=str(destination),
    )


def run_classifier_task(
    data, task, destination, seed, threads, overrides, datamodule_factory=None
):
    """Call the original tuning, early-warning scoring, and bootstrap workflows."""
    from scripts import train_classifiers

    factory = datamodule_factory or partial(NumericalTaskDatamodule, data)
    _write_identities(data, destination)
    if task["kind"] == "classifier":
        args = classifier_arguments(task, destination, seed, threads)
        for name, value in overrides.items():
            if not hasattr(args, name):
                raise ValueError(f"Unknown shared classifier override: {name}")
            setattr(args, name, value)
        np.random.seed(seed)
        return train_classifiers.main(
            args, datamodule_factory=factory, max_parallel_jobs=threads
        )
    raise ValueError("The paper exposes mixture classifiers only")


def _write_identities(data, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(
            alias_id=(
                data.utility_aliases[_identity(stay)]
                if hasattr(data, "utility_aliases")
                else index
            ),
            synthetic=stay["split"] == "synthetic",
            centre=stay["centre"],
            stay_id=stay["stay_id"],
            patient_uid=stay["patient_uid"],
            generator_split=stay["split"],
        )
        for index, stay in enumerate(data.bundle["stays"] + data.synthetic)
    ]
    pd.DataFrame(rows).to_parquet(destination / "identities.parquet", index=False)
