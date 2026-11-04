"""Clinical-endpoint utility adapters; generator training and audits are untouched."""

from functools import partial

import numpy as np
import pandas as pd
import torch

from syntheticicu.tasks_numeric import NumericalTaskDatamodule, NumericalWindows


def clinical_metadata(path, hours_max):
    """Validate exact keys and ICU duration in days without an endpoint fallback."""
    frame = pd.read_parquet(
        path, columns=["centre", "stay_id", "patient_uid", "death_icu", "los_icu"]
    )
    frame["centre"] = frame.centre.str.lower()
    for name in ["stay_id", "patient_uid"]:
        if not pd.api.types.is_integer_dtype(frame[name]) or frame[name].isna().any():
            raise ValueError("Clinical metadata requires exact integer identifiers")
    if frame.centre.isna().any() or frame.duplicated(["centre", "stay_id"]).any():
        raise ValueError("Clinical metadata has missing or duplicate center/stay keys")
    hours = frame.los_icu.to_numpy(dtype=np.float64) * 24
    if not np.isfinite(hours).all() or np.any(hours <= 0):
        raise ValueError("Clinical LOS must be finite, present and positive")
    if hours_max < 24 or np.any(hours > hours_max):
        raise ValueError("Clinical LOS exceeds the configured hourly-grid bound")
    if frame.death_icu.isna().any() or not frame.death_icu.isin([0, 1]).all():
        raise ValueError("Clinical mortality labels must be complete and binary")
    frame["endpoint_hours"] = hours
    return frame.set_index(["centre", "stay_id"], verify_integrity=True)


class ClinicalWindows(NumericalWindows):
    """Real grids start at ICU admission; timestamps denote completed-hour ends."""

    reference_dtype = torch.float64

    def __init__(self, *args, endpoints, hours_max, **options):
        if options["lookback"] != 24 or options["syn_lookback"] != 0:
            raise ValueError(
                "Clinical utility requires real lookback=24 and synthetic=0"
            )
        self.endpoints, self.hours_max = endpoints, hours_max
        self.clinical_report = {
            "real_stays": 0,
            "real_excluded_minimum_duration": 0,
            "real_excluded_no_window": 0,
            "synthetic_stays": 0,
        }
        super().__init__(*args, **options)
        sources = np.asarray(self.stay_sources)
        self.clinical_report["real_windows"] = int(np.count_nonzero(sources != "syn"))
        self.clinical_report["synthetic_windows"] = int(
            np.count_nonzero(sources == "syn")
        )

    def _add_stay(self, stay, schema, min_length, metadata):
        if stay["split"] == "synthetic":
            self.clinical_report["synthetic_stays"] += 1
            return super()._add_stay(stay, schema, min_length, metadata)
        self.clinical_report["real_stays"] += 1
        key = (stay["centre"], stay["stay_id"])
        if key not in self.endpoints:
            raise ValueError(f"Missing clinical endpoint for {key}")
        reference = float(self.endpoints[key])
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("Invalid clinical endpoint")
        if reference > self.hours_max:
            raise ValueError("Clinical endpoint exceeds the configured grid bound")
        if reference < min_length:
            self.clinical_report["real_excluded_minimum_duration"] += 1
            return
        cutoff = reference - self.lookback
        kept = max(0, int(np.floor(cutoff)))
        if kept < self.window_size:
            self.clinical_report["real_excluded_no_window"] += 1
            return
        hours = stay["hours"].numpy()
        values = stay["values"].numpy()
        if hours.ndim != 1 or values.shape != (len(hours), len(schema["features"])):
            raise ValueError("Clinical hour/value axes do not align")
        if not np.isfinite(hours).all() or np.any(hours != np.floor(hours)):
            raise ValueError("Clinical source hours must be finite integers")
        if np.any(hours < 0) or np.any(np.diff(hours) <= 0):
            raise ValueError("Clinical source hours must be nonnegative and increasing")
        columns = [list(schema["features"]).index(name) for name in self.features]
        raw = np.full((kept, len(columns)), np.nan, dtype=np.float32)
        selected = hours < kept
        raw[hours[selected].astype(np.int64)] = values[selected][:, columns]
        self._append_values(stay, schema, raw, metadata, reference, 1)
        assert len(raw) <= self.hours_max
        assert kept <= cutoff


class ClinicalTaskDatamodule(NumericalTaskDatamodule):
    """Keep active task sampling; apply zero synthetic lookback in every split."""

    def __init__(self, data, endpoints, hours_max, **options):
        super().__init__(data, **options)
        self.window_class = partial(
            ClinicalWindows, endpoints=endpoints, hours_max=hours_max
        )

    def _apply_lookback_per_split(self):
        dataset = self.full_dataset
        gap = (dataset.reference_hours - dataset.times).numpy()
        synthetic = np.asarray(dataset.stay_sources) == "syn"
        if np.any(gap[~synthetic] < np.nextafter(24.0, -np.inf)):
            raise ValueError("Real window violates the clinical 24-hour gap")
        for partition in ["train", "val", "test"]:
            indices = getattr(self, f"{partition}_indices")
            if indices is not None:
                indices = np.asarray(indices, dtype=np.int64)
                if np.any(gap[indices][synthetic[indices]] < 0):
                    raise ValueError(
                        "Synthetic window extends beyond its generated stay"
                    )
                setattr(self, f"{partition}_ids", dataset.ids.numpy()[indices])
