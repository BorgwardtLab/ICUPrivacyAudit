"""Replay the completed native clinical discrete evaluation on numerical inputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from syntheticicu.clinical_utility import ClinicalTaskDatamodule
from syntheticicu.tasks_numeric import _identity


def verify_windows(reference, actual, reference_indices, actual_indices, context):
    """Require identical real window IDs, times, endpoints and labels before fitting."""

    def table(dataset, indices):
        indices = np.asarray(indices, dtype=np.int64)
        frame = pd.DataFrame(
            {
                "ids": dataset.ids.numpy()[indices],
                "times": dataset.times.numpy()[indices],
                "reference": dataset.reference_hours.numpy()[indices],
                "label": dataset.y.numpy()[indices],
                "onset": dataset.onset.numpy()[indices],
            }
        )
        return frame.sort_values(["ids", "times"]).reset_index(drop=True)

    expected, observed = table(reference, reference_indices), table(
        actual, actual_indices
    )
    try:
        pd.testing.assert_frame_equal(
            expected, observed, check_dtype=False, check_exact=True
        )
    except AssertionError as error:
        raise ValueError(
            f"Native clinical {context} windows differ: {error}"
        ) from error
    return len(expected)


class NativeUtilityReplay(ClinicalTaskDatamodule):
    """Freeze real assignments using the native discrete clinical task sampler."""

    def __init__(self, data, reference_data, endpoints, hours_max, reports, **options):
        self.reference_data = reference_data
        self.reference_endpoints = endpoints
        self.reference_hours_max = hours_max
        self.reference_reports = reports
        self.native_options = dict(options)
        aliases = {
            _identity(stay): index
            for index, stay in enumerate(
                reference_data.bundle["stays"] + reference_data.synthetic
            )
        }
        reference_data.utility_aliases = aliases
        data.utility_aliases = {}
        real_count = len(reference_data.bundle["stays"])
        for stay in data.bundle["stays"]:
            identity = _identity(stay)
            if identity not in aliases:
                raise ValueError("Numerical stay absent from native discrete source")
            data.utility_aliases[identity] = aliases[identity]
        for index, stay in enumerate(data.synthetic):
            data.utility_aliases[_identity(stay)] = real_count + index
        super().__init__(data, endpoints, hours_max, **options)

    def split(self):
        reference = ClinicalTaskDatamodule(
            self.reference_data,
            self.reference_endpoints,
            self.reference_hours_max,
            **self.native_options,
        )
        state = np.random.get_state()
        try:
            reference.setup()
        finally:
            np.random.set_state(state)
        # Synthetic releases contain different patients: preserve native numerical sampling.
        super().split()
        real = np.asarray(self.full_dataset.stay_sources) != "syn"
        reference_real = np.asarray(reference.full_dataset.stay_sources) != "syn"
        report = {}
        for partition in ["train", "val", "test"]:
            selected = getattr(reference, f"{partition}_indices")
            selected = np.asarray([] if selected is None else selected, dtype=np.int64)
            expected = selected[reference_real[selected]]
            aliases = np.unique(reference.full_dataset.ids.numpy()[expected])
            missing = set(aliases) - set(self.full_dataset.ids.numpy())
            if missing:
                lookup = {
                    value: key
                    for key, value in self.reference_data.utility_aliases.items()
                }
                destination = Path("outputs/clinical_utility_native_replay/diagnostics")
                destination.mkdir(parents=True, exist_ok=True)
                (destination / f"{partition}_missing_stays.json").write_text(
                    json.dumps(
                        [
                            dict(alias=int(alias), identity=lookup[alias])
                            for alias in sorted(missing)
                        ],
                        indent=2,
                    )
                )
            matching = np.flatnonzero(
                real & np.isin(self.full_dataset.ids.numpy(), aliases)
            )
            report[partition] = dict(
                real_stays=len(aliases),
                real_windows=verify_windows(
                    reference.full_dataset,
                    self.full_dataset,
                    expected,
                    matching,
                    partition,
                ),
            )
            native = getattr(self, f"{partition}_indices")
            native = np.asarray([] if native is None else native, dtype=np.int64)
            synthetic = native[~real[native]]
            indices = np.sort(np.r_[matching, synthetic])
            setattr(self, f"{partition}_indices", indices)
            setattr(self, f"{partition}_ids", self.full_dataset.ids.numpy()[indices])
        self.reference_reports.append(report)
        self.native_reference_module = reference
        return self.train_indices, self.val_indices, self.test_indices
