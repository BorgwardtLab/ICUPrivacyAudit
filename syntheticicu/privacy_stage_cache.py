"""Atomic checkpoints for expensive audit stages, including RNG continuation."""

import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch


class AuditStageCache:
    """Reuse stages only within an identical, explicitly bound audit contract."""

    def __init__(self, directory, contract):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract = json.loads(json.dumps(contract, sort_keys=True))
        path = self.directory / "contract.json"
        if path.exists():
            if json.loads(path.read_text()) != self.contract:
                raise ValueError(
                    "Audit cache contract differs; use a new evaluation_dir"
                )
        else:
            temporary = path.with_suffix(".json.partial")
            temporary.write_text(json.dumps(self.contract, indent=2) + "\n")
            temporary.replace(path)

    def run(self, name, function, *args, **kwargs):
        """Replay results and post-stage RNG state, or atomically save a fresh stage."""
        if self is None:
            return function(*args, **kwargs)
        if not name or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in name
        ):
            raise ValueError("Stage names must be lowercase identifiers")
        path = self.directory / f"{name}.joblib"
        if path.exists():
            saved = joblib.load(path)
            if saved["contract"] != self.contract:
                raise ValueError("Saved stage contract differs from the audit")
            random.setstate(saved["random"])
            np.random.set_state(saved["numpy"])
            torch.set_rng_state(saved["torch"])
            if saved["cuda"]:
                torch.cuda.set_rng_state_all(saved["cuda"])
            print(f"Reusing completed audit stage: {name}", flush=True)
            return saved["result"]
        result = function(*args, **kwargs)
        saved = {
            "contract": self.contract,
            "result": result,
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
            ),
        }
        temporary = path.with_suffix(".joblib.partial")
        joblib.dump(saved, temporary)
        temporary.replace(path)
        print(f"Saved completed audit stage: {name}", flush=True)
        return result
