"""Export uncut sequence quality and the exact source-cohort reference."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import hydra
import joblib
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch

from scripts.summarize_tokenization_corpora import summarize_corpus
from scripts.prepare_clinical_figure import real_histograms
from syntheticicu.privacy_bridge_numeric import export_audit_corpus


def observation_signature(source):
    """Compare observed feature/hour masks across Q without exporting identities."""
    data = joblib.load(source)
    digest = hashlib.sha256()
    order = sorted(
        range(len(data.y)), key=lambda i: (str(data.stay_sources[i]), int(data.y[i, 0]))
    )
    for index in order:
        digest.update(f"{data.stay_sources[index]}:{int(data.y[index,0])}".encode())
        times = np.asarray(data.times[index])
        times = times[: int(np.argmax(times)) + 1]
        mask = np.asarray(data.X_value_ids[index])[: len(times)] < data.num_bins
        digest.update(times.astype(np.int64).tobytes())
        digest.update(np.packbits(mask).tobytes())
    digest.update(json.dumps(list(data.features)).encode())
    return digest.hexdigest()


@hydra.main(version_base="1.3", config_path="../config", config_name="paper_quality")
def main(cfg: DictConfig):
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in cfg.corpora:
        target = root / f"Q{item.q}" / item.mode
        target.mkdir(parents=True, exist_ok=True)
        source = Path(item.path)
        if item.kind == "numerical":
            schema = torch.load(
                source / "model.pt", map_location="cpu", weights_only=True
            )["schema"]
            source = export_audit_corpus(
                source, schema, target / "generated_tokens.joblib"
            )
        summary = summarize_corpus(source, item.representation, item.mode, target)
        if summary["sequences"] != cfg.release_size or not np.isclose(
            summary["outcome_prevalence"], 0.07, atol=1e-15
        ):
            raise ValueError("Paper release must contain 12000 stays and 840 positives")
        rows.append(dict(q=item.q, **summary))
    pd.DataFrame(rows).to_csv(root / "corpus_summary.csv", index=False)
    signatures = {}
    for item in cfg.real_sources:
        signatures[str(item.q)] = observation_signature(item.path)
        settings = SimpleNamespace(
            real_dataset_path=item.path,
            real_metadata_path=cfg.metadata_path,
            real_hours_max=9000,
            modes=["null", "conditioned"],
        )
        target = root / f"Q{item.q}"
        target.mkdir(parents=True, exist_ok=True)
        real_histograms(settings, target)
    (root / "real_observation_masks.json").write_text(
        json.dumps(
            dict(
                signatures=signatures,
                identical=bool(signatures) and len(set(signatures.values())) == 1,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
