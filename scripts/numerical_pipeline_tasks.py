"""Audit released continuous weights with raw and conditioning-swap scores."""

import json
from pathlib import Path

import hydra
import joblib
from omegaconf import DictConfig, OmegaConf

from syntheticicu.evaluation_numeric import load_evaluation_data
from syntheticicu.privacy_bridge_numeric import NumericalAuditModel
from syntheticicu.privacy_audit_ext import (
    StaticsDecoder,
    run_calibrated_membership_audit,
)
from syntheticicu.workflow_numeric import load_numerical_model


@hydra.main(
    version_base="1.3", config_path="../config", config_name="numerical_pipeline"
)
def main(cfg: DictConfig):
    """Score actual generator members; utility selections never define membership."""
    repository = Path(hydra.utils.get_original_cwd()).resolve()
    root = Path(cfg.output_dir).resolve()
    destination = (
        Path(cfg.evaluation_dir).resolve() / f"likelihood_{cfg.privacy.score_kind}"
    )
    if not root.is_relative_to(repository) or not destination.is_relative_to(
        repository
    ):
        raise ValueError("Numerical audit paths must remain inside the repository")
    marker = destination / "complete.json"
    if marker.exists() and not cfg.overwrite:
        raise FileExistsError(
            "Completed audit exists; choose another evaluation_dir or overwrite=true"
        )
    data = load_evaluation_data(root, require_release=False)
    core, checkpoint = load_numerical_model(root / "model.pt", cfg.privacy.device)
    if checkpoint["data_signature"] != data.bundle["signature"]:
        raise ValueError("Likelihood model and prepared population do not match")
    model = NumericalAuditModel(core, data, score_kind=cfg.privacy.score_kind)
    decoder = StaticsDecoder(model=model)
    result = {
        "membership": run_calibrated_membership_audit(model, decoder=decoder),
        "no_demographics": run_calibrated_membership_audit(
            model, decoder=decoder, exclude_static_tokens=True
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "results.joblib.partial"
    joblib.dump(result, temporary)
    temporary.replace(destination / "results.joblib")
    report = dict(
        configuration=OmegaConf.to_container(cfg, resolve=True),
        data_signature=data.bundle["signature"],
        reference="actual generator training and held-out membership",
    )
    temporary = marker.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(marker)
    print(f"Completed weight audit: {marker}", flush=True)


if __name__ == "__main__":
    main()
