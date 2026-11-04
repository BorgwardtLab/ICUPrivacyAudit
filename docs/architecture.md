# Paper workflow architecture

`data_classes.py` and the local FIDDLE implementation load direct discrete datasets. `model_gap.py`, `base_model.py` and `model_emb.py` implement the discrete generator. `demo_utils.py` supports tabular demographic prompts. `data_numeric.py`, `model_numeric.py`, `workflow_numeric.py`, `train_multinode_numeric.py` and the numerical generation modules implement the continuous branch. Both share the underlying clinical variable definitions.

`clinical_sources.py` adapts each representation into the common hourly schema. `clinical_utility.py` applies clinical endpoints and four-hour windows. `tasks_numeric.py` defines the eleven mixture tasks and invokes the shared logistic classifier code in `scripts/train_classifiers.py`. `native_utility_replay.py` and `native_utility_source.py` replay discrete utility selections for continuous inputs; they validate historical references instead of changing privacy membership.

`privacy_audit.py` and `privacy_audit_ext.py` implement categorical likelihood scores, swaps, discrete edit distance and calibration. `privacy_bridge_numeric.py`, `audit_numeric.py` connect continuous scoring; `privacy_stage_cache.py` supports resumable discrete audit stages. `value_distances.py` and `value_corpus.py` implement the reported full-trajectory DTW-L1 audit. Canary runners retrain independent models with duplicated training records; corpus recovery plants copies without retraining.

The `paper_*` modules and scripts provide a small artificial example, checked saved-result exports and figure reconstruction. They do not fit replacement models when an artifact is missing. `quack/` contains deterministic CPU tests and artificial fixtures. GPU-dependent checks skip when hardware is unavailable.

`config/` contains portable Hydra defaults. `scripts/` contains scientific entry points; `experiments/` contains allocation wrappers. `inputs/`, `data/`, `outputs/`, environments and checkpoints are private local artifacts excluded from git. No Slurm job IDs or production cluster recovery graph is part of the reproduction interface.
