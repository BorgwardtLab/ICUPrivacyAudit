# Reproducing the paper

## Inputs

1. `data/core.parquet`: harmonized hourly measurements, exact integer `stay_id`, `centre`, and `stay_time` in hours. Feature columns follow `config/numerical_full.yaml` (`data.features`).
2. `data/metadata.parquet`: exact center/stay keys, mortality, demographics and positive finite `los_icu` in **days**, including SIC. HiRID's last-observation LOS is an accepted proxy, not an independently verified discharge timestamp.
3. Direct FIDDLE datasets at Q=2/5/15/25/50. Preserve exact integer source IDs, feature/hour alignment, bin edges and missing-token conventions. The original pickle class aliases are registered by the loaders. Only load trusted pickle/joblib inputs.
4. Recorded discrete `hparams.yaml` and checkpoints, with local paths updated to these inputs. Recover the generator stay assignments with `poetry run python scripts/prepare_paper_inputs.py hparams=inputs/hparams.yaml dataset_path=/your/dataset.pkl`. The private `inputs/generator_splits.parquet` has `centre`, `stay_id`, `split` (`train`, `val`, `test`). This replays saved stay assignments, not a new patient-disjoint split.
5. For continuous utility replay, retain the discrete clinical utility results and recorded selection metadata. Point `reference_utility_root` to the discrete shuffled run. Supply the raw core/metadata tables through `recovery_core_path` and `metadata_path` so evaluation-only stays omitted by the generator filter can be restored. Use the same clinical-endpoint definition for each representation.

Run data loading and all subsequent commands in suitable CPU/GPU allocations. No real datasets or results are downloaded automatically.

## Training and generation

Discrete workflows use `poetry run python scripts/train.py train` and `poetry run python scripts/generate.py generate`. Override `datamodule.data_path` through the local paths configuration and set `datamodule.shuffle_tokens_within_timepoint=false` for fixed order. Preserve the recorded run's scientific settings when reproducing a specific checkpoint.

The continuous workflow is staged:

```bash
poetry run python scripts/numerical_workflow.py --config-name numerical_matched action=prepare
poetry run python scripts/train_matched_multinode.py
poetry run python scripts/generate_numerical_full.py --config-name numerical_matched
poetry run python scripts/generate_numerical_conditioned.py --config-name numerical_matched
```

Hydra-decorated entry points accept `--cfg job`; the original discrete entry points instead take the configuration name as their first positional argument. Check generation phase controls before submitting. The default continuous production configuration requests eight GPUs, batch 32/GPU and accumulation two (effective batch 512). Keep empty-boundary/trail trimming enabled and use the exact reference split manifest. Generation is outcome-conditioned at 7%; center-conditioned and null-center corpora each contain 12,000 stays. Conditional demographic generation and chunk continuation remain part of the workflow.

## Clinical utility

`clinical_pipeline_tasks.py` runs the shared mixture matrix with `task_index=0` through `10`, fractions 0.0 to 1.0. Select a source using `clinical_source@source=discrete_shuffled`, `discrete_nonshuffled`, or `discrete_bins` with the appropriate Q. Quote the null mode as a string in Hydra overrides: `generation_mode='"null"'`. Use `generation_mode=conditioned` for the other release.

For matched continuous utility, use `run_native_continuous_utility.py` with `clinical_native_replay`. Real trajectories use a clinical endpoint minus 24 hours; synthetic trajectories remain uncut. Retain the discrete sampling population, classifier folds and scoring aliases. Export uncertainty by resampling complete test stays (default 2,000 replicates).

## Privacy and power

For the discrete audit use `run_privacy_audit_ext.py generate model.pretrained_path=YOUR_CHECKPOINT`; reduced-regime runs use `run_privacy_audit_ext.py testbed --testbed` or `testbed_multicentre --testbed`. The numerical audit uses `numerical_pipeline_tasks.py` with `privacy.score_kind=categorical` or `numerical`. Keep actual generator members/nonmembers, independent of utility selections. Scores are aggregated at the patient-key level used by the audit. The numerical attack scores value log-density; structural tokens contribute no numerical measurement likelihood.

Fixed-order and continuous canary stages use `run_canary_curves.py action=prepare/train/score/collect` (continuous preparation is `action=continuous_prepare`), `representation=discrete/continuous`, and factors 5, 20, 50. Preparation freezes 500 exact shared keys; each training stage duplicates all their chunks. For shuffled reference retraining, `run_testbed_full.py train --hparams YOUR_RECORDED_HPARAMS --out YOUR_OUTPUT --dup 20 --n-canaries 500` and its `score` subcommand preserve the recorded shuffle setting. Do not interpret repetition as proof that every canary was memorized. The reference configuration, training budget and score determine the measured sensitivity.

`value_corpus_audit.py` exposes separate `action` stages in `config/value_corpus_audit.yaml`: real/generated preparation, membership, replication, floor and collectors. Only DTW-L1 is retained. Candidate pools remain capped at 140 per cell, so this is not an exhaustive release search. Recovery uses 200 shared stays and nested deletion fractions 0/10/20/30/40/50/60/70/80%; strict ties fail. Synthetic-to-training versus synthetic-to-holdout preference counts ties as one half. Discrete corpus attacks use canonical edit distance on initial chunks; their historical recovery extension checks the saved 40% point before exporting new fractions.

## Export manifests and figures

Extend `config/paper_results.yaml` with lists of saved artifacts. Each entry has a unique `name` and a local `path`; add `q`, `mode`, `representation` and `fraction` where relevant. Use `discrete_shuffled`, `discrete_fixed`, and `continuous` as Figure 2–4 representation labels. Keep names unique across Q and mode, for example `discrete_shuffled_Q15` and `continuous_null`.

- `utility`: `results.joblib`, plus representation, Q, mode and fraction.
- `discrete_audits`: a saved `privacy_audit_results.joblib`, plus Q and representation.
- `numerical_audits`: numerical membership audit result or membership-stage joblib payload, plus Q and representation.
- `canaries`: measured `curve.csv`; a sibling `curve_rocs.npz` is copied if present.
- `replication`: genuine reverse-direction distance joblib, plus representation and mode.
- `value_corpus`: completed audit directory containing membership metrics/ROCs and recovery CSV, plus representation and mode.
- `recovery_extensions`: verified historical `canary_fractions.joblib`.

Run `poetry run python scripts/export_paper_results.py --config-name YOUR_MANIFEST`. The exporter checks saved AUCs against scores and utility curves against grouped predictions. It writes aggregate metrics, anonymous test prediction pairs, ROC arrays, availability/provenance and checksums. Missing artifacts are pending, never filled from another model. Empty manifests intentionally contain no results.

For quality, extend `paper_quality` with `corpora` entries `{representation, q, mode, kind, path}` (`kind: discrete` for a joblib corpus, `numerical` for a continuous model output root) and `real_sources` entries `{q, path}`. These use the **same uncut release** as utility. Observation-mask hashes verify whether a real tokens/hour reference can be shared across Q; otherwise use the per-Q reference.

Run `scripts/summarize_paper_quality.py` with that manifest, then `scripts/plot_paper_figures.py` with `results_root` and `quality_root` pointing to the exports. Figures are saved as PDF/PNG/SVG. Missing measured canary exports remain labeled pending. Historical publication styling and manually composed schematics require the authors' original figure sources.

## Scope and provenance

DTW-L1 is checked against twelve artificial reference cases. The scientific scaling, missingness and time terms are tested without using patient observations.

Per-token bin information bounds do not imply trajectory anonymity or a formal membership guarantee. The exact split assignments and generator architectures must be recorded for each comparison. Privacy confidence intervals are read from the saved measured audits; utility uncertainty defaults to 2,000 stay-bootstrap replicates.

## Source-only distribution

Use `scripts/export_public_source.py` after committing reviewed changes. It exports the current committed tree into `outputs/public_source/`, validates its file types and checks for obvious credentials, then writes a per-file SHA256 inventory. It does not copy Git history or local artifacts. Source review remains necessary for text contents: a file-extension check alone cannot establish that a Python literal contains no private observation. Both included numerical fixtures are generated independently of patient records.
