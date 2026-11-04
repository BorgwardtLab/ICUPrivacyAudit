# SyntheticICUAugmentation — RECOMB27

Code for **Privacy auditing with measured detection floors for generative models of ICU trajectories** submitted for review at RECOMB 2027.

This repository contains the training, generation and evaluation workflows used for the paper: quantile-token and continuous time series generators, clinical mortality prediction, and privacy audits of released weights and synthetic cohorts.

![Figure 1](docs/figures/figure_1.pdf)

Figure 1. Overview of the framework: ICU trajectory tokenization, generator training, and privacy auditing of released model weights and synthetic cohorts, with measured detection floors for both release
scenarios.

## Install and try the artificial example

Use Python 3.11 or 3.12 and Poetry 2.x:

```bash
poetry install --with dev
poetry run python scripts/paper_example.py
poetry run python scripts/check_numerical_model.py quack
```

The example presented here relies entirely on a small artificial dataset, trains for two CPU updates, generates eight stays, and checks the clinical window cutoff and literal-copy DTW distance. Outputs go to `outputs/example/`. Reproducing the exact numbers presented in the manuscript requires access to all six datasets we used.

## Reproduce the analyses

The workflows use Hydra: append `key=value` overrides, or provide a local configuration extending the committed defaults. Run commands from the repository root. The [reproduction guide](docs/reproduction.md) describes inputs, stages and manifest entries; [architecture](docs/architecture.md) describes their dependencies.

| Analysis | Entry point | Configuration |
|---|---|---|
| Discrete training and generation | `scripts/train.py`, `scripts/generate.py` | `train`, `generate` |
| Continuous preparation, training and generation | `scripts/numerical_workflow.py`, `scripts/train_matched_multinode.py` | `numerical_matched` |
| Shared clinical utility | `scripts/clinical_pipeline_tasks.py` | `clinical_pipeline` |
| Continuous replay of the discrete utility cohort | `scripts/run_native_continuous_utility.py` | `clinical_native_replay` |
| Released-weight audit and swaps | `scripts/run_privacy_audit_ext.py`, `scripts/numerical_pipeline_tasks.py` | audit CLI, `numerical_pipeline` |
| Full-scale canary curves, factors 5/20/50 | `scripts/run_canary_curves.py` | `canary_curves` |
| Reduced training-regime testbeds | `scripts/run_privacy_audit_ext.py` | `testbed`, `testbed_multicentre` |
| Discrete corpus replication | `scripts/discrete_replication_audit.py` | `discrete_replication` |
| Continuous corpus membership, replication and recovery | `scripts/value_corpus_audit.py` | `value_corpus_audit` |
| Historical discrete recovery extensions | `scripts/run_audit_followups.py` | CLI plus audit Hydra overrides |
| Cohort summary table | `scripts/summarize_paper_cohort.py` | `paper_cohort` |
| Uncut quality summaries | `scripts/summarize_paper_quality.py` | `paper_quality` |
| Checked, identifier-free result exports | `scripts/export_paper_results.py` | `paper_results` |
| Figures 2–4 and S1–S4 | `scripts/plot_paper_figures.py` | `paper_figures` |

The plotting entry point reconstructs all figure panels individually from measured exports. Configuration manifests are empty by default: populate them with your own artifact paths before exporting or plotting.

## Data and interpretation

Clinical inputs require the original databases' access permissions and are not distributed. Supply the harmonized physiology/metadata Parquet tables, direct FIDDLE datasets at Q=2/5/15/25/50, saved discrete training configurations/checkpoints, and exact generator stay assignments. See [input contracts](docs/reproduction.md#inputs). Never commit patient identifiers, real trajectories or model artifacts.

## Publish a source-only copy

After committing the reviewed source, run:

```bash
poetry run python scripts/export_public_source.py
```

This writes `outputs/public_source/RECOMB27-source.tar.gz`, its SHA256 checksum and an inventory inside the archive. It includes committed source files and the two artificial fixtures only. It excludes the working directory's ignored/untracked contents and all Git history. Extract it into an empty directory and initialize a new repository there; do not recursively copy this checkout or push its existing history to the public repository.

## License

This code is licensed under the [BSD 3-Clause License](LICENSE).
