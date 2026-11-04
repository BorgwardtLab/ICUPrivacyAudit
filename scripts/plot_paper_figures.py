"""Render paper panels from verified exports; missing evidence stays labeled."""

import ast
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd


def save(figure, root, name):
    root.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png", "svg"]:
        figure.savefig(root / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def interval(row, field="auc"):
    key = f"{field}_ci"
    if key not in row or pd.isna(row[key]) or row[key] == "":
        return None
    value = ast.literal_eval(str(row[key]))
    return np.asarray([[row[field] - value[0]], [value[1] - row[field]]])


def recovery_table(path):
    frame = pd.read_csv(path, keep_default_na=False)
    if "canonicalize" in frame:
        frame = frame.loc[frame.canonicalize.astype(str).str.lower() == "true"].copy()
    if "variant" in frame:
        frame = frame.loc[
            (frame.variant == "literal") | frame.variant.str.startswith("dropped@")
        ].copy()
        frame["fraction"] = (
            frame.variant.str.extract(r"(\d+)")[0].fillna(0).astype(float) / 100
        )
    return frame.sort_values("fraction")


def utility_figures(cfg, table, root):
    for mode, figure_name in [("null", "figure2"), ("conditioned", "figureS1")]:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for representation in cfg.representations:
            quality = Path(cfg.quality_root) / "Q5" / mode
            source = quality / f"{representation}_{mode}_sequences.csv"
            if source.exists():
                rows = pd.read_csv(source)
                axes[0, 0].hist(
                    rows.hours,
                    bins=np.geomspace(1, 9000, 35),
                    density=True,
                    histtype="step",
                    label=representation,
                )
                counts = pd.read_csv(
                    quality / f"{representation}_{mode}_tokens_per_hour.csv"
                )
                axes[0, 1].plot(
                    counts.tokens_per_hour,
                    counts.hours / counts.hours.sum(),
                    label=representation,
                )
            if not {"representation", "mode", "fraction"}.issubset(table):
                continue
            selected = table.loc[
                (table.section == "utility")
                & (table.representation == representation)
                & (table["mode"] == mode)
            ]
            if "q" in selected:
                selected = selected.loc[selected.q == 5]
            selected = selected.sort_values("fraction")
            for axis, metric in [
                (axes[1, 0], "auroc"),
                (axes[1, 1], "average_precision"),
            ]:
                axis.plot(
                    selected.fraction,
                    selected[metric],
                    marker="o",
                    label=representation,
                )
                for _, row in selected.iterrows():
                    bounds = interval(row, metric)
                    if bounds is not None:
                        axis.errorbar(
                            row.fraction,
                            row[metric],
                            yerr=bounds,
                            fmt="none",
                            color="gray",
                        )
        real = Path(cfg.quality_root) / "Q5"
        if (real / "real_duration_histogram.csv").exists():
            values = pd.read_csv(real / "real_duration_histogram.csv")
            widths = values.right_hours - values.left_hours
            axes[0, 0].bar(
                values.left_hours,
                values.stays / (values.stays.sum() * widths),
                width=widths,
                align="edge",
                color="gray",
                alpha=0.3,
                edgecolor="none",
                label="Real cohort",
            )
            counts = pd.read_csv(real / "real_tokens_per_hour.csv")
            axes[0, 1].bar(
                counts.tokens_per_hour,
                counts.hours / counts.hours.sum(),
                color="gray",
                alpha=0.3,
                edgecolor="none",
            )
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlabel("Hours")
        axes[0, 1].set_xlabel("Physiology tokens/hour")
        axes[1, 0].set_ylim(0.5, 1)
        axes[1, 1].set_ylim(0.07, 1)
        for axis, title in zip(
            axes.flat,
            [
                "A. Stay duration",
                "B. Hourly measurement density",
                "C. Mortality AUROC",
                "D. Mortality average precision",
            ],
        ):
            axis.set_title(title)
            if axis.lines or axis.patches:
                axis.legend(fontsize=7)
        save(figure, root, figure_name)


def privacy_figures(cfg, table, root, corpus=False):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    source = Path(cfg.results_root)
    section = "corpus" if corpus else "weights"
    for index, representation in enumerate(cfg.representations):
        if corpus:
            rows = table.loc[
                (table.section == "corpus") & (table.representation == representation)
            ]
        else:
            rows = table.loc[
                table.section.isin(["weights", "numerical_audits"])
                & (table.representation == representation)
            ]
        if "q" in rows:
            rows = rows.loc[rows.q == 5]
        for _, row in rows.iterrows():
            statistic = row.get("statistic", "raw")
            if not corpus and statistic != cfg.primary_statistics[index]:
                continue
            axes[0, 0].errorbar(
                row.auc,
                index,
                xerr=interval(row),
                marker="o",
                label=f'{representation} {row.get("mode",statistic)}',
            )
            mode = row.get("mode", "null")
            if corpus:
                path = (
                    source / "value_corpus" / row["name"] / "C.npz"
                    if representation == "continuous"
                    else source / "discrete_audits" / row["name"] / f"C_{mode}.npz"
                )
            else:
                path = (
                    source
                    / (
                        "numerical_audits"
                        if representation == "continuous"
                        else "discrete_audits"
                    )
                    / row["name"]
                    / (
                        f"{statistic}.npz"
                        if representation == "continuous"
                        else f"A_{statistic}.npz"
                    )
                )
            if path.exists():
                values = np.load(path, allow_pickle=False)
                mask = values["fpr"] > 0
                axes[0, 1].semilogx(
                    values["fpr"][mask],
                    values["tpr"][mask] / values["fpr"][mask],
                    label=representation,
                )
        if not corpus and not rows.empty:
            axes[1, 0].plot(rows.statistic, rows.auc, marker="o", label=representation)
    if corpus:
        for path in source.rglob("recovery.csv"):
            frame = recovery_table(path)
            axes[1, 0].plot(
                frame.fraction * 100,
                frame.rank1_rate,
                marker="o",
                label=path.parent.name,
            )
        for _, row in table.loc[table.section == "replication"].iterrows():
            axes[1, 1].errorbar(
                row.closer_to_training,
                len(axes[1, 1].lines),
                xerr=[
                    [row.closer_to_training - row.ci_low],
                    [row.ci_high - row.closer_to_training],
                ],
                fmt="o",
                label=row["name"],
            )
        axes[1, 1].axvline(0.5, color="gray", linestyle=":")
    else:
        for path in (source / "canaries").glob("*/curve.csv"):
            frame = pd.read_csv(path, keep_default_na=False)
            frame = frame.loc[
                (frame.contrast == "canary_vs_holdout") & (frame.statistic == "raw")
            ]
            axes[1, 1].plot(frame.factor, frame.auc, marker="o", label=path.parent.name)
        if not axes[1, 1].lines:
            axes[1, 1].text(0.5, 0.5, "Measured canary exports pending", ha="center")
    axes[0, 0].axvline(0.5, color="gray", linestyle=":")
    axes[0, 0].set_yticks(range(len(cfg.representations)), cfg.representations)
    axes[0, 1].axhline(1, color="gray", linestyle=":")
    titles = (
        [
            "A. Corpus membership",
            "B. True-positive lift",
            "C. Corruption recovery",
            "D. Replication preference",
        ]
        if corpus
        else [
            "A. Weight membership",
            "B. True-positive lift",
            "C. Conditioning swaps",
            "D. Canary sensitivity",
        ]
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        if axis.lines:
            axis.legend(fontsize=6)
    save(figure, root, "figure4" if corpus else "figure3")


def sweep_figures(cfg, table, root):
    """Render resolution analyses from measured per-Q exports only."""
    source = Path(cfg.results_root)
    resolutions = [2, 5, 15, 25, 50]
    figure, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for column, q in enumerate(resolutions):
        quality = Path(cfg.quality_root) / f"Q{q}"
        for mode in ["null", "conditioned"]:
            path = quality / mode / f"discrete_shuffled_{mode}_sequences.csv"
            if not path.exists():
                continue
            sequences = pd.read_csv(path)
            axes[0, column].hist(
                sequences.hours,
                bins=np.geomspace(1, 9000, 35),
                density=True,
                histtype="step",
                label=mode,
            )
            counts = pd.read_csv(
                quality / mode / f"discrete_shuffled_{mode}_tokens_per_hour.csv"
            )
            axes[1, column].plot(
                counts.tokens_per_hour, counts.hours / counts.hours.sum(), label=mode
            )
        real = quality / "real_duration_histogram.csv"
        if real.exists():
            values = pd.read_csv(real)
            width = values.right_hours - values.left_hours
            axes[0, column].bar(
                values.left_hours,
                values.stays / (values.stays.sum() * width),
                width=width,
                align="edge",
                color="gray",
                alpha=0.3,
                edgecolor="none",
            )
            counts = pd.read_csv(quality / "real_tokens_per_hour.csv")
            axes[1, column].bar(
                counts.tokens_per_hour,
                counts.hours / counts.hours.sum(),
                color="gray",
                alpha=0.3,
                edgecolor="none",
            )
        axes[0, column].set(title=f"Q={q}", xscale="log", xlabel="Hours")
        axes[1, column].set_xlabel("Physiology tokens/hour")
    save(figure, root, "figureS2")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    if {"q", "fraction", "mode"}.issubset(table):
        selected = table.loc[(table.section == "utility") & table.fraction.isin([0, 1])]
        for (mode, fraction), frame in selected.groupby(["mode", "fraction"]):
            frame = frame.sort_values("q")
            for axis, metric in zip(axes, ["auroc", "average_precision"]):
                axis.plot(
                    frame.q,
                    frame[metric],
                    marker="o",
                    label=f"{mode}, synthetic={fraction}",
                )
                axis.set(xlabel="Quantile bins", ylabel=metric)
                axis.legend(fontsize=7)
    axes[0].set_ylim(0.5, 1)
    axes[1].set_ylim(0.07, 1)
    save(figure, root, "figureS3")
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    if "q" in table:
        for section in ["weights", "corpus"]:
            frame = table.loc[
                (table.section == section)
                & (table.representation == "discrete_shuffled")
            ]
            if section == "weights":
                frame = frame.loc[frame.statistic == "raw"]
            for mode, rows in frame.groupby("mode", dropna=False):
                rows = rows.sort_values("q")
                axes[0, 0].plot(rows.q, rows.auc, marker="o", label=f"{section} {mode}")
    for path in (source / "discrete_audits").glob("*/demographic_group_sizes.csv"):
        name = path.parent.name
        sizes = pd.read_csv(path).members_sharing_profile
        axes[0, 1].plot(
            np.sort(sizes), np.arange(1, len(sizes) + 1) / len(sizes), label=name
        )
        rarity = pd.read_csv(path.with_name("rarity.csv"))
        for attack, axis in [("A", axes[1, 0]), ("C", axes[1, 1])]:
            for mode, rows in rarity.loc[rarity.attack == attack].groupby("mode"):
                axis.plot(
                    rows.lower.clip(lower=1),
                    rows.auc,
                    marker="o",
                    label=f"{name} {mode}",
                )
                axis.set_xscale("log")
    for path in source.rglob("recovery.csv"):
        frame = recovery_table(path)
        axes[0, 2].plot(
            frame.fraction, frame.rank1_rate, marker="o", label=path.parent.name
        )
        raw = pd.read_csv(path, keep_default_na=False)
        if {"variant", "canonicalize"}.issubset(raw):
            shuffled = raw.loc[raw.variant == "shuffled"]
            axes[1, 2].plot(
                shuffled.canonicalize.astype(str),
                shuffled.rank1_rate,
                marker="o",
                label=path.parent.name,
            )
    axes[0, 1].set_xscale("log")
    for axis, title in zip(
        axes.flat,
        [
            "Global membership AUC",
            "Demographic group sizes",
            "Corruption recovery",
            "Weight AUC by rarity",
            "Corpus AUC by rarity",
            "Canonicalization",
        ],
    ):
        axis.set_title(title)
        if axis.lines:
            axis.legend(fontsize=5)
    save(figure, root, "figureS4")


@hydra.main(version_base="1.3", config_path="../config", config_name="paper_figures")
def main(cfg: DictConfig):
    table = pd.read_csv(Path(cfg.results_root) / "metrics.csv", keep_default_na=False)
    for column in [
        "q",
        "fraction",
        "auc",
        "auroc",
        "average_precision",
        "n_members",
        "n_nonmembers",
        "closer_to_training",
        "ci_low",
        "ci_high",
    ]:
        if column in table:
            table[column] = pd.to_numeric(
                table[column].replace("", np.nan), errors="raise"
            )
    root = Path(cfg.output_dir)
    if cfg.figure in ["all", "utility"]:
        utility_figures(cfg, table, root)
    if cfg.figure in ["all", "weights"]:
        privacy_figures(cfg, table, root)
    if cfg.figure in ["all", "corpus"]:
        privacy_figures(cfg, table, root, corpus=True)
    if cfg.figure in ["all", "sweep"]:
        sweep_figures(cfg, table, root)


if __name__ == "__main__":
    main()
