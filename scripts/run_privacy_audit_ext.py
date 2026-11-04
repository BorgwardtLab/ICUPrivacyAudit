#!/usr/bin/env python
"""
Full privacy audit, single entry point.

    python run_privacy_audit_ext.py generate                 # full audit
    python run_privacy_audit_ext.py testbed --testbed        # memorisation testbed
    python run_privacy_audit_ext.py generate --skip-seeds    # faster, no multi-seed
    python run_privacy_audit_ext.py generate --smoke         # minutes, checks the code runs;
                                                             # numbers are meaningless

The generated corpora come from privacy.gen_cond_path / privacy.gen_null_path
(defaults in config/generate.yaml). On SLURM use experiments/run_privacy_audit.sh.
"""

import os
import sys
import copy
from functools import partial
import joblib
import numpy as np
import pandas as pd
import torch
import lightning as L
from omegaconf import OmegaConf
from hydra import initialize, compose

from syntheticicu.resolvers import register_resolvers
from syntheticicu.privacy_stage_cache import AuditStageCache
from syntheticicu.privacy_audit import (
    collect_real_sequences_with_ids,
    per_patient_percentile_risk,
    real_real_baseline_distance,
    stratified_subsample_by_center,
    _composite_key,
)
from syntheticicu.privacy_audit_ext import (
    StaticsDecoder,
    run_calibrated_membership_audit,
    unmatched_covariate_report,
    match_cohorts,
    worst_case_attack,
    run_generation_mia_v2,
    run_canary_audit_v3,
    compare_distance_distributions,
    run_subgroup_analysis_v2,
    subgroup_report_v3,
    competing_risk_model,
    chunk_count_confound_check,
    summarize_attack,
    neighborhood_calibrate,
    calibrate_threshold,
    prepare_sequences,
    multiseed_summary,
    meta_to_frame,
    inject_canaries,
    evaluate_testbed,
    patient_token_profile,
    combine_risks_explicit,
    atypical_subgroup_report,
)

register_resolvers()

# generated corpora come from cfg.privacy.gen_cond_path / gen_null_path
# (defaults in config/generate.yaml)

K_MINK = 0.2
LOCAL_PCT = 99.0  # kept in line with the risk tail percentile
TAIL_PERCENTILES = (95.0, 99.0, 99.5)
# a time-series token held by fewer patients than this in the full dataset is
# rare, and its carriers are atypical. A fixed count, so it means the same at every K
RARE_TOKEN_PATIENTS = 100
# earlier audits ran on a stratified cohort of this many chunks per centre (seed
# 0). It is rebuilt only to report in 5c what that cohort was missing.

# With 12k synthetics and ~7% prevalence: conditioned gives ~140 candidates per
# cell (class x centre) and null ~840 per cell (class). Matching at 140 is what
# makes the two modes comparable.
FULL_SIZES = dict(
    patients_per_center=None,  # None = the whole train/holdout split
    max_candidates_per_cell=140,
    pool_sweep=(10, 25, 50, 100),
    n_canaries=200,
    real_real_queries=2000,
    real_real_candidates=20000,
    matched_seeds=(0, 1, 2, 3, 4),
    gen_seeds=(0, 1, 2),
)

# --smoke: every stage runs end to end in minutes. Only checks that nothing is
# broken; the numbers it produces mean nothing.
SMOKE_SIZES = dict(
    patients_per_center=40,
    max_candidates_per_cell=20,
    pool_sweep=(5, 10),
    n_canaries=10,
    real_real_queries=50,
    real_real_candidates=500,
    matched_seeds=(0, 1),
    gen_seeds=(0, 1),
)

SMOKE_WARNING = (
    "SMOKE RUN: cohorts and pools are capped to a few patients per centre.\n"
    "# The numbers below are MEANINGLESS. This only checks that the code runs."
)


def get_model_class(cfg):
    target = getattr(cfg.model, "_target_", "syntheticicu.model_emb.get_llama_model")
    if "model_gap" in target:
        from syntheticicu.model_gap import ICULlamaGap

        return ICULlamaGap
    from syntheticicu.model_emb import ICULlama

    return ICULlama


def banner(text):
    print("\n" + "#" * 78)
    print(f"# {text}")
    print("#" * 78, flush=True)


def run_testbed(cfg):
    """
    Positive control. It needs small models to be trained, which is why it sits
    behind --testbed and does not run by default. Uses config/testbed.yaml, which
    composes from train.yaml.

    Build order follows scripts/train.py: the model instantiates the datamodule
    in its __init__ (instantiate_datamodule -> ICUDatamodule._setup -> _split),
    and nothing in model construction reads the train indices. So the canaries
    are injected into model._datamodule after construction and before fit, which
    is the point where train_indices exist and have not been consumed yet.
    """
    import hydra
    from torch.utils.data import Subset

    if "train" not in cfg or "optimizer" not in cfg.train:
        raise ValueError(
            "The testbed trains models and needs train.optimizer/lr_scheduler. "
            "Run it with the 'testbed' config: "
            "python run_privacy_audit_ext.py testbed --testbed"
        )

    from collections import Counter
    from itertools import product

    tb = cfg.get("testbed", {}) or {}
    # `centers` (list) for the multi-centre grid, `center` (single) for the first one
    centers = [
        str(c).lower() for c in (tb.get("centers") or [tb.get("center", "miiv")])
    ]
    samples = tb.get("samples", None)
    patients_per_center = tb.get("patients_per_center", None)
    n_canaries = int(tb.get("n_canaries", 200))
    dup_factors = list(tb.get("dup_factors", [50]))
    dropout_sweep = list(tb.get("center_dropout_prob_sweep", [0.1]))
    # no sweep given: the label dropout from train.yaml, and the old tag format
    sweep_label = tb.get("label_dropout_sweep") is not None
    label_sweep = list(tb.get("label_dropout_sweep") or [cfg.model.label_dropout])

    if len(centers) > 1 and samples is not None:
        # ICUDataset samples stay ids across all centres at once: centres end up
        # weighted by size, and ids that repeat across centres are kept everywhere
        raise ValueError(
            "testbed.samples only works with one centre; "
            "use testbed.patients_per_center for a multi-centre grid"
        )

    os.makedirs(cfg.logs.path, exist_ok=True)
    out = cfg.logs.path + "/memorization_testbed.joblib"
    all_results = {}
    for dup_factor, center_dropout_prob, label_dropout in product(
        dup_factors, dropout_sweep, label_sweep
    ):
        if sweep_label:
            tag = f"dup{dup_factor}_cdrop{center_dropout_prob}_ldrop{label_dropout}"
        else:
            tag = f"dup{dup_factor}_drop{center_dropout_prob}"
        banner(f"testbed: {tag} (centres {', '.join(centers)})")
        L.seed_everything(cfg.seed, workers=True)

        # few centres and few patients, so the model stays small enough to retrain.
        # ICUDatamodule forwards unknown kwargs to ICUDataset(sources=...)
        run_cfg = copy.deepcopy(cfg)
        OmegaConf.set_struct(run_cfg, False)
        run_cfg.datamodule.sources = centers
        run_cfg.datamodule.samples = samples
        run_cfg.model.center_dropout_prob = float(center_dropout_prob)
        run_cfg.model.label_dropout = float(label_dropout)
        run_cfg.logs.path = f"{cfg.logs.path}/{tag}"

        # as in train.py: the model builds the datamodule
        ModelClass = get_model_class(run_cfg)
        model = ModelClass(run_cfg)
        dm = model._datamodule

        balance = None
        if patients_per_center is not None:
            balance = balance_centres(dm, int(patients_per_center), seed=cfg.seed)

        original_train_indices = np.asarray(dm.train_indices).copy()
        canary_keys = inject_canaries(dm, n_canaries, dup_factor, cfg.seed)
        canaries_per_centre = Counter(int(k.split("_")[0]) for k in canary_keys)

        # make silently ignored overrides visible before spending GPU time
        effective = {
            "center_dropout_prob": model.label_embedder.center_dropout_prob,
            "label_dropout": model.label_embedder.dropout_prob,
        }
        loaded = np.asarray(dm.base_icu_dataset.stay_sources)
        names, counts = np.unique([str(s).lower() for s in loaded], return_counts=True)
        for name, value in effective.items():
            print(f"  Effective {name}: {value}")
        print(f"  Centres loaded (rows): {dict(zip(names.tolist(), counts.tolist()))}")
        print(f"  Canaries per centre id: {dict(sorted(canaries_per_centre.items()))}")
        if set(names.tolist()) != set(centers):
            raise ValueError(f"Expected centres {centers}, loaded {names.tolist()}")
        for name, asked in [
            ("center_dropout_prob", center_dropout_prob),
            ("label_dropout", label_dropout),
        ]:
            if abs(effective[name] - float(asked)) > 1e-9:
                raise ValueError(
                    f"{name} override ignored: asked {asked}, got {effective[name]}"
                )

        logger = [L.pytorch.loggers.CSVLogger(run_cfg.logs.path, name="csv_logs")]
        callbacks = [L.pytorch.callbacks.LearningRateMonitor()]
        trainer = hydra.utils.instantiate(
            run_cfg.trainer,
            logger=logger,
            callbacks=callbacks,
            enable_checkpointing=False,
        )
        trainer.fit(model, dm)
        trainer.save_checkpoint(
            f"{run_cfg.logs.path}/{run_cfg.model.model_name}-last.ckpt"
        )

        # score each training row once: duplicates add nothing to the per-patient
        # max but multiply the scoring cost by dup_factor
        dm.train_indices = original_train_indices
        dm.train_icu_dataset = Subset(dm.base_icu_dataset, dm.train_indices)

        # bf16-true leaves the weights in bf16; score in fp32 like the main audit
        model.float()
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()

        df, audit = evaluate_testbed(model, canary_keys, dup_factor=dup_factor)
        headline = testbed_headline(df, audit, canary_keys, n_canaries)
        print_testbed_headline({tag: headline}, title=f"testbed headline: {tag}")
        all_results[tag] = {
            "table": df,
            "summaries": audit["summaries"],
            "headline": headline,
            "dup_factor": dup_factor,
            "center_dropout_prob": float(center_dropout_prob),
            "label_dropout": float(label_dropout),
            "centers": centers,
            "patients_per_center": patients_per_center,
            "balance": balance,
            "canaries_per_centre": dict(canaries_per_centre),
            "max_steps": int(run_cfg.trainer.max_steps),
            "canary_keys": sorted(canary_keys),
        }
        # saved after every grid point, so a crash late in the sweep keeps the rest
        joblib.dump(all_results, out)

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nResults saved to: {out}")
    print("\nHow to read it:")
    print("  - canary_vs_holdout (raw) grows with dup_factor: the sensitivity curve")
    print("  - with a single centre cal_center is untestable: the true and null centre")
    print("    embeddings are two constant offsets with nothing to tell apart")
    print(
        "  - the module header predicts cal_center near chance at center_dropout_prob"
    )
    print("    0.3, because the null pathway was also trained on the members")
    print_testbed_headline(
        {t: r["headline"] for t, r in all_results.items()},
        title=f"testbed headline, all grid points (centres {', '.join(centers)})",
    )
    return all_results


def testbed_headline(df, audit, canary_keys, n_requested):
    """
    The numbers to read first from one grid point. The canary count is taken
    from the score keys rather than from the table, because evaluate_testbed
    drops a contrast row when fewer than 10 canaries match.
    """

    def auc(stat, contrast):
        row = df[(df.statistic == stat) & (df.contrast == contrast)]
        if row.empty:
            return None
        r = row.iloc[0]
        return {"auc": float(r["auc"]), "ci": (float(r["ci_low"]), float(r["ci_high"]))}

    return {
        "n_requested": int(n_requested),
        "n_injected": len(canary_keys),
        "n_found": len(set(canary_keys) & set(audit["train_scores"]["raw"])),
        "canary_vs_holdout_raw": auc("raw", "canary_vs_holdout"),
        "canary_vs_holdout_cal_center": auc("cal_center", "canary_vs_holdout"),
        # no canaries involved: the counterpart of the deployed model's train vs holdout
        "normal_vs_holdout_raw": auc("raw", "normal_vs_holdout"),
        "normal_vs_holdout_cal_center": auc("cal_center", "normal_vs_holdout"),
        "canary_vs_normal_raw": auc("raw", "canary_vs_normal"),
        "canary_vs_normal_cal_center": auc("cal_center", "canary_vs_normal"),
        "canary_vs_normal_cal_class": auc("cal_class", "canary_vs_normal"),
        "canary_vs_normal_cal_both": auc("cal_both", "canary_vs_normal"),
    }


def print_testbed_headline(headlines, title):
    def fmt(a):
        return (
            "n/a"
            if a is None
            else f"{a['auc']:.4f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]"
        )

    banner(title)
    for tag, h in headlines.items():
        flag = "" if h["n_found"] == h["n_injected"] else "   <-- MISMATCH"
        print(f"  {tag}")
        print(
            f"    canaries found in scores / injected : "
            f"{h['n_found']} / {h['n_injected']} (requested {h['n_requested']}){flag}"
        )
        # older headlines carry fewer keys; print whatever is there, in this order
        for contrast, stats in [
            ("canary_vs_holdout", ["raw", "cal_center"]),
            ("normal_vs_holdout", ["raw", "cal_center"]),
            ("canary_vs_normal", ["raw", "cal_center", "cal_class", "cal_both"]),
        ]:
            for stat in stats:
                key = f"{contrast}_{stat}"
                if key in h:
                    print(f"    {contrast + ',':<18} {stat:<11}: {fmt(h[key])}")
    sys.stdout.flush()


def _sample_patients(dm, indices, n_per_centre, rng):
    """
    Keeps all the chunks of `n_per_centre` random patients per centre among
    `indices`. Patients are (source, stay_id) pairs, since stay ids repeat across
    centres. Returns the kept indices and the patients kept per centre.
    """
    base = dm.base_icu_dataset
    stay_ids = (
        base.y[:, 0].numpy() if torch.is_tensor(base.y) else np.asarray(base.y)[:, 0]
    )
    sources = np.asarray(base.stay_sources)

    by_patient = {}
    for i in np.asarray(indices):
        by_patient.setdefault((str(sources[i]).lower(), int(stay_ids[i])), []).append(
            int(i)
        )
    by_centre = {}
    for key in sorted(by_patient):
        by_centre.setdefault(key[0], []).append(key)
    keep, kept = [], {}
    for centre, keys in by_centre.items():
        take = min(n_per_centre, len(keys))
        for j in rng.choice(len(keys), take, replace=False):
            keep.extend(by_patient[keys[j]])
        kept[centre] = take
    return np.array(sorted(keep)), kept


def shrink_splits(dm, patients_per_center, seed=0):
    """
    Smoke mode: replaces the train, val and holdout subsets with all the chunks
    of `patients_per_center` random patients per centre. Every stage iterates
    over these subsets, so capping them here caps the batches of every stage
    while keeping whole patients, which the per-patient joins between attacks
    rely on. Val is included because the token rarity profile reads it.
    """
    from torch.utils.data import Subset

    rng = np.random.default_rng(seed)
    for split in ["train", "val", "test"]:
        idx, _ = _sample_patients(
            dm, getattr(dm, f"{split}_indices"), patients_per_center, rng
        )
        setattr(dm, f"{split}_indices", idx)
        setattr(dm, f"{split}_icu_dataset", Subset(dm.base_icu_dataset, idx))
    for name, attr in [
        ("train", "train_indices"),
        ("val", "val_indices"),
        ("holdout", "test_indices"),
    ]:
        idx = getattr(dm, attr)
        print(
            f"  smoke {name}: {len(idx)} rows, ~{int(np.ceil(len(idx) / 32))} batches per pass"
        )


def attack_row(s):
    # n next to the CI: with the full population the CI is narrow, and its width
    # says nothing about magnitude, which is the AUC distance from 0.5 and the TPR
    return {
        "n_members": s["n_members"],
        "n_nonmembers": s["n_nonmembers"],
        "auc": round(s["auc"], 4),
        "auc_ci": f"[{s['auc_ci'][0]:.4f}, {s['auc_ci'][1]:.4f}]",
        "tpr@1%fpr": round(s["tpr@fpr=0.01"], 4),
        "tpr@0.1%fpr": round(s["tpr@fpr=0.001"], 4),
    }


def balance_centres(dm, patients_per_center, seed=0, min_fraction=0.8):
    """
    Multi-centre testbed: keeps `patients_per_center` patients per centre in
    total, spread over train / val / test with the datamodule's own fractions,
    so every centre weighs the same and the total stays fixed as centres are
    added. Runs before the canaries are injected, so they are drawn from the
    balanced train split. Raises if a centre falls short of `min_fraction` of
    the request, since a nearly missing centre would make cal_center untestable
    for it.
    """
    from torch.utils.data import Subset

    rng = np.random.default_rng(seed)
    n_test = int(round(patients_per_center * dm.test_fraction))
    n_val = int(round(patients_per_center * dm.val_fraction))
    n_train = patients_per_center - n_test - n_val

    report = {}
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        idx, kept = _sample_patients(dm, getattr(dm, f"{split}_indices"), n, rng)
        setattr(dm, f"{split}_indices", idx)
        setattr(dm, f"{split}_icu_dataset", Subset(dm.base_icu_dataset, idx))
        report[split] = {"requested": n, "patients": kept, "rows": len(idx)}

    print(
        "  Patients per centre after balancing (requested per split: "
        f"train {n_train}, val {n_val}, test {n_test}):"
    )
    short = []
    for split, r in report.items():
        print(f"    {split:<5} {r['patients']}  ({r['rows']} rows)")
        short += [
            f"{split}/{c}={k}"
            for c, k in r["patients"].items()
            if k < min_fraction * r["requested"]
        ]
    if short:
        raise ValueError(
            f"Centres below {min_fraction:.0%} of the requested patients: {short}. "
            "Lower testbed.patients_per_center or pick larger centres."
        )
    return report


# complete audit


def run_audit(cfg, model, skip_seeds=False, smoke=False, stage_cache=None):
    run_stage = partial(AuditStageCache.run, stage_cache)
    sizes = SMOKE_SIZES if smoke else FULL_SIZES
    gen_cond, gen_null = cfg.privacy.gen_cond_path, cfg.privacy.gen_null_path
    results = {
        "smoke": smoke,
        "sizes": dict(sizes),
        "gen_cond_path": gen_cond,
        "gen_null_path": gen_null,
    }

    if smoke:
        shrink_splits(model._datamodule, sizes["patients_per_center"], seed=cfg.seed)

    banner("0) statics decoding (bug fix)")
    decoder = StaticsDecoder(model=model)
    print(decoder.describe())
    # a batch from val is enough to check the layout before anything else runs
    probe = next(iter(model._datamodule.val_dataloader()))
    probe_statics = probe[-2]
    decoder.sanity_check(
        probe_statics.numpy()
        if torch.is_tensor(probe_statics)
        else np.asarray(probe_statics)
    )

    # every attack runs over the full train and holdout populations. A stratified
    # cohort missed exactly the atypical patients finer binning is meant to expose
    banner("2) attack A: raw vs calibrated by conditioning swap")
    mia = run_stage(
        "membership", run_calibrated_membership_audit, model, k=K_MINK, decoder=decoder
    )
    results["membership_calibrated"] = mia["summaries"]

    banner("2b) sensitivity: Min-K% without demographic tokens")
    print("  demographic tokens sit inside the scored sequence and the tail is then")
    print("  tested for enrichment in those same categories. If the effect survives")
    print("  masking them, it is physiology rather than circularity.")
    mia_nodemo = run_stage(
        "membership_no_demo",
        run_calibrated_membership_audit,
        model,
        k=K_MINK,
        decoder=decoder,
        exclude_static_tokens=True,
    )
    results["membership_no_demo_tokens"] = mia_nodemo["summaries"]

    banner("1) train vs holdout covariate imbalance")
    results["covariate_report"] = unmatched_covariate_report(
        mia["train_meta"], mia["holdout_meta"]
    )

    banner("3) attack A on the matched cohort")
    tkeys, hkeys, match_report = match_cohorts(mia["train_meta"], mia["holdout_meta"])
    results["match_report"] = match_report
    matched = {}
    for stat in ["raw", "cal_center", "cal_class", "cal_both"]:
        # matching runs on the scored population, so every matched pair is kept
        # and a missing key is a bug rather than something to filter out
        t = [mia["train_scores"][stat][k] for k in tkeys]
        h = [mia["holdout_scores"][stat][k] for k in hkeys]
        matched[stat] = summarize_attack(
            t, h, name=f"matched/{stat}", local_percentile=LOCAL_PCT
        )
    results["membership_matched"] = matched

    banner("3b) neighbourhood calibration on the holdout")
    # holdout against itself gives the reference distribution for the same statistic
    nb_t = neighborhood_calibrate(
        mia["train_scores"]["raw"],
        mia["train_meta"],
        mia["holdout_scores"]["raw"],
        mia["holdout_meta"],
    )
    nb_h = neighborhood_calibrate(
        mia["holdout_scores"]["raw"],
        mia["holdout_meta"],
        mia["holdout_scores"]["raw"],
        mia["holdout_meta"],
    )
    results["membership_neighborhood"] = summarize_attack(
        list(nb_t.values()),
        list(nb_h.values()),
        name="neighborhood-calibrated",
        local_percentile=LOCAL_PCT,
    )

    banner("4) attack A in the worst case")
    results["worst_case_membership"] = worst_case_attack(
        mia["train_scores"]["cal_both"],
        mia["holdout_scores"]["cal_both"],
        mia["train_meta"],
        mia["holdout_meta"],
        name="minK-cal_both",
    )

    banner("5) attack C: matched pools and power curve")
    print("Collecting real sequences...")
    seqs_t, ids_t, cls_t, cen_t = run_stage(
        "sequences_train",
        collect_real_sequences_with_ids,
        model,
        model._datamodule.train_icu_dataset,
        batch_size=32,
        max_batches=None,
        collect_statics=False,
        collect_labels=True,
        collect_centers=True,
    )
    seqs_h, ids_h, cls_h, cen_h = run_stage(
        "sequences_holdout",
        collect_real_sequences_with_ids,
        model,
        model._datamodule.test_icu_dataset,
        batch_size=32,
        max_batches=None,
        collect_statics=False,
        collect_labels=True,
        collect_centers=True,
    )
    # val is never attacked; it only completes the full dataset for token rarity
    seqs_v, ids_v, cen_v = run_stage(
        "sequences_val",
        collect_real_sequences_with_ids,
        model,
        model._datamodule.val_icu_dataset,
        batch_size=32,
        max_batches=None,
        collect_statics=False,
        collect_labels=False,
        collect_centers=True,
    )

    real_train = {
        "seqs": seqs_t,
        "ids": np.asarray(ids_t),
        "classes": np.asarray(cls_t),
        "centers": np.asarray(cen_t),
    }
    real_holdout = {
        "seqs": seqs_h,
        "ids": np.asarray(ids_h),
        "classes": np.asarray(cls_h),
        "centers": np.asarray(cen_h),
    }
    full_real = {
        "train": {"seqs": seqs_t, "ids": ids_t, "centers": cen_t},
        "val": {"seqs": seqs_v, "ids": ids_v, "centers": cen_v},
        "test": {"seqs": seqs_h, "ids": ids_h, "centers": cen_h},
    }

    gen_results = {}
    for label, path in [("conditioned", gen_cond), ("null", gen_null)]:
        banner(f"5.{label}) attack C, {label} mode")
        gen_results[label] = run_stage(
            f"generation_{label}",
            run_generation_mia_v2,
            model,
            path,
            real_train,
            real_holdout,
            use_center="auto",
            canonicalize=True,
            chunk_zero_only=True,
            require_eos=True,
            max_candidates_per_cell=sizes["max_candidates_per_cell"],
            pool_sweep=sizes["pool_sweep"],
            label=label,
        )
    print(
        f"\n  Both modes share {sizes['max_candidates_per_cell']} candidates per cell, "
        "so their AUCs are comparable."
    )

    banner("5b) real-real baseline and calibrated threshold")
    # the same preparation as the attack, otherwise the baseline would not be
    # comparable with the synthetic distances
    prep_t = prepare_sequences(
        real_train["seqs"], canonicalize=True, chunk_zero_only=True
    )
    prep_h = prepare_sequences(
        real_holdout["seqs"], canonicalize=True, chunk_zero_only=True
    )
    real_real = run_stage(
        "real_real",
        real_real_baseline_distance,
        prep_h[: sizes["real_real_queries"]],
        prep_t,
        max_candidates=sizes["real_real_candidates"],
    )
    results["real_real_dists"] = real_real
    results["tau"] = calibrate_threshold(real_real, percentile=1.0)

    raw_c = gen_results["conditioned"]["_raw"]

    banner("5c) token rarity in the full audited population")
    profile = patient_token_profile(full_real, rare_threshold=RARE_TOKEN_PATIENTS)
    patients = profile["patients"]
    atypical = set(patients.index[patients["atypical"]])
    results["token_profile"] = profile

    banner("5d) distance distributions after the corrections")
    results["distance_comparison"] = compare_distance_distributions(
        raw_c["train_scores"], raw_c["holdout_scores"], real_real
    )

    banner("6) canary: rank-1, corruption sweep, canonicalisation ablation")
    print("  this measures the sensitivity of the distance metric, not the model's")
    print("  memorisation. For that there is the --testbed mode.")
    results["canary"] = run_stage(
        "canary",
        run_canary_audit_v3,
        raw_c["train_prepared"]["seqs"],
        raw_c["train_prepared"]["classes"],
        raw_c["train_prepared"]["centers"],
        raw_c["gen_seqs"],
        raw_c["gen_labels"],
        raw_c["gen_centers"],
        n_canaries=sizes["n_canaries"],
        drop_fractions=(0.05, 0.10, 0.20, 0.40, 0.80),
        canonicalize_settings=(True, False),
        max_candidates_per_cell=sizes["max_candidates_per_cell"],
    )

    banner("7) per-patient risk and subgroups")

    # the competing model shows age loses to log(length), but that only closes
    # the proxy argument if the two are actually correlated
    meta_df = meta_to_frame(mia["train_meta"])
    rho = meta_df[["age_static", "total_len"]].corr(method="spearman").iloc[0, 1]
    print(f"\n  Spearman(age_static, total_len) = {rho:.3f}")
    print(
        meta_df.groupby("age_static")["total_len"]
        .agg(["median", "mean", "count"])
        .to_string()
    )
    results["age_length_correlation"] = float(rho)

    # attack A does not depend on the generation mode, so this part is shared
    risks_mink = per_patient_percentile_risk(
        mia["train_scores"]["cal_both"],
        mia["holdout_scores"]["cal_both"],
        higher_is_riskier=True,
    )
    subgroup_out, combined_by_mode = {}, {}
    for label in ["conditioned", "null"]:
        risks_gen = per_patient_percentile_risk(
            gen_results[label]["train_patient_scores"],
            gen_results[label]["holdout_patient_scores"],
            higher_is_riskier=False,
        )
        print(f"\n--- {label} mode ---")
        combined, combination = combine_risks_explicit(
            {"attack_a": risks_mink, "attack_c": risks_gen}, method="max"
        )
        out = run_subgroup_analysis_v2(combined, mia["train_meta"], percentile=99.0)
        out["full_report"] = subgroup_report_v3(
            combined, mia["train_meta"], percentiles=TAIL_PERCENTILES
        )
        out["competing_model"] = competing_risk_model(
            combined, mia["train_meta"], percentile=99.0
        )
        out["chunk_confound"] = chunk_count_confound_check(combined, mia["train_meta"])
        out["combination"] = combination
        subgroup_out[label] = out
        combined_by_mode[label] = combined
    results["subgroups"] = subgroup_out

    banner(
        f"7b) atypical patients (a time-series token held by < {RARE_TOKEN_PATIENTS} patients)"
    )
    attacks = {
        "A/cal_both": (
            mia["train_scores"]["cal_both"],
            mia["holdout_scores"]["cal_both"],
            True,
        ),
        "A/raw": (mia["train_scores"]["raw"], mia["holdout_scores"]["raw"], True),
    }
    for label, g in gen_results.items():
        attacks[f"C/{label}"] = (
            g["train_patient_scores"],
            g["holdout_patient_scores"],
            False,
        )
    results["atypical"] = atypical_subgroup_report(
        attacks, atypical, combined_by_mode, mia["train_meta"], percentile=99.0
    )

    # saved so subgroups (e.g. another rarity threshold, via
    # token_profile["patients"].rarest_token_patients) can be re-cut offline
    results["per_patient"] = {
        "attack_a": {"train": mia["train_scores"], "holdout": mia["holdout_scores"]},
        "attack_a_no_demo_tokens": {
            "train": mia_nodemo["train_scores"],
            "holdout": mia_nodemo["holdout_scores"],
        },
        "attack_c": {
            label: {
                "train": g["train_patient_scores"],
                "holdout": g["holdout_patient_scores"],
            }
            for label, g in gen_results.items()
        },
        "combined_risk": combined_by_mode,
        "train_meta": mia["train_meta"],
        "holdout_meta": mia["holdout_meta"],
    }

    if not skip_seeds:
        banner("8) cross-seed stability")

        def _matched_scores(seed):
            # for attack A the seed only changes the matching, so that is what we redraw
            tk, hk, _ = match_cohorts(mia["train_meta"], mia["holdout_meta"], seed=seed)
            return (
                [mia["train_scores"]["cal_both"][k] for k in tk],
                [mia["holdout_scores"]["cal_both"][k] for k in hk],
            )

        results["seed_stability_matched"] = multiseed_summary(
            _matched_scores, seeds=sizes["matched_seeds"], name="matched/cal_both"
        )

        def _gen_scores(seed):
            # for attack C the seed changes which candidates enter each cell
            r = run_stage(
                f"generation_null_seed{seed}",
                run_generation_mia_v2,
                model,
                gen_null,
                real_train,
                real_holdout,
                use_center="auto",
                canonicalize=True,
                chunk_zero_only=True,
                require_eos=True,
                max_candidates_per_cell=sizes["max_candidates_per_cell"],
                pool_sweep=None,
                seed=seed,
                label=f"null-seed{seed}",
            )
            return (
                [-v for v in r["train_patient_scores"].values()],
                [-v for v in r["holdout_patient_scores"].values()],
            )

        results["seed_stability_gen"] = multiseed_summary(
            _gen_scores, seeds=sizes["gen_seeds"], name="gen-MIA null"
        )

    # the full sequences behind attack C would make the results file several GB;
    # the per-patient scores above are what re-analysis needs
    results["generation_mia"] = {
        label: {k: v for k, v in g.items() if k != "_raw"}
        for label, g in gen_results.items()
    }

    banner("summary")
    print("\nRead the CIs as precision, not effect size. With ~215k members and ~57k")
    print("non-members they are narrow enough that a negligible AUC excess separates")
    print("from 0.5; magnitude is |AUC - 0.5| and above all the TPR at low FPR.")

    print("\nTable 1: attack a (weight release), full population")
    print(
        pd.DataFrame(
            [{"statistic": k, **attack_row(v)} for k, v in mia["summaries"].items()]
        ).to_string(index=False)
    )
    print("  cal_center below 0.5 is not a neutral calibration: with centre dropout")
    print("  0.3 the null pathway was trained on the members too. Read the module")
    print("  header note before writing this up.")

    print(
        f"\nTable 2: attack a on the matched cohort "
        f"({match_report['n_matched']} pairs, "
        f"{match_report['match_rate_holdout']:.1%} of the holdout)"
    )
    print(
        pd.DataFrame(
            [{"statistic": k, **attack_row(v)} for k, v in matched.items()]
        ).to_string(index=False)
    )

    print("\nTable 3: attack c with matched pools, full population")
    print(
        pd.DataFrame(
            [
                {
                    "mode": k,
                    **attack_row(v["summary"]),
                    "n_synthetic": v["n_gen_used"],
                    "center_restricted": v["use_center"],
                }
                for k, v in gen_results.items()
            ]
        ).to_string(index=False)
    )

    print("\nTable 4: attack c power curve (conditioned)")
    print(gen_results["conditioned"]["pool_sweep"].to_string(index=False))

    print("\nTable 5: canary: rank-1 recovery")
    print(results["canary"].to_string(index=False))

    print("\nTable 6: combined risk population (max over attacks a and c)")
    print(
        pd.DataFrame(
            [{"mode": k, **v["combination"]} for k, v in subgroup_out.items()]
        ).to_string(index=False)
    )

    print(
        f"\nTable 7: atypical patients (a time-series token held by "
        f"< {RARE_TOKEN_PATIENTS} patients)"
    )
    print(profile["by_split"].to_string(float_format=lambda x: f"{x:.4g}"))
    print(
        results["atypical"]["by_attack"].to_string(
            index=False, float_format=lambda x: f"{x:.4g}"
        )
    )
    print(
        results["atypical"]["tail_enrichment"].to_string(
            index=False, float_format=lambda x: f"{x:.4g}"
        )
    )

    print("\nPost-processing bound: the synthetic corpus is a function of the model,")
    print("so the bound compares optimal adversaries in the two release scenarios.")
    print("The measured attack A does not upper-bound the measured attack C;")
    print("each attack needs its own empirical sensitivity calibration.")
    return results


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    unknown = flags - {"--testbed", "--skip-seeds", "--smoke"}
    if unknown:
        # a typo such as --smoek would otherwise launch the full audit
        raise SystemExit(f"Unknown flags: {sorted(unknown)}")
    default_config = "testbed" if "--testbed" in flags else "generate"
    config_name = args[0] if args else default_config
    overrides = args[1:]
    smoke = "--smoke" in flags

    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
    L.seed_everything(cfg.seed, workers=True)

    if "--testbed" in flags:
        run_testbed(cfg)
        return

    # fail before loading the model, not an hour into attack A
    missing = [
        p
        for p in (cfg.privacy.gen_cond_path, cfg.privacy.gen_null_path)
        if not os.path.isfile(p)
    ]
    if missing:
        raise FileNotFoundError(f"Generated corpus not found: {missing}")
    print(f"Conditioned corpus: {cfg.privacy.gen_cond_path}")
    print(f"Null corpus:        {cfg.privacy.gen_null_path}")
    if smoke:
        banner(SMOKE_WARNING)

    print(f"Loading model from {cfg.model.pretrained_path}...", flush=True)
    ModelClass = get_model_class(cfg)
    model = ModelClass.load_from_checkpoint(
        cfg.model.pretrained_path, new_cfg=cfg, weights_only=False
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    results = run_audit(cfg, model, skip_seeds="--skip-seeds" in flags, smoke=smoke)

    os.makedirs(cfg.logs.path, exist_ok=True)
    name = "privacy_audit_smoke.joblib" if smoke else "privacy_audit_results.joblib"
    out_path = os.path.join(cfg.logs.path, name)
    joblib.dump(results, out_path)
    print(f"\nResults saved to: {out_path}")
    if smoke:
        banner(SMOKE_WARNING)


if __name__ == "__main__":
    main()
