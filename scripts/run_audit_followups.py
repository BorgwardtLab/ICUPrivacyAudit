"""Extend saved discrete corpus corruption floors with reproduction checks."""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import lightning as L
from omegaconf import OmegaConf
from hydra import initialize, compose

from syntheticicu.resolvers import register_resolvers
from syntheticicu.privacy_audit import collect_real_sequences_with_ids
from syntheticicu.privacy_audit_ext import prepare_sequences, run_canary_audit_v3

register_resolvers()

# mink-score

# mink-analyse, as in Privacy_analysis.ipynb so the tables line up with the bin sweep

# canary-fractions
CHECKED_COLUMNS = ["n", "rank1_rate", "median_margin", "median_d_canary"]


def get_model_class(cfg):
    target = getattr(cfg.model, "_target_", "syntheticicu.model_emb.get_llama_model")
    if "model_gap" in target:
        from syntheticicu.model_gap import ICULlamaGap

        return ICULlamaGap
    from syntheticicu.model_emb import ICULlama

    return ICULlama


def load_audited_model(overrides):
    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name="generate", overrides=overrides)
    L.seed_everything(cfg.seed, workers=True)
    print(f"Loading model from {cfg.model.pretrained_path}...", flush=True)
    model = get_model_class(cfg).load_from_checkpoint(
        cfg.model.pretrained_path, new_cfg=cfg, weights_only=False
    )
    model.eval()
    return cfg, model


# mink-score


# mink-analyse


# canary-fractions
def variant_label(frac):
    # the label run_canary_audit_v3 gives a dropped variant
    return f"dropped@{int(frac * 100)}%"


def load_train_prepared(model):
    # same call as section 5 of the audit
    seqs, ids, classes, centers = collect_real_sequences_with_ids(
        model,
        model._datamodule.train_icu_dataset,
        batch_size=32,
        max_batches=None,
        collect_statics=False,
        collect_labels=True,
        collect_centers=True,
    )
    prepared, mask = prepare_sequences(
        seqs,
        canonicalize=True,
        chunk_zero_only=True,
        require_eos=False,
        return_mask=True,
    )
    print(f"  Train chunk-0 sequences: {len(prepared)}/{len(seqs)}")
    return prepared, np.asarray(classes)[mask], np.asarray(centers)[mask]


def load_gen_prepared(path):
    # as in run_generation_mia_v2
    samples, labels, centers = joblib.load(path)
    to_np = lambda a: np.asarray(a.cpu().numpy() if torch.is_tensor(a) else a).astype(
        int
    )
    labels, centers = to_np(labels), to_np(centers)
    seqs, mask = prepare_sequences(
        [list(s) for s in samples],
        canonicalize=True,
        chunk_zero_only=False,
        require_eos=True,
        return_mask=True,
    )
    print(f"  Synthetics after dropping incomplete ones: {len(seqs)}/{len(samples)}")
    return seqs, labels[mask], centers[mask]


def compare_with_stored(new, stored, labels):
    keys = ["canonicalize", "variant"]
    a = new[new.variant.isin(labels)].set_index(keys)[CHECKED_COLUMNS]
    b = stored[stored.variant.isin(labels)].set_index(keys)[CHECKED_COLUMNS]
    if not a.index.sort_values().equals(b.index.sort_values()):
        print(f"  Rows differ: recomputed {sorted(a.index)}, stored {sorted(b.index)}")
        return False
    b = b.loc[a.index]
    close = np.isclose(
        a.to_numpy(float), b.to_numpy(float), rtol=0, atol=1e-9, equal_nan=True
    )
    report = pd.concat({"recomputed": a, "stored": b}, axis=1)
    print(report.to_string())
    return bool(close.all())


def cmd_canary_fractions(args, overrides):
    stored = joblib.load(args.audit)
    if stored["smoke"]:
        raise SystemExit(f"{args.audit} is a smoke run")
    stored_canary = stored["canary"]
    sizes = stored["sizes"]
    check_label = variant_label(args.check_fraction)
    if check_label not in set(stored_canary.variant):
        raise SystemExit(
            f"{check_label} is not in the stored table: {sorted(set(stored_canary.variant))}"
        )
    print("Stored canary table:")
    print(stored_canary.to_string(index=False))

    # only the datamodule is used: collecting sequences needs no forward pass
    cfg, model = load_audited_model(overrides)
    if cfg.privacy.gen_cond_path != stored["gen_cond_path"]:
        raise SystemExit(
            f"Corpus differs from the audit's: {cfg.privacy.gen_cond_path} "
            f"vs {stored['gen_cond_path']}"
        )

    train_seqs, train_classes, train_centers = load_train_prepared(model)
    gen_seqs, gen_labels, gen_centers = load_gen_prepared(stored["gen_cond_path"])

    fractions = sorted(set(args.fractions) | {args.check_fraction})
    df = run_canary_audit_v3(
        train_seqs,
        train_classes,
        train_centers,
        gen_seqs,
        gen_labels,
        gen_centers,
        n_canaries=sizes["n_canaries"],
        drop_fractions=tuple(fractions),
        canonicalize_settings=(True, False),
        max_candidates_per_cell=sizes["max_candidates_per_cell"],
    )

    print("\nCheck against the stored table:")
    check_passed = compare_with_stored(
        df, stored_canary, ["literal", "shuffled", check_label]
    )
    print(f"  Check passed: {check_passed}")

    new_labels = [variant_label(f) for f in args.fractions]
    merged = pd.concat(
        [stored_canary, df[df.variant.isin(new_labels)]], ignore_index=True
    )
    merged = merged.drop_duplicates(["canonicalize", "variant"], keep="first")

    if not check_passed:
        raise ValueError(
            "Check failed; refusing to export incomparable corruption points"
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(
        {
            "canary": df,
            "canary_merged": merged,
            "fractions": args.fractions,
            "check_fraction": args.check_fraction,
            "check_passed": check_passed,
            "audit": args.audit,
            "gen_cond_path": stored["gen_cond_path"],
            "sizes": sizes,
        },
        args.out,
    )
    print(f"\nMerged canary table (stored + new levels):")
    print(merged.to_string(index=False))
    print(f"\nResults saved to: {args.out}")
    if not check_passed:
        sys.exit(
            "The recomputed rows do not match the stored audit; the new levels "
            "are not comparable with it."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extend saved corpus recovery fractions"
    )
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.3, 0.5, 0.6, 0.7]
    )
    parser.add_argument("--check-fraction", type=float, default=0.4)
    args, overrides = parser.parse_known_args()
    cmd_canary_fractions(args, overrides)


if __name__ == "__main__":
    main()
