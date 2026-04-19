"""Train v14 with 5 different MODEL_SEEDs via the toolkit.

After all 5 register, auto-blend them. Expected gain vs single v14: +0.0005-0.0015.

This validates that the toolkit's `train_variant` + `registry` + `blend` chain
works for a real multi-model ensembling flow.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# Import v14 setup from reproduction example
sys.path.insert(0, str(Path(__file__).parent))
from s6e4_v14_reproduction import (
    DATA_RAW, TARGET, TARGET_MAPPING, V14_XGB_PARAMS,
    build_v14_features, S6E4_ROOT,
)
from kaggle_playground_utils import config
from kaggle_playground_utils.features import get_cat_cols
from kaggle_playground_utils.train import TrainConfig, train_variant
from kaggle_playground_utils.registry import Registry
from kaggle_playground_utils.blend import (
    simple_average, nelder_mead_blend, pairwise_correlations,
)

SEEDS = [11, 42, 123, 2026, 7]


def main():
    print("=" * 60)
    print(f"5-seed sweep (MODEL_SEEDs={SEEDS})")
    print("=" * 60)
    t_total = time.time()

    # Load data + FE once
    print("\n[1/3] Loading data + FE (once, reused for all seeds)...")
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    y_full = train[TARGET].map(TARGET_MAPPING).values
    train = train.drop(columns=["id", TARGET])
    test = test.drop(columns=["id"])

    train, test = build_v14_features(train, test)
    cat_cols = get_cat_cols(train)
    for c in cat_cols:
        train[c] = train[c].astype(str).astype("category")
        test[c] = test[c].astype(str).astype("category")

    tr_idx, ho_idx = train_test_split(
        np.arange(len(y_full)), test_size=config.HOLDOUT_FRAC,
        stratify=y_full, random_state=config.HOLDOUT_SEED,
    )
    X_tr = train.iloc[tr_idx].reset_index(drop=True)
    X_ho = train.iloc[ho_idx].reset_index(drop=True)
    y_tr = y_full[tr_idx]
    y_ho = y_full[ho_idx]

    # Train each seed, register
    print(f"\n[2/3] Training {len(SEEDS)} seeds via train_variant()...")
    reg = Registry(root=S6E4_ROOT / "registry")
    results = []
    for seed in SEEDS:
        params_with_seed = dict(V14_XGB_PARAMS)
        params_with_seed["random_state"] = seed
        cfg = TrainConfig(
            algo="xgb",
            params=params_with_seed,
            n_classes=3,
            cv_seed=config.CV_SEED,
            model_seed=seed,
            n_folds=5,
            optuna_trials=200,
            optuna_n_jobs=4,
            register_as=f"v14_seed_{seed}",
            tags=["s6e4", "seed_sweep", "xgb"],
            notes=f"v14 XGB with MODEL_SEED={seed}",
            verbose=True,
        )
        print(f"\n--- SEED {seed} ---")
        result = train_variant(cfg, X_tr, y_tr, X_ho, y_ho, test, registry=reg)
        results.append((seed, result))

    # Blend
    print(f"\n[3/3] Auto-blend across {len(SEEDS)} seeds...")

    # Correlation diagnostic
    corr = pairwise_correlations(reg, tags=["seed_sweep"], split="oof_probs")
    print(f"\n  Pairwise correlations (OOF):")
    for i, id1 in enumerate(corr["ids"]):
        for j, id2 in enumerate(corr["ids"]):
            if j > i:
                print(f"    {id1} vs {id2}: ρ = {corr['matrix'][i, j]:.5f}")

    # Simple average
    avg_ho = simple_average(reg, tags=["seed_sweep"], split="holdout_probs")
    avg_test = simple_average(reg, tags=["seed_sweep"], split="test_probs")
    avg_bacc = balanced_accuracy_score(y_ho, avg_ho.argmax(axis=1))
    print(f"\n  Simple average holdout bacc: {avg_bacc:.5f}")

    # OOF-weighted blend with Optuna-tuned class weights
    blend_result = nelder_mead_blend(
        reg, y_tr, tags=["seed_sweep"], regularization=0.01,
        split_for_apply="holdout_probs",
    )
    print(f"\n  Nelder-Mead weights: {blend_result['weights']}")
    print(f"  OOF score: {blend_result['oof_score']:.5f}")
    blend_ho_bacc = balanced_accuracy_score(y_ho, blend_result["applied"].argmax(axis=1))
    print(f"  Blend holdout bacc: {blend_ho_bacc:.5f}")

    # Summary
    print("\n" + "=" * 60)
    print("5-SEED SWEEP SUMMARY")
    print("=" * 60)
    v14_baseline = 0.97423
    print(f"\n  v14 single-seed baseline:  {v14_baseline:.5f}")
    print(f"  Simple average (5 seeds):  {avg_bacc:.5f}  ({avg_bacc - v14_baseline:+.5f})")
    print(f"  Nelder-Mead blend:         {blend_ho_bacc:.5f}  ({blend_ho_bacc - v14_baseline:+.5f})")
    print(f"\n  Individual seed scores:")
    for seed, result in results:
        print(f"    seed={seed:<5}: holdout={result['holdout_score_tuned']:.5f}")
    print(f"\n  Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
