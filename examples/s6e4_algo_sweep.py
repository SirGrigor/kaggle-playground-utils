"""Train LGB + CatBoost variants via the toolkit on S6E4.

Tests whether algorithm diversity produces lower correlation than seed diversity.
Combined with 5 existing XGB seeds in registry → auto-blend should give real gain.

Expected per L13:
  - XGB vs LGB:      ρ ~ 0.94-0.96 (different splitting strategies)
  - XGB vs CatBoost: ρ ~ 0.93-0.95 (ordered boosting different)
  - LGB vs CatBoost: ρ ~ 0.95

Expected blend gain with XGB(×5) + LGB + CatBoost: +0.001-0.003 over best single.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from s6e4_v14_reproduction import (
    DATA_RAW, TARGET, TARGET_MAPPING, build_v14_features, S6E4_ROOT,
)
from kaggle_playground_utils import config
from kaggle_playground_utils.features import get_cat_cols
from kaggle_playground_utils.train import TrainConfig, train_variant
from kaggle_playground_utils.registry import Registry
from kaggle_playground_utils.blend import (
    simple_average, nelder_mead_blend, pairwise_correlations,
)

# yunsuxiaozi-style LGB params (from grandmaster audit)
LGB_PARAMS = dict(
    n_estimators=6000,
    max_depth=4,
    num_leaves=32,
    learning_rate=0.05,
    feature_fraction=0.6,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l1=10,
    lambda_l2=10,
    min_child_samples=12,
    max_bin=15000,
    subsample_for_bin=100000,
    subsample_freq=1,
    objective="multiclass",
    num_class=3,
    metric="multi_logloss",
    random_state=config.MODEL_SEED,
    verbose=-1,
    early_stopping_rounds=250,
)

# CatBoost GPU params
CATBOOST_PARAMS = dict(
    iterations=3000,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=5,
    random_seed=config.MODEL_SEED,
    task_type="GPU",
    objective="MultiClass",
    eval_metric="MultiClass",
    early_stopping_rounds=300,
)


def main():
    print("=" * 60)
    print("Algorithm diversity sweep: LGB + CatBoost")
    print("=" * 60)
    t_total = time.time()

    # Load data + FE (same as v14)
    print("\n[1/5] Loading data + FE...")
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

    # For CatBoost: get list of cat col INDICES (CatBoost needs this explicitly)
    cat_indices = [train.columns.get_loc(c) for c in cat_cols]

    tr_idx, ho_idx = train_test_split(
        np.arange(len(y_full)), test_size=config.HOLDOUT_FRAC,
        stratify=y_full, random_state=config.HOLDOUT_SEED,
    )
    X_tr = train.iloc[tr_idx].reset_index(drop=True)
    X_ho = train.iloc[ho_idx].reset_index(drop=True)
    y_tr = y_full[tr_idx]
    y_ho = y_full[ho_idx]

    reg = Registry(root=S6E4_ROOT / "registry")
    results = []

    # --- LGB ---
    print("\n[2/5] LGB variant (yunsuxiaozi-style params)...")
    lgb_cfg = TrainConfig(
        algo="lgb",
        params=LGB_PARAMS,
        n_classes=3,
        cv_seed=config.CV_SEED,
        model_seed=config.MODEL_SEED,
        n_folds=5,
        optuna_trials=200,
        optuna_n_jobs=4,
        register_as="v14_lgb",
        tags=["s6e4", "algo_sweep", "lgb"],
        notes="LGB with yunsuxiaozi-style params (lambda_l1=10, lambda_l2=10, max_bin=15000)",
        verbose=True,
    )
    lgb_result = train_variant(lgb_cfg, X_tr, y_tr, X_ho, y_ho, test, registry=reg)
    results.append(("lgb", lgb_result))

    # --- CatBoost ---
    print("\n[3/5] CatBoost variant (GPU, balanced class weights)...")
    cb_params = dict(CATBOOST_PARAMS)
    cb_params["cat_features"] = cat_indices
    cb_cfg = TrainConfig(
        algo="catboost",
        params=cb_params,
        n_classes=3,
        cv_seed=config.CV_SEED,
        model_seed=config.MODEL_SEED,
        n_folds=5,
        optuna_trials=200,
        optuna_n_jobs=4,
        register_as="v14_catboost",
        tags=["s6e4", "algo_sweep", "catboost"],
        notes="CatBoost GPU with auto_class_weights=Balanced",
        verbose=True,
    )
    # CatBoost doesn't accept pandas category dtype directly; convert to string
    X_tr_cb = X_tr.copy()
    X_ho_cb = X_ho.copy()
    test_cb = test.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_ho_cb[c] = X_ho_cb[c].astype(str)
        test_cb[c] = test_cb[c].astype(str)
    cb_result = train_variant(cb_cfg, X_tr_cb, y_tr, X_ho_cb, y_ho, test_cb, registry=reg)
    results.append(("catboost", cb_result))

    # --- Correlation analysis ---
    print("\n[4/5] Correlation analysis across all registered models...")
    all_models_corr = pairwise_correlations(reg, tags=["s6e4"], split="oof_probs")
    print(f"\n  {'Model pair':<50} {'ρ':<10}")
    ids = all_models_corr["ids"]
    mat = all_models_corr["matrix"]
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            print(f"  {ids[i]:<20} vs {ids[j]:<20} ρ = {mat[i,j]:.5f}")

    # --- Full blend ---
    print("\n[5/5] Full blend (all XGB seeds + LGB + CatBoost)...")
    avg_ho = simple_average(reg, tags=["s6e4"], split="holdout_probs")
    avg_bacc = balanced_accuracy_score(y_ho, avg_ho.argmax(axis=1))
    print(f"  Simple average holdout:   {avg_bacc:.5f}")

    blend_result = nelder_mead_blend(
        reg, y_tr, tags=["s6e4"], regularization=0.01,
        split_for_apply="holdout_probs",
    )
    blend_bacc = balanced_accuracy_score(y_ho, blend_result["applied"].argmax(axis=1))
    print(f"  Nelder-Mead weights: {blend_result['weights']}")
    print(f"  Nelder-Mead holdout:      {blend_bacc:.5f}")

    # Summary
    print("\n" + "=" * 60)
    print("ALGO SWEEP SUMMARY")
    print("=" * 60)
    print(f"  v14 XGB baseline:      0.97423")
    print(f"  LGB single:            {lgb_result['holdout_score_tuned']:.5f}")
    print(f"  CatBoost single:       {cb_result['holdout_score_tuned']:.5f}")
    print(f"  Simple avg (all):      {avg_bacc:.5f}  ({avg_bacc - 0.97423:+.5f})")
    print(f"  Nelder-Mead blend:     {blend_bacc:.5f}  ({blend_bacc - 0.97423:+.5f})")
    print(f"  Total time:            {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
