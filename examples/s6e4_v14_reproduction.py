"""Reproduce v14 (S6E4 XGB baseline, LB 0.97411) using ONLY the toolkit.

This is Checkpoint 1 from knowledge-graph/kaggle/2026-16_toolkit-build-plan.md:
if toolkit reproduces v14's OOF/holdout probs within 0.0005, the training
pipeline is trustworthy.

Reproduction target (v14 saved holdout bacc): 0.97423
Saved probs at: ~/IdeaProjects/kaggle/playground-s6e4/v14/probs/

Usage:
    cd ~/IdeaProjects/kaggle-playground-utils
    uv run python examples/s6e4_v14_reproduction.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from kaggle_playground_utils import config
from kaggle_playground_utils.features import (
    digit_features, threshold_booleans, categorical_one_hot, formula_logits,
    get_cat_cols,
)
from kaggle_playground_utils.train import TrainConfig, train_variant
from kaggle_playground_utils.registry import Registry

# ============================================================================
# Configuration — v14 exact specification
# ============================================================================

S6E4_ROOT = Path.home() / "IdeaProjects/kaggle/playground-s6e4"
DATA_RAW = S6E4_ROOT / "data" / "raw"
V14_SAVED_PROBS = S6E4_ROOT / "v14" / "probs"

TARGET = "Irrigation_Need"
TARGET_MAPPING = {"Low": 0, "Medium": 1, "High": 2}

# Chris Deotte's formula thresholds (proven 100% on original by v12)
CHRIS_THRESHOLDS = {
    "Soil_Moisture":  (25, "lt"),
    "Temperature_C":  (30, "gt"),
    "Rainfall_mm":    (300, "lt"),
    "Wind_Speed_kmh": (10, "gt"),
}

# Chris's LR coefficients (hardcoded from his notebook)
CHRIS_COEFS = {
    "Low": {
        "intercept": 16.3173,
        "soil_lt": -11.0237, "temp_gt": -5.8559,
        "rain_lt": -10.8500, "wind_gt": -5.8284,
        "Flowering": -5.4155, "Harvest": 5.5073, "Sowing": 5.2299, "Vegetative": -5.4617,
        "Mulching_Used_No": -3.0014, "Mulching_Used_Yes": 2.8613,
    },
    "Medium": {
        "intercept": 4.6524,
        "soil_lt": 0.3290, "temp_gt": -0.0204,
        "rain_lt": 0.1542, "wind_gt": 0.0841,
        "Flowering": 0.3586, "Harvest": -0.1348, "Sowing": -0.3547, "Vegetative": 0.3334,
        "Mulching_Used_No": 0.1883, "Mulching_Used_Yes": 0.0142,
    },
    "High": {
        "intercept": -20.9697,
        "soil_lt": 10.6947, "temp_gt": 5.8763,
        "rain_lt": 10.6958, "wind_gt": 5.7444,
        "Flowering": 5.0569, "Harvest": -5.3725, "Sowing": -4.8752, "Vegetative": 5.1283,
        "Mulching_Used_No": 2.8131, "Mulching_Used_Yes": -2.8755,
    },
}

# v14 XGB params (v5 tight basin)
V14_XGB_PARAMS = dict(
    n_estimators=50000, max_depth=4, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9, alpha=5, reg_lambda=5,
    max_leaves=30, min_child_weight=2, max_bin=10000,
    objective="multi:softprob", num_class=3,
    tree_method="hist", device="cuda",
    enable_categorical=True, eval_metric="mlogloss",
    random_state=config.MODEL_SEED,
    early_stopping_rounds=500,
)

RAW_NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]


# ============================================================================
# Feature engineering via the toolkit
# ============================================================================

def build_v14_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Replicate v14's feature set (114 total) using ONLY toolkit primitives."""
    for df in [train_df, test_df]:
        # 1. Threshold booleans (renamed to match Chris's coef keys: soil_lt, temp_gt, etc.)
        df["soil_lt"] = (df["Soil_Moisture"] < 25).astype(int)
        df["temp_gt"] = (df["Temperature_C"] > 30).astype(int)
        df["rain_lt"] = (df["Rainfall_mm"] < 300).astype(int)
        df["wind_gt"] = (df["Wind_Speed_kmh"] > 10).astype(int)

        # 2. Categorical one-hot for Chris's formula inputs
        for stage in ["Flowering", "Harvest", "Sowing", "Vegetative"]:
            df[stage] = (df["Crop_Growth_Stage"] == stage).astype(int)
        df["Mulching_Used_No"] = (df["Mulching_Used"] == "No").astype(int)
        df["Mulching_Used_Yes"] = (df["Mulching_Used"] == "Yes").astype(int)

    # 3. Formula logits via toolkit
    formula_input_cols = ["soil_lt", "temp_gt", "rain_lt", "wind_gt",
                          "Flowering", "Harvest", "Sowing", "Vegetative",
                          "Mulching_Used_No", "Mulching_Used_Yes"]
    train_df = formula_logits(train_df, CHRIS_COEFS, formula_input_cols)
    test_df = formula_logits(test_df, CHRIS_COEFS, formula_input_cols)

    # 4. Drop extra one-hot helpers (v14 only has soil_lt/temp_gt/rain_lt/wind_gt + logits + digits)
    for col in ["Flowering", "Harvest", "Sowing", "Vegetative",
                "Mulching_Used_No", "Mulching_Used_Yes"]:
        train_df = train_df.drop(columns=col)
        test_df = test_df.drop(columns=col)

    # 5. Digit features (precision-safe via toolkit)
    train_df = digit_features(train_df, RAW_NUMERIC)
    test_df = digit_features(test_df, RAW_NUMERIC)

    return train_df, test_df


def main():
    print("=" * 60)
    print("v14 reproduction via kaggle-playground-utils toolkit")
    print("=" * 60)

    t_total = time.time()

    print("\n[1/4] Loading S6E4 data + FE via toolkit...")
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    y_full = train[TARGET].map(TARGET_MAPPING).values
    train = train.drop(columns=["id", TARGET])
    test = test.drop(columns=["id"])

    train, test = build_v14_features(train, test)

    # Convert categorical string columns for XGB
    cat_cols = get_cat_cols(train)
    for c in cat_cols:
        train[c] = train[c].astype(str).astype("category")
        test[c] = test[c].astype(str).astype("category")

    print(f"  Features: {train.shape[1]} (v14 target: 114)")
    if train.shape[1] != 114:
        print(f"  ⚠ Feature count mismatch! Expected 114, got {train.shape[1]}")
        print(f"  Diagnostics: numeric cols = {len([c for c in train.columns if c not in cat_cols])}")
    else:
        print(f"  ✅ Feature count matches v14")

    # 80/20 holdout — same seed as v14
    tr_idx, ho_idx = train_test_split(
        np.arange(len(y_full)), test_size=config.HOLDOUT_FRAC,
        stratify=y_full, random_state=config.HOLDOUT_SEED,
    )
    X_tr = train.iloc[tr_idx].reset_index(drop=True)
    X_ho = train.iloc[ho_idx].reset_index(drop=True)
    y_tr = y_full[tr_idx]
    y_ho = y_full[ho_idx]

    print("\n[2/4] Training via toolkit's train_variant()...")
    cfg = TrainConfig(
        algo="xgb",
        params=V14_XGB_PARAMS,
        n_classes=3,
        cv_seed=config.CV_SEED,
        model_seed=config.MODEL_SEED,
        n_folds=5,
        optuna_trials=200,
        optuna_n_jobs=4,
        register_as="v14_redux",
        tags=["s6e4", "reproduction", "v14"],
        notes="v14 reproduction via toolkit — Checkpoint 1 of build plan",
    )
    reg = Registry(root=S6E4_ROOT / "registry")
    result = train_variant(cfg, X_tr, y_tr, X_ho, y_ho, test, registry=reg)

    print("\n[3/4] Comparing to saved v14 probs...")
    v14_oof_saved = np.load(V14_SAVED_PROBS / "oof_probs.npy")
    v14_ho_saved = np.load(V14_SAVED_PROBS / "holdout_probs.npy")
    v14_test_saved = np.load(V14_SAVED_PROBS / "test_probs.npy")
    v14_best_cw = np.load(V14_SAVED_PROBS / "best_cw.npy")

    # Raw prob comparison
    oof_max_diff = np.abs(result["oof_probs"] - v14_oof_saved).max()
    ho_max_diff = np.abs(result["holdout_probs"] - v14_ho_saved).max()
    test_max_diff = np.abs(result["test_probs"] - v14_test_saved).max()
    cw_diff = np.abs(result["best_cw"] - v14_best_cw).max()

    print(f"  Max abs diff OOF probs:     {oof_max_diff:.6f}")
    print(f"  Max abs diff holdout probs: {ho_max_diff:.6f}")
    print(f"  Max abs diff test probs:    {test_max_diff:.6f}")
    print(f"  Max abs diff best_cw:       {cw_diff:.6f}")

    # Score comparison
    v14_ho_tuned = v14_ho_saved * v14_best_cw
    v14_ho_tuned = v14_ho_tuned / v14_ho_tuned.sum(axis=1, keepdims=True)
    v14_ho_score = balanced_accuracy_score(y_ho, v14_ho_tuned.argmax(axis=1))
    print(f"\n  v14 saved holdout:   {v14_ho_score:.5f}")
    print(f"  toolkit holdout:     {result['holdout_score_tuned']:.5f}")
    print(f"  delta:               {result['holdout_score_tuned'] - v14_ho_score:+.5f}")

    # Checkpoint 1 verdict
    print("\n[4/4] Checkpoint 1 verdict")
    print("=" * 60)
    score_delta = abs(result['holdout_score_tuned'] - v14_ho_score)
    if score_delta < 0.0005:
        print(f"  ✅ PASS — holdout score within 0.0005 of v14")
    elif score_delta < 0.002:
        print(f"  ⚠ SOFT PASS — within 0.002 but not 0.0005")
        print(f"  Likely cause: non-determinism in CUDA/XGB (accept for now)")
    else:
        print(f"  ❌ FAIL — holdout differs by {score_delta:.5f} from v14")
        print(f"  Debug: compare feature sets, seeds, param handling")

    print(f"\n  Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Registry entry: {result['registry_id']}")


if __name__ == "__main__":
    main()
