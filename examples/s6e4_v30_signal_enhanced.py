"""v30 — v14 + top-5 orthogonal signals from signal_factory.

Tests whether genuinely-independent features discovered by signal_factory
produce a real holdout improvement over v14 (0.97423).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from s6e4_v14_reproduction import (
    DATA_RAW, TARGET, TARGET_MAPPING, V14_XGB_PARAMS,
    build_v14_features, S6E4_ROOT,
)
from kaggle_playground_utils import config
from kaggle_playground_utils.features import get_cat_cols
from kaggle_playground_utils.train import TrainConfig, train_variant
from kaggle_playground_utils.registry import Registry


def add_signal_factory_features(train, test):
    """Add top-5 signals discovered by signal_factory."""
    for df in [train, test]:
        # Most orthogonal: sin/cos of Rainfall
        df["sig_sin_Rainfall"] = np.sin(df["Rainfall_mm"].values.astype(np.float64) * 0.1)
        df["sig_cos_Rainfall"] = np.cos(df["Rainfall_mm"].values.astype(np.float64) * 0.1)

        # Distance features — partial orthogonality (~0.4 corr)
        rain_median = train["Rainfall_mm"].median()
        rain_mean = train["Rainfall_mm"].mean()
        df["sig_dist_median_Rainfall"] = np.abs(df["Rainfall_mm"].values - rain_median)
        df["sig_dist_mean_Rainfall"] = np.abs(df["Rainfall_mm"].values - rain_mean)

        # Inverse — higher MI but more correlated
        df["sig_inv_Soil_Moisture"] = 1.0 / (np.abs(df["Soil_Moisture"].values.astype(np.float64)) + 1)

    return train, test


def main():
    print("=" * 60)
    print("v30 — v14 + 5 signal-factory features")
    print("=" * 60)
    t_total = time.time()

    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    y_full = train[TARGET].map(TARGET_MAPPING).values
    train = train.drop(columns=["id", TARGET])
    test = test.drop(columns=["id"])

    print("\n[1/3] Building v14 features + signal additions...")
    train, test = build_v14_features(train, test)
    train, test = add_signal_factory_features(train, test)
    cat_cols = get_cat_cols(train)
    for c in cat_cols:
        train[c] = train[c].astype(str).astype("category")
        test[c] = test[c].astype(str).astype("category")

    EXPECTED = 114 + 5
    assert train.shape[1] == EXPECTED, f"Expected {EXPECTED}, got {train.shape[1]}"
    print(f"  Feature count: {train.shape[1]} (v14=114 + 5 signals)")

    tr_idx, ho_idx = train_test_split(
        np.arange(len(y_full)), test_size=config.HOLDOUT_FRAC,
        stratify=y_full, random_state=config.HOLDOUT_SEED,
    )
    X_tr = train.iloc[tr_idx].reset_index(drop=True)
    X_ho = train.iloc[ho_idx].reset_index(drop=True)
    y_tr = y_full[tr_idx]
    y_ho = y_full[ho_idx]

    print("\n[2/3] Training via toolkit (v14 params + 5 new features)...")
    cfg = TrainConfig(
        algo="xgb",
        params=V14_XGB_PARAMS,
        n_classes=3,
        cv_seed=config.CV_SEED,
        model_seed=config.MODEL_SEED,
        n_folds=5,
        optuna_trials=200,
        optuna_n_jobs=4,
        register_as="v30_signal_enhanced",
        tags=["s6e4", "signal_factory", "xgb"],
        notes="v14 + 5 top signal_factory features",
        verbose=True,
    )
    reg = Registry(root=S6E4_ROOT / "registry")
    result = train_variant(cfg, X_tr, y_tr, X_ho, y_ho, test, registry=reg)

    print("\n[3/3] Summary")
    v14_baseline = 0.97423
    delta = result["holdout_score_tuned"] - v14_baseline
    print(f"  v14 baseline:     {v14_baseline:.5f}")
    print(f"  v30 toolkit:      {result['holdout_score_tuned']:.5f}")
    print(f"  Delta:            {delta:+.5f}")
    if delta > 0.0005:
        print(f"  ✅ IMPROVED — signal factory found a real signal!")
    elif delta > -0.0005:
        print(f"  ⚠ within noise — signals redundant with v14's implicit extraction")
    else:
        print(f"  ❌ WORSE — added noise not signal")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
