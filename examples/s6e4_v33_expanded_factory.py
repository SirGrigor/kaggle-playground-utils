"""v33 — expanded signal factory on v30 features.

Expansions over round 1:
  - Arithmetic: +cube, +inv_sq, +pow03/07, +log2, +exp_small
  - Thresholds: 100 percentiles (was 30)
  - Mod: 13 divisors up to 100 (was 4)
  - Exotic: 7 frequencies for sin/cos (was 1)
  - NEW: cluster_distance (distance to KMeans centroids at k=3,5,8)

Total candidates expected: ~3500 (vs 649 in round 1).
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
from s6e4_v30_signal_enhanced import add_signal_factory_features
from kaggle_playground_utils import config
from kaggle_playground_utils.features import get_cat_cols
from kaggle_playground_utils.train import TrainConfig, train_variant
from kaggle_playground_utils.registry import Registry
from kaggle_playground_utils.signal_factory import (
    discover_signals,
    _build_arithmetic, _build_binning, _build_thresholds, _build_pairwise,
    _build_rank, _build_distance, _build_mod, _build_exotic, _build_cluster_distance,
)


BUILDERS_BY_FAMILY = {
    "arithmetic": _build_arithmetic,
    "binning": _build_binning,
    "thresholds": _build_thresholds,
    "pairwise": _build_pairwise,
    "rank": _build_rank,
    "distance": _build_distance,
    "mod": _build_mod,
    "exotic": _build_exotic,
    "cluster_distance": _build_cluster_distance,
}


def main():
    print("=" * 60)
    print("v33 — Expanded signal factory on v30 features")
    print("=" * 60)
    t_total = time.time()

    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    y = train[TARGET].map(TARGET_MAPPING).values
    train_raw = train.drop(columns=["id", TARGET])
    test_raw = test.drop(columns=["id"])

    print("\n[1/4] Building v30 features as baseline for dedup...")
    train_v30, test_v30 = build_v14_features(train_raw, test_raw)
    train_v30, test_v30 = add_signal_factory_features(train_v30, test_v30)

    existing_features = {}
    for c in train_v30.columns:
        if train_v30[c].dtype in [np.int8, np.int64, np.int32, np.float64, np.float32, bool]:
            try:
                existing_features[c] = train_v30[c].values.astype(np.float64)
            except Exception:
                continue
    print(f"  v30 has {len(existing_features)} numeric features to dedupe against")

    NUMERIC = [
        "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
    ]
    CATEGORICAL = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
                   "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

    print("\n[2/4] Running EXPANDED signal factory...")
    result = discover_signals(
        train_df=train_raw,
        y=y,
        numeric_cols=NUMERIC,
        categorical_cols=CATEGORICAL,
        sample_size=50000,
        top_n=30,
        min_mi=0.005,
        max_corr_with_existing=0.80,  # stricter
        existing_features=existing_features,
        verbose=True,
    )

    print(f"\n  Top {len(result)} candidates (after dedup vs v30):")
    print(result.to_string(index=False))

    if len(result) == 0:
        print("\n  ❌ NO new signals. Skip v33.")
        return

    # Top 7 by orthogonality-weighted MI
    result["score"] = result["mi"] * (1 - result["max_corr_with_existing"])
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    top_n = min(7, len(result))
    top = result.head(top_n)
    print(f"\n  Top {top_n} by orthogonality-weighted MI:")
    print(top[["feature_name", "mi", "max_corr_with_existing", "family", "description"]].to_string(index=False))

    # Add top features on FULL train and test
    print(f"\n[3/4] Adding top {top_n} to v30 features + training v33...")
    train_v33 = train_v30.copy()
    test_v33 = test_v30.copy()
    # Generate builders per family (batch per family for efficiency)
    families_needed = top["family"].unique().tolist()
    for fam in families_needed:
        builder = BUILDERS_BY_FAMILY.get(fam)
        if builder is None:
            continue
        tr_sigs = builder(train_raw, NUMERIC)
        te_sigs = builder(test_raw, NUMERIC)
        for _, row in top[top["family"] == fam].iterrows():
            feat_name = row["feature_name"]
            if feat_name in tr_sigs:
                train_v33[feat_name] = tr_sigs[feat_name][0]
                test_v33[feat_name] = te_sigs[feat_name][0]
                print(f"  ✅ Added: {feat_name} (MI={row['mi']:.4f}, corr={row['max_corr_with_existing']:.3f})")

    cat_cols = get_cat_cols(train_v33)
    for c in cat_cols:
        train_v33[c] = train_v33[c].astype(str).astype("category")
        test_v33[c] = test_v33[c].astype(str).astype("category")

    print(f"\n  v33 feature count: {train_v33.shape[1]} ({train_v33.shape[1] - train_v30.shape[1]} new)")

    tr_idx, ho_idx = train_test_split(
        np.arange(len(y)), test_size=config.HOLDOUT_FRAC,
        stratify=y, random_state=config.HOLDOUT_SEED,
    )
    X_tr = train_v33.iloc[tr_idx].reset_index(drop=True)
    X_ho = train_v33.iloc[ho_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_ho = y[ho_idx]

    cfg = TrainConfig(
        algo="xgb", params=V14_XGB_PARAMS, n_classes=3,
        cv_seed=config.CV_SEED, model_seed=config.MODEL_SEED, n_folds=5,
        optuna_trials=200, optuna_n_jobs=4,
        register_as="v33_expanded_factory",
        tags=["s6e4", "expanded_factory", "xgb"],
        notes=f"v30 + top {top_n} expanded-factory signals",
        verbose=True,
    )
    reg = Registry(root=S6E4_ROOT / "registry")
    result_v33 = train_variant(cfg, X_tr, y_tr, X_ho, y_ho, test_v33, registry=reg)

    print("\n[4/4] Summary")
    v14_baseline = 0.97423
    v30_holdout = 0.97479
    v30_lb = 0.97654
    delta_vs_v30 = result_v33["holdout_score_tuned"] - v30_holdout
    print(f"  v14 baseline:    0.97423")
    print(f"  v30 holdout:     0.97479 (LB 0.97654)")
    print(f"  v33 holdout:     {result_v33['holdout_score_tuned']:.5f}")
    print(f"  Delta vs v30:    {delta_vs_v30:+.5f}")
    if delta_vs_v30 > 0.0005:
        print(f"  ✅ IMPROVED — expanded factory found new signals!")
    elif delta_vs_v30 > -0.0005:
        print(f"  ⚠ within noise — factory saturated")
    else:
        print(f"  ❌ WORSE")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
