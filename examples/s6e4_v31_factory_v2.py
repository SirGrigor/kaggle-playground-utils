"""v31 — Signal factory round 2 with v30 features as the new baseline.

Run signal_factory deduping against v30's 119 features (v14's 114 + 5 from round 1).
Add top 5 newly-discovered orthogonal signals, retrain.

Tests whether the factory can keep finding signals iteratively.
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
from kaggle_playground_utils.signal_factory import discover_signals


def main():
    print("=" * 60)
    print("v31 — Signal Factory Round 2 (dedup vs v30's 119 features)")
    print("=" * 60)
    t_total = time.time()

    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    y = train[TARGET].map(TARGET_MAPPING).values
    train_raw = train.drop(columns=["id", TARGET])
    test_raw = test.drop(columns=["id"])

    # Build v30 features (v14 + round-1 signals)
    print("\n[1/4] Building v30 features as new baseline...")
    train_v30, test_v30 = build_v14_features(train_raw.copy(), test_raw.copy())
    train_v30, test_v30 = add_signal_factory_features(train_v30, test_v30)
    print(f"  v30 has {train_v30.shape[1]} features (114 v14 + 5 round-1)")

    # Prepare existing_features for dedup (numeric only)
    existing_features = {}
    for c in train_v30.columns:
        if train_v30[c].dtype in [np.int8, np.int64, np.int32, np.float64, np.float32, bool]:
            try:
                existing_features[c] = train_v30[c].values.astype(np.float64)
            except Exception:
                continue
    print(f"  Dedup pool: {len(existing_features)} numeric features")

    # Run factory
    NUMERIC = [
        "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
    ]
    CATEGORICAL = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
                   "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

    print("\n[2/4] Running signal factory (dedup threshold 0.85)...")
    result = discover_signals(
        train_df=train_raw,
        y=y,
        numeric_cols=NUMERIC,
        categorical_cols=CATEGORICAL,
        sample_size=50000,
        top_n=20,
        min_mi=0.005,
        max_corr_with_existing=0.85,  # stricter than v30's 0.90
        existing_features=existing_features,
        verbose=True,
    )

    print(f"\n  Top {len(result)} NEW candidates after dedup vs v30:")
    print(result.to_string(index=False))

    if len(result) == 0:
        print("\n  ❌ NO new signals found. Factory exhausted for this feature base.")
        return

    # Take top 5 by (MI × (1-max_corr)) to favor orthogonal ones
    result["score"] = result["mi"] * (1 - result["max_corr_with_existing"])
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    top5 = result.head(5)
    print(f"\n  Top 5 by orthogonality-weighted MI:")
    print(top5[["feature_name", "mi", "max_corr_with_existing", "description"]].to_string(index=False))

    # Add top 5 as functions — replicate on train + test
    print("\n[3/4] Adding top 5 to v30 features + training v31...")
    from kaggle_playground_utils.signal_factory import (
        _build_arithmetic, _build_binning, _build_thresholds, _build_pairwise,
        _build_rank, _build_distance, _build_mod, _build_exotic,
    )

    # Generate the top 5 features on FULL train and test
    # (signal_factory returned on 50K sample; we need full arrays)
    BUILDERS_BY_FAMILY = {
        "arithmetic": _build_arithmetic,
        "binning": _build_binning,
        "thresholds": _build_thresholds,
        "pairwise": _build_pairwise,
        "rank": _build_rank,
        "distance": _build_distance,
        "mod": _build_mod,
        "exotic": _build_exotic,
    }

    train_v31 = train_v30.copy()
    test_v31 = test_v30.copy()
    for _, row in top5.iterrows():
        feat_name = row["feature_name"]
        family = row["family"]
        builder = BUILDERS_BY_FAMILY.get(family)
        if builder is None:
            print(f"  ⚠ No builder for family '{family}', skipping {feat_name}")
            continue
        # Regenerate on full train + test
        tr_sigs = builder(train_raw, NUMERIC)
        te_sigs = builder(test_raw, NUMERIC)
        if feat_name in tr_sigs:
            train_v31[feat_name] = tr_sigs[feat_name][0]
            test_v31[feat_name] = te_sigs[feat_name][0]
            print(f"  ✅ Added: {feat_name} (MI={row['mi']:.4f})")
        else:
            print(f"  ⚠ {feat_name} not in builder output — skipping")

    cat_cols = get_cat_cols(train_v31)
    for c in cat_cols:
        train_v31[c] = train_v31[c].astype(str).astype("category")
        test_v31[c] = test_v31[c].astype(str).astype("category")

    print(f"  v31 feature count: {train_v31.shape[1]} ({train_v31.shape[1] - train_v30.shape[1]} new)")

    tr_idx, ho_idx = train_test_split(
        np.arange(len(y)), test_size=config.HOLDOUT_FRAC,
        stratify=y, random_state=config.HOLDOUT_SEED,
    )
    X_tr = train_v31.iloc[tr_idx].reset_index(drop=True)
    X_ho = train_v31.iloc[ho_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_ho = y[ho_idx]

    cfg = TrainConfig(
        algo="xgb",
        params=V14_XGB_PARAMS,
        n_classes=3,
        cv_seed=config.CV_SEED,
        model_seed=config.MODEL_SEED,
        n_folds=5,
        optuna_trials=200,
        optuna_n_jobs=4,
        register_as="v31_factory_v2",
        tags=["s6e4", "signal_factory_v2", "xgb"],
        notes="v30 + top 5 round-2 orthogonal signals",
        verbose=True,
    )
    reg = Registry(root=S6E4_ROOT / "registry")
    v31_result = train_variant(cfg, X_tr, y_tr, X_ho, y_ho, test_v31, registry=reg)

    print("\n[4/4] Summary")
    v14_baseline = 0.97423
    v30_holdout = 0.97479
    v30_lb = 0.97654
    delta_vs_v14 = v31_result["holdout_score_tuned"] - v14_baseline
    delta_vs_v30 = v31_result["holdout_score_tuned"] - v30_holdout
    print(f"  v14 baseline holdout:  {v14_baseline:.5f}")
    print(f"  v30 holdout:           {v30_holdout:.5f}")
    print(f"  v30 LB:                {v30_lb:.5f}")
    print(f"  v31 holdout:           {v31_result['holdout_score_tuned']:.5f}")
    print(f"  Delta vs v14:          {delta_vs_v14:+.5f}")
    print(f"  Delta vs v30:          {delta_vs_v30:+.5f}")
    if delta_vs_v30 > 0.0005:
        print(f"  ✅ IMPROVED — iterative signal discovery works!")
    elif delta_vs_v30 > -0.0005:
        print(f"  ⚠ within noise — factory may be exhausted")
    else:
        print(f"  ❌ WORSE — round-2 added noise not signal")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
