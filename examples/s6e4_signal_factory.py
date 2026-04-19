"""Run signal_factory on S6E4 to discover new signals beyond Chris's formula.

For each top-ranked signal, check:
  - MI with target
  - Correlation with Chris's formula features (dedup)
  - Verdict: is this a genuinely new signal?

Then append top 3-5 to v14 feature set and re-train to measure actual holdout delta.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s6e4_v14_reproduction import (
    DATA_RAW, TARGET, TARGET_MAPPING, build_v14_features,
)
from kaggle_playground_utils.features import get_cat_cols
from kaggle_playground_utils.signal_factory import discover_signals


def main():
    print("=" * 60)
    print("Signal Factory on S6E4 — systematic discovery")
    print("=" * 60)

    train = pd.read_csv(DATA_RAW / "train.csv")
    y = train[TARGET].map(TARGET_MAPPING).values
    train_raw = train.drop(columns=["id", TARGET])

    NUMERIC = [
        "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
    ]
    CATEGORICAL = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
                   "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

    # Get v14's existing features (to deduplicate against)
    print("\n[1/3] Building v14 feature set for deduplication baseline...")
    test_stub = pd.read_csv(DATA_RAW / "test.csv").drop(columns=["id"])
    train_fe, _ = build_v14_features(train_raw.copy(), test_stub)

    # Extract existing features as arrays for deduplication
    existing_features = {}
    for c in train_fe.columns:
        if train_fe[c].dtype in [np.int8, np.int64, np.int32, np.float64, np.float32, bool]:
            try:
                existing_features[c] = train_fe[c].values.astype(np.float64)
            except Exception:
                continue

    print(f"  v14 has {len(existing_features)} existing numeric features to dedupe against")

    # Run discovery
    print("\n[2/3] Running signal factory (8 transformation families)...")
    result = discover_signals(
        train_df=train_raw,
        y=y,
        numeric_cols=NUMERIC,
        categorical_cols=CATEGORICAL,
        sample_size=50000,
        top_n=30,
        min_mi=0.005,
        max_corr_with_existing=0.90,
        existing_features=existing_features,
        verbose=True,
    )

    print(f"\n[3/3] Top {len(result)} candidate signals (after dedup vs v14):")
    print(result.to_string(index=False))

    # Save for potential v30 integration
    out_path = Path(__file__).parent.parent / "s6e4_signal_factory_top.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Interpretation
    if len(result) == 0:
        print("\n❌ NO new signals found above noise threshold.")
        print("   This confirms v14's Bayes limit on the current feature set.")
        print("   Chris's formula features already extract everything MI-detectable.")
    else:
        print(f"\n✅ {len(result)} candidate signals with MI > 0.005 and corr < 0.90.")
        top_3 = result.head(3)
        print(f"\nTop 3 to test:")
        for _, row in top_3.iterrows():
            print(f"  {row['feature_name']:<30} MI={row['mi']:.4f}  {row['description']}")
        print(f"\nNext step: add top 3 to v14 features → retrain XGB → measure holdout delta")


if __name__ == "__main__":
    main()
