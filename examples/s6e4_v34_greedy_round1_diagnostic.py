"""v34 DIAGNOSTIC — greedy round 1 only on top 30 candidates vs v30 base.

Purpose: cheap probe before committing to full greedy run.
  - If >=1 candidate has mini-test lift > 0.0005 → full greedy worth running
  - If all lifts <= threshold → factory exhausted vs v30, skip

Time: ~30 min (factory ~5min + 30 mini-tests at ~1min each).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from s6e4_v14_reproduction import (
    DATA_RAW, TARGET, TARGET_MAPPING, V14_XGB_PARAMS,
    build_v14_features, S6E4_ROOT,
)
from s6e4_v30_signal_enhanced import add_signal_factory_features
from kaggle_playground_utils import config
from kaggle_playground_utils.features import get_cat_cols
from kaggle_playground_utils.signal_factory import (
    discover_signals,
    _build_arithmetic, _build_binning, _build_thresholds, _build_pairwise,
    _build_rank, _build_distance, _build_mod, _build_exotic, _build_cluster_distance,
)
from kaggle_playground_utils.greedy_selection import greedy_forward_selection


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
    t_total = time.time()
    print("=" * 60)
    print("v34 DIAGNOSTIC — greedy round 1 on top 30 candidates")
    print("=" * 60)

    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    y = train[TARGET].map(TARGET_MAPPING).values
    train_raw = train.drop(columns=["id", TARGET])
    test_raw = test.drop(columns=["id"])

    print("\n[1/4] Building v30 baseline features...")
    train_v30, test_v30 = build_v14_features(train_raw.copy(), test_raw.copy())
    train_v30, test_v30 = add_signal_factory_features(train_v30, test_v30)

    existing_features = {}
    for c in train_v30.columns:
        if train_v30[c].dtype in [np.int8, np.int64, np.int32,
                                  np.float64, np.float32, bool]:
            try:
                existing_features[c] = train_v30[c].values.astype(np.float64)
            except Exception:
                continue
    print(f"  v30 numeric features (dedup pool): {len(existing_features)}")

    NUMERIC = [
        "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
        "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
    ]
    CATEGORICAL = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
                   "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

    print("\n[2/4] Running signal factory (expanded, dedup vs v30)...")
    result = discover_signals(
        train_df=train_raw,
        y=y,
        numeric_cols=NUMERIC,
        categorical_cols=CATEGORICAL,
        sample_size=50000,
        top_n=30,
        min_mi=0.005,
        max_corr_with_existing=0.80,
        existing_features=existing_features,
        verbose=True,
    )

    if len(result) == 0:
        print("\n  ❌ factory returned 0 candidates — nothing to test")
        return

    n_cand = min(30, len(result))
    top = result.head(n_cand).reset_index(drop=True)
    print(f"\n  {n_cand} candidates after dedup:")
    print(top[["feature_name", "mi", "max_corr_with_existing",
               "family"]].to_string(index=False))

    print(f"\n[3/4] Regenerating {n_cand} candidates on full train ...")
    families = top["family"].unique().tolist()
    family_sigs = {}
    for fam in families:
        builder = BUILDERS_BY_FAMILY.get(fam)
        if builder is None:
            print(f"  ⚠ no builder for family '{fam}'")
            continue
        family_sigs[fam] = builder(train_raw, NUMERIC)

    candidates = {}
    for _, row in top.iterrows():
        fam = row["family"]
        name = row["feature_name"]
        sigs = family_sigs.get(fam, {})
        if name in sigs:
            candidates[name] = sigs[name][0]
        else:
            print(f"  ⚠ {name} not in {fam} builder output — skipped")
    print(f"  Regenerated {len(candidates)} candidate arrays "
          f"(length {len(next(iter(candidates.values())))})")

    tr_idx, ho_idx = train_test_split(
        np.arange(len(y)), test_size=config.HOLDOUT_FRAC,
        stratify=y, random_state=config.HOLDOUT_SEED,
    )
    X_tr_v30 = train_v30.iloc[tr_idx].reset_index(drop=True)
    X_ho_v30 = train_v30.iloc[ho_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_ho = y[ho_idx]

    cat_cols = get_cat_cols(X_tr_v30)
    for c in cat_cols:
        X_tr_v30[c] = X_tr_v30[c].astype(str).astype("category")
        X_ho_v30[c] = X_ho_v30[c].astype(str).astype("category")

    candidates_tr = {k: v[tr_idx] for k, v in candidates.items()}
    base_cols = X_tr_v30.columns.tolist()

    print(f"\n[4/4] Mini-test each candidate (max_rounds=1) ...")
    selected, log = greedy_forward_selection(
        X_tr=X_tr_v30, y_tr=y_tr,
        X_ho=X_ho_v30, y_ho=y_ho,
        base_cols=base_cols,
        candidates=candidates_tr,
        algo="xgb",
        params=V14_XGB_PARAMS,
        lift_threshold=0.0005,
        max_rounds=1,
        sample_frac=0.20,
        verbose=True,
    )

    round1 = log.iloc[0]
    all_lifts = round1["all_lifts"]
    lifts_df = pd.DataFrame([
        {"candidate": k, "lift": v}
        for k, v in sorted(all_lifts.items(), key=lambda x: -x[1])
    ])
    out_dir = S6E4_ROOT / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v34_round1_lifts.csv"
    lifts_df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULT")
    print("=" * 60)
    print(lifts_df.to_string(index=False))
    print(f"\n  Candidates with lift > 0:        "
          f"{(lifts_df['lift'] > 0).sum()} / {len(lifts_df)}")
    print(f"  Candidates with lift > 0.0005:   "
          f"{(lifts_df['lift'] > 0.0005).sum()}")
    print(f"  Candidates with lift > 0.001:    "
          f"{(lifts_df['lift'] > 0.001).sum()}")
    print(f"  Candidates with lift > 0.002:    "
          f"{(lifts_df['lift'] > 0.002).sum()}")
    if len(lifts_df):
        best = lifts_df.iloc[0]
        print(f"  Best: {best['candidate']}  lift={best['lift']:+.5f}")

    if lifts_df["lift"].max() > 0.0005:
        print("\n  ✅ At least one candidate crosses threshold "
              "→ full greedy worth running")
    else:
        print("\n  ❌ No candidate crosses threshold "
              "→ factory exhausted vs v30; try different strategy")

    print(f"\n  Saved: {out_path}")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
