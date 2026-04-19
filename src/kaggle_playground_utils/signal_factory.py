"""signal_factory.py — systematic automated signal discovery.

The 5 rounds of S6E4 probes we ran tested ~30 hypotheses. This module tests
hundreds automatically via the following transformation families:

  1. Arithmetic:    log, sqrt, square, reciprocal, abs, log-diff
  2. Binning:       quantile bins at multiple resolutions
  3. Thresholds:    100 candidate cutoffs per numeric
  4. Pairwise:      products, ratios, diffs, signed-diffs
  5. Rank:          percentile rank, global rank
  6. Distance:      from mean/median/mode
  7. Hash/mod:      mod-N features (generator artifact detection)
  8. Exotic:        sin/cos (periodic), interaction with quantile bins
  9. Cross-cat:     categorical × numeric target-encoded

Usage:
    from kaggle_playground_utils.signal_factory import discover_signals

    result = discover_signals(
        train_df=train,
        y=y,
        numeric_cols=NUMERIC,
        categorical_cols=CATEGORICAL,
        sample_size=50000,
        top_n=20,
    )

    # result is a DataFrame with columns:
    #   feature_name, mi, description, family
    #
    # Feed top-N back into your feature engineering pipeline:
    for row in result.head(10).itertuples():
        # add row.feature_name using row.description
        ...
"""
from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ============================================================================
# Transformation builders — each returns dict {name: (array, description)}
# ============================================================================

def _build_arithmetic(df: pd.DataFrame, numeric: list[str]) -> dict:
    out = {}
    for c in numeric:
        v = df[c].values.astype(np.float64)
        out[f"log1p_{c}"] = (np.log1p(np.abs(v)), f"log(1 + |{c}|)")
        out[f"sqrt_{c}"] = (np.sqrt(np.abs(v)), f"sqrt(|{c}|)")
        out[f"square_{c}"] = (v ** 2, f"{c}²")
        out[f"inv_{c}"] = (1.0 / (np.abs(v) + 1), f"1 / (|{c}| + 1)")
    return out


def _build_binning(df: pd.DataFrame, numeric: list[str]) -> dict:
    out = {}
    for c in numeric:
        v = df[c].values
        for n_bins in [5, 10, 20, 50]:
            try:
                bins = np.quantile(v, np.linspace(0, 1, n_bins + 1))
                bins = np.unique(bins)
                if len(bins) < 3:
                    continue
                binned = np.digitize(v, bins[1:-1])
                out[f"qbin{n_bins}_{c}"] = (binned, f"{c} quantile-binned ({n_bins})")
            except Exception:
                continue
    return out


def _build_thresholds(df: pd.DataFrame, numeric: list[str],
                      n_scan: int = 30) -> dict:
    """Threshold scan: for each numeric, test cutoffs at percentiles 5-95."""
    out = {}
    for c in numeric:
        v = df[c].values
        percentiles = np.linspace(5, 95, n_scan)
        thresholds = np.quantile(v, percentiles / 100)
        for pct, thr in zip(percentiles, thresholds):
            out[f"lt_{c}_p{int(pct)}"] = ((v < thr).astype(int), f"{c} < {thr:.3f} (p{int(pct)})")
    return out


def _build_pairwise(df: pd.DataFrame, numeric: list[str],
                    max_pairs: int = 30) -> dict:
    """Pairwise products, ratios, and differences. Limited to top pairs."""
    out = {}
    cols = numeric[:min(len(numeric), 8)]  # cap to keep combinatorial small
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            v1 = df[c1].values.astype(np.float64)
            v2 = df[c2].values.astype(np.float64)
            out[f"prod_{c1}_x_{c2}"] = (v1 * v2, f"{c1} × {c2}")
            out[f"ratio_{c1}_by_{c2}"] = (v1 / (np.abs(v2) + 0.1), f"{c1} / {c2}")
            out[f"diff_{c1}_{c2}"] = (v1 - v2, f"{c1} − {c2}")
            out[f"absdiff_{c1}_{c2}"] = (np.abs(v1 - v2), f"|{c1} − {c2}|")
    return out


def _build_rank(df: pd.DataFrame, numeric: list[str]) -> dict:
    out = {}
    for c in numeric:
        v = df[c].values
        out[f"rankpct_{c}"] = (pd.Series(v).rank(pct=True).values, f"pct-rank of {c}")
    return out


def _build_distance(df: pd.DataFrame, numeric: list[str]) -> dict:
    out = {}
    for c in numeric:
        v = df[c].values.astype(np.float64)
        out[f"dist_mean_{c}"] = (np.abs(v - v.mean()), f"|{c} − mean({c})|")
        out[f"dist_median_{c}"] = (np.abs(v - np.median(v)), f"|{c} − median({c})|")
    return out


def _build_mod(df: pd.DataFrame, numeric: list[str]) -> dict:
    out = {}
    for c in numeric:
        v_int = (df[c].values * 100).astype(np.int64)
        for m in [3, 5, 7, 13]:
            out[f"mod{m}_{c}"] = (v_int % m, f"int({c}*100) mod {m}")
    return out


def _build_exotic(df: pd.DataFrame, numeric: list[str]) -> dict:
    out = {}
    for c in numeric:
        v = df[c].values.astype(np.float64)
        out[f"sin_{c}"] = (np.sin(v * 0.1), f"sin({c}*0.1)")
        out[f"cos_{c}"] = (np.cos(v * 0.1), f"cos({c}*0.1)")
    return out


def _build_cross_cat(df: pd.DataFrame, numeric: list[str],
                     categorical: list[str]) -> dict:
    """For each (cat, num) pair, compute 'bin-of-num within cat group' (5 bins)."""
    out = {}
    # Cap combinations to keep fast
    num_sub = numeric[:5]
    cat_sub = categorical[:4]
    for cat in cat_sub:
        for num in num_sub:
            try:
                v = df.groupby(cat)[num].rank(pct=True).values
                out[f"rankin_{cat}_{num}"] = (v, f"rank of {num} within {cat}")
            except Exception:
                continue
    return out


TRANSFORMATION_BUILDERS: list[tuple[str, Callable]] = [
    ("arithmetic", _build_arithmetic),
    ("binning", _build_binning),
    ("thresholds", _build_thresholds),
    ("pairwise", _build_pairwise),
    ("rank", _build_rank),
    ("distance", _build_distance),
    ("mod", _build_mod),
    ("exotic", _build_exotic),
]


# ============================================================================
# Signal discovery
# ============================================================================

def discover_signals(
    train_df: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    sample_size: int = 50000,
    families: list[str] | None = None,
    top_n: int = 20,
    min_mi: float = 0.005,
    max_corr_with_existing: float = 0.90,
    existing_features: dict[str, np.ndarray] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full signal factory: test 100s of transformations, rank by MI.

    Returns DataFrame sorted by descending MI with columns:
        feature_name, mi, description, family, max_corr_with_existing

    Args:
        train_df: training data with numeric + categorical columns
        y: target vector (integer-encoded)
        numeric_cols: default = all numeric in train_df
        categorical_cols: default = all categorical in train_df
        sample_size: MI computation uses a random sample for speed
        families: subset of TRANSFORMATION_BUILDERS to run (None = all)
        top_n: return top-N by MI
        min_mi: drop features with MI below this
        max_corr_with_existing: drop features correlated above this with existing ones
        existing_features: dict of feature_name -> array for deduplication
    """
    if numeric_cols is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = train_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    selected_builders = TRANSFORMATION_BUILDERS
    if families is not None:
        selected_builders = [(n, f) for n, f in TRANSFORMATION_BUILDERS if n in families]

    # Sample for MI speed (keep indices for dedup alignment)
    n = len(train_df)
    if sample_size < n:
        idx = np.random.RandomState(42).choice(n, sample_size, replace=False)
        df_sample = train_df.iloc[idx].reset_index(drop=True)
        y_sample = y[idx]
    else:
        idx = np.arange(n)
        df_sample = train_df
        y_sample = y

    # Generate all candidate features
    if verbose:
        print(f"Generating candidate transformations from {len(numeric_cols)} numeric + "
              f"{len(categorical_cols)} categorical cols...")

    all_candidates = {}
    for family_name, builder in selected_builders:
        t0 = time.time()
        if family_name == "cross_cat":
            sig = builder(df_sample, numeric_cols, categorical_cols)
        elif family_name in ("thresholds", "pairwise"):
            sig = builder(df_sample, numeric_cols)
        else:
            sig = builder(df_sample, numeric_cols)
        for name, (arr, desc) in sig.items():
            all_candidates[name] = {"array": arr, "description": desc, "family": family_name}
        if verbose:
            print(f"  {family_name}: {len(sig)} candidates ({time.time()-t0:.1f}s)")

    # Include cross_cat explicitly since it's not in the default builders loop
    if categorical_cols and "cross_cat" in (families or []) + ["cross_cat"]:
        cc = _build_cross_cat(df_sample, numeric_cols, categorical_cols)
        for name, (arr, desc) in cc.items():
            all_candidates[name] = {"array": arr, "description": desc, "family": "cross_cat"}
        if verbose:
            print(f"  cross_cat: {len(cc)} candidates")

    if verbose:
        print(f"\nTotal candidates: {len(all_candidates)}")
        print(f"Computing mutual information with target...")

    # Batch MI computation
    feat_names = list(all_candidates.keys())
    X_candidates = np.column_stack([all_candidates[n]["array"] for n in feat_names])

    t0 = time.time()
    mi_scores = mutual_info_classif(X_candidates, y_sample, random_state=42, n_jobs=-1)
    if verbose:
        print(f"  MI computation: {time.time()-t0:.1f}s")

    # Filter by min MI
    records = []
    for name, mi in zip(feat_names, mi_scores):
        if mi < min_mi:
            continue
        records.append({
            "feature_name": name,
            "mi": float(mi),
            "description": all_candidates[name]["description"],
            "family": all_candidates[name]["family"],
            "max_corr_with_existing": 0.0,
        })

    if not records:
        return pd.DataFrame(records)

    # Deduplicate against existing features via Pearson corr
    # CRITICAL: use the SAME sample indices for both candidates and existing features
    if existing_features:
        # existing_features are full-size arrays; sample them with the same idx
        existing_names = list(existing_features.keys())
        existing_sample = {}
        for name, arr in existing_features.items():
            if len(arr) == n:
                existing_sample[name] = arr[idx]
            elif len(arr) == sample_size:
                existing_sample[name] = arr  # already sampled, assume same idx
            else:
                # mismatched size — skip this one
                continue

        for r in records:
            candidate_arr = all_candidates[r["feature_name"]]["array"]
            max_corr = 0.0
            for ename, earr in existing_sample.items():
                try:
                    c = abs(np.corrcoef(candidate_arr, earr)[0, 1])
                    if np.isnan(c):
                        continue
                    if c > max_corr:
                        max_corr = c
                except Exception:
                    continue
            r["max_corr_with_existing"] = float(max_corr)

        # Filter out overly-correlated
        records = [r for r in records if r["max_corr_with_existing"] < max_corr_with_existing]

    result = pd.DataFrame(records)
    result = result.sort_values("mi", ascending=False).reset_index(drop=True)
    return result.head(top_n)
