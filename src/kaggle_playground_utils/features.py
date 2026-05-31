"""features.py — reusable feature engineering helpers.

Target (Day 2):
  - digit_features(df, num_cols, k_range) — precision-safe digit extraction (L10)
  - threshold_booleans(df, thresholds)
  - categorical_one_hot(df, cols)
  - formula_logits(df, coefficients) — Chris Deotte-style logit injection
  - get_cat_cols(df) — pandas-version-safe categorical detection
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def digit_features(df: pd.DataFrame, num_cols: list[str], k_range=range(-4, 4)) -> pd.DataFrame:
    """Precision-safe digit extraction (FLOOR with exact-integer shift).

    For each numeric column and each k in k_range, adds column `{c}_digit{k}` equal
    to the digit at the 10^k decimal place.

    The naive `v // (10**k) % 10` fails for k<0 because 10^k is not exactly
    representable in float64 (e.g., 0.01 = 0.010000000000000000208...).
    This implementation multiplies/divides by positive integers (exact) then floors.

    See L10 in knowledge-graph/kaggle/training-configuration-lessons.md.
    """
    df = df.copy()
    for c in num_cols:
        v = df[c].values.astype(np.float64)
        for k in k_range:
            if k < 0:
                shifted = v * (10 ** (-k))
            elif k > 0:
                shifted = v / (10 ** k)
            else:
                shifted = v
            df[f"{c}_digit{k}"] = (np.floor(shifted).astype(np.int64) % 10).astype("int8")
    return df


def threshold_booleans(df: pd.DataFrame, thresholds: dict[str, float],
                       direction: dict[str, str] | None = None) -> pd.DataFrame:
    """Add binary threshold features.

    Args:
        thresholds: {column: threshold_value}
        direction: {column: "lt"|"gt"|"le"|"ge"} — default "lt" for all.
                   Produces column named `{col}_{direction}_{int(threshold)}` or similar.

    Note: caller controls exact column naming via return; for now we use
    f"{col}_thresh_{direction}_{threshold}" to be explicit.
    """
    df = df.copy()
    direction = direction or {}
    for col, thr in thresholds.items():
        d = direction.get(col, "lt")
        if d == "lt":
            df[f"{col}_lt_{thr}"] = (df[col] < thr).astype(int)
        elif d == "gt":
            df[f"{col}_gt_{thr}"] = (df[col] > thr).astype(int)
        elif d == "le":
            df[f"{col}_le_{thr}"] = (df[col] <= thr).astype(int)
        elif d == "ge":
            df[f"{col}_ge_{thr}"] = (df[col] >= thr).astype(int)
    return df


def get_cat_cols(df: pd.DataFrame) -> list[str]:
    """Pandas-version-safe categorical column detection.

    Pandas 4 uses `str` dtype; pandas 2 uses `object`. This handles both.
    """
    return df.select_dtypes(include=["object", "string", "str", "category"]).columns.tolist()


def categorical_one_hot(df: pd.DataFrame, col: str, values: list[str],
                        prefix: str | None = None) -> pd.DataFrame:
    """Add explicit one-hot columns for given categorical values.

    Unlike pd.get_dummies, this produces deterministic column names even if
    some values are missing from the data. Useful for matching columns
    between train and test.
    """
    df = df.copy()
    prefix = prefix or col
    for v in values:
        df[f"{prefix}_{v}"] = (df[col] == v).astype(int)
    return df


def formula_logits(df: pd.DataFrame, coefs: dict[str, dict[str, float]],
                   feature_cols: list[str]) -> pd.DataFrame:
    """Apply a pre-fit LR formula (per class) as logit features.

    Args:
        coefs: {class_name: {feature: coef, "intercept": value}}
        feature_cols: names of columns that match coefs' feature keys
                      (caller ensures they exist in df, typically after
                      threshold_booleans + categorical_one_hot)

    Returns df with new columns `logit_{class_name}`.

    Pattern: used to inject Chris Deotte-style formula recovery.
    """
    df = df.copy()
    for cls_name, class_coefs in coefs.items():
        col = f"logit_{cls_name}"
        df[col] = class_coefs.get("intercept", 0.0)
        for feat in feature_cols:
            if feat in class_coefs:
                df[col] = df[col] + class_coefs[feat] * df[feat]
    return df


def safe_label_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    min_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frequency-thresholded label encoding (fit on train only).

    Values whose train count < `min_count` are folded into a shared "other"
    bucket. Test values unseen in train (or rare) also map to "other". Returns
    (train_encoded, test_encoded) with dense int codes per column. Code 0 is
    always reserved for the "other"/unseen bucket; real categories start at 1.

    Leakage-safe: the kept-value vocabulary and codes are derived from train
    only and applied to test.
    """
    train_enc = pd.DataFrame(index=train_df.index)
    test_enc = pd.DataFrame(index=test_df.index)
    for c in cols:
        counts = train_df[c].value_counts()
        kept = counts[counts >= min_count].index
        mapping = {v: i + 1 for i, v in enumerate(sorted(kept, key=lambda x: str(x)))}
        train_enc[c] = train_df[c].map(mapping).fillna(0).astype("int32")
        test_enc[c] = test_df[c].map(mapping).fillna(0).astype("int32")
    return train_enc, test_enc


def decimal_round_by_magnitude(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Round numeric columns by magnitude of their max absolute value.

    3 dp if max(|col|) < 10, 2 dp if < 100, else 1 dp. Synthetic generators
    often emit values at a fixed precision; trailing digits are noise. Returns a
    copy of `df` with the listed columns rounded.
    """
    df = df.copy()
    for c in num_cols:
        m = float(np.abs(df[c].to_numpy(dtype=np.float64)).max())
        ndp = 3 if m < 10 else (2 if m < 100 else 1)
        df[c] = df[c].round(ndp)
    return df


def drop_uniform_in_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop columns that are constant (nunique==1) in `test_df`.

    Such columns carry no discriminative signal at inference time. Only columns
    present in test are checked; columns unique to train are retained. Returns
    (train_kept, test_kept, kept_cols).
    """
    drop = [c for c in test_df.columns if test_df[c].nunique(dropna=False) == 1]
    kept = [c for c in train_df.columns if c not in drop]
    test_kept = [c for c in test_df.columns if c not in drop]
    return train_df[kept], test_df[test_kept], kept
