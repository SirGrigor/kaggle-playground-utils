"""Leakage-free feature encoders for tabular ML.

Every encoder follows the fit_transform-style API but splits the work into:
  - (oof_encoding, test_encoding) = encode(train, test, target, col, ...)

This keeps leakage visible at the call site. The encoders refuse to compute
"global" target-aware features on train without folds.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


# ------------------------------------------------------------------------
#  target encoding (the grandmaster move)
# ------------------------------------------------------------------------

def kfold_target_encode(
    train: pd.DataFrame,
    test: Optional[pd.DataFrame],
    col: str,
    target: str,
    n_splits: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
    stratified: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """KFold target encoding with Bayesian smoothing.

    For train: each row's encoding is computed from rows in OTHER folds (leakage-free).
    For test: encoding uses full train data (safe — test labels aren't used).

    Smoothing formula: encoding = (count * category_mean + smoothing * global_mean) / (count + smoothing)
    Higher smoothing = rare categories pulled more toward the global mean.

    Args:
        train: DataFrame containing `col` and `target`.
        test: DataFrame containing `col` (or None if only encoding train).
        col: categorical column name to encode.
        target: target column name in train (int/binary or continuous).
        n_splits: number of folds.
        smoothing: smoothing weight (0 = no smoothing, inf = all global mean).
        random_state: seed for fold splits.
        stratified: use StratifiedKFold (good for classification) vs KFold.

    Returns:
        (oof_encoded, test_encoded) — numpy arrays.
    """
    y = train[target].values
    if stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(train, y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(train)

    global_mean = float(np.mean(y))
    oof = np.zeros(len(train))

    for tr_idx, va_idx in split_iter:
        fold_train = train.iloc[tr_idx]
        stats = fold_train.groupby(col)[target].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        enc_map = smoothed.to_dict()
        oof[va_idx] = train.iloc[va_idx][col].map(enc_map).fillna(global_mean).values

    test_enc = None
    if test is not None:
        stats = train.groupby(col)[target].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        test_enc = test[col].map(smoothed.to_dict()).fillna(global_mean).values

    return oof, test_enc


def pairwise_concat_target_encode(
    train: pd.DataFrame,
    test: Optional[pd.DataFrame],
    col1: str,
    col2: str,
    target: str,
    n_splits: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Create a pairwise concat feature and target-encode it.

    Returns (oof_encoded, test_encoded, new_col_name).
    """
    new_col = f"{col1}_X_{col2}"
    train_tmp = train.copy()
    train_tmp[new_col] = (train_tmp[col1].astype(str) + "__" + train_tmp[col2].astype(str))
    test_tmp = None
    if test is not None:
        test_tmp = test.copy()
        test_tmp[new_col] = (test_tmp[col1].astype(str) + "__" + test_tmp[col2].astype(str))
    oof, test_enc = kfold_target_encode(
        train_tmp, test_tmp, new_col, target,
        n_splits=n_splits, smoothing=smoothing, random_state=random_state,
    )
    return oof, test_enc, new_col


def quantile_bin(
    train: pd.DataFrame,
    test: Optional[pd.DataFrame],
    col: str,
    n_bins: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Quantile-bin a numeric column. Useful before target encoding for
    non-monotonic numerics."""
    train_bins, bin_edges = pd.qcut(
        train[col], n_bins, labels=False, duplicates="drop", retbins=True
    )
    test_bins = None
    if test is not None:
        # Use train's edges to bin test
        test_bins = pd.cut(
            test[col], bins=bin_edges, labels=False, include_lowest=True
        ).fillna(-1).astype(int).values
    return train_bins.values, test_bins


# ------------------------------------------------------------------------
#  simple derived features (no leakage risk)
# ------------------------------------------------------------------------

def count_flags(
    df: pd.DataFrame,
    cols: Sequence[str],
    positive_value: str = "Yes",
) -> np.ndarray:
    """Count how many of the given categorical columns equal `positive_value`.
    Useful for service bundle count, add-on count, etc."""
    return sum((df[c] == positive_value).astype(int) for c in cols).values


def ratio(
    df: pd.DataFrame, numerator: str, denominator: str, fill: float = 0.0
) -> np.ndarray:
    """Safe numerator/denominator ratio (zero-safe)."""
    return np.where(df[denominator] > 0, df[numerator] / df[denominator].replace(0, np.nan), fill)
