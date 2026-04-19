"""Tests for train.py — smoke test with synthetic data on CPU."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from kaggle_playground_utils.train import TrainConfig, train_variant
from kaggle_playground_utils.registry import Registry


@pytest.fixture
def tiny_3class():
    """Small 3-class synthetic dataset."""
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=8, n_redundant=2,
        n_classes=3, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    return df, y


def test_xgb_train_variant_smoke(tiny_3class):
    X, y = tiny_3class
    # Split into tr / ho / test
    n = len(y)
    X_tr, y_tr = X.iloc[:400].reset_index(drop=True), y[:400]
    X_ho, y_ho = X.iloc[400:500].reset_index(drop=True), y[400:500]
    X_test = X.iloc[500:].reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(root=tmp)
        cfg = TrainConfig(
            algo="xgb",
            params=dict(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                objective="multi:softprob", num_class=3,
                tree_method="hist", eval_metric="mlogloss",
                random_state=42,
            ),
            n_folds=3,
            optuna_trials=10,
            optuna_n_jobs=1,
            verbose=False,
        )
        result = train_variant(cfg, X_tr, y_tr, X_ho, y_ho, X_test, registry=reg)

    assert result is not None
    assert result["oof_probs"].shape == (400, 3)
    assert result["holdout_probs"].shape == (100, 3)
    assert result["test_probs"].shape == (100, 3)
    assert len(result["fold_scores"]) == 3
    assert 0.0 <= result["oof_score_tuned"] <= 1.0
    assert 0.0 <= result["holdout_score_tuned"] <= 1.0
    # Probabilities sum to 1
    assert np.allclose(result["oof_probs_tuned"].sum(axis=1), 1.0, atol=1e-6)


def test_mini_test_abort(tiny_3class):
    """Mini-test with unrealistically high threshold should abort."""
    X, y = tiny_3class
    cfg = TrainConfig(
        algo="xgb",
        params=dict(
            n_estimators=20, max_depth=2, learning_rate=0.1,
            objective="multi:softprob", num_class=3,
            tree_method="hist", random_state=42,
        ),
        n_folds=3,
        optuna_trials=5,
        mini_test_threshold=0.99,   # unrealistic
        verbose=False,
    )
    result = train_variant(cfg, X, y)
    assert result is None
