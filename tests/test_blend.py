"""Tests for blend.py — registry-based ensembling."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from kaggle_playground_utils.registry import Registry, ModelRecord
from kaggle_playground_utils.blend import (
    pairwise_correlations, simple_average, nelder_mead_blend, stacking_meta,
)


@pytest.fixture
def registry_with_3_models():
    """Registry with 3 synthetic models of varying quality + correlation."""
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(root=tmp)
        n_oof, n_ho, n_test, n_cls = 500, 100, 100, 3
        rng = np.random.RandomState(42)

        # True labels (for OOF)
        y_oof = rng.randint(0, n_cls, n_oof)
        y_ho = rng.randint(0, n_cls, n_ho)

        def make_probs(bias_toward_truth: float, shape):
            """Mock probs: weighted mix of true-labels one-hot + uniform noise."""
            if len(shape) == 2:
                noise = rng.dirichlet([1] * n_cls, shape[0])
            return noise

        for i, (name, bias) in enumerate([("modelA", 0.8), ("modelB", 0.7), ("modelC", 0.6)]):
            # Make OOF probs that actually correlate with y_oof
            oof = np.zeros((n_oof, n_cls)) + 0.1
            oof[np.arange(n_oof), y_oof] = 0.8 - i * 0.1
            oof += rng.uniform(0, 0.05, oof.shape)
            oof /= oof.sum(axis=1, keepdims=True)

            ho = np.zeros((n_ho, n_cls)) + 0.1
            ho[np.arange(n_ho), y_ho] = 0.8 - i * 0.1
            ho += rng.uniform(0, 0.05, ho.shape)
            ho /= ho.sum(axis=1, keepdims=True)

            test = rng.dirichlet([1] * n_cls, n_test)

            record = ModelRecord(
                registry_id=f"model_{name}",
                algo="xgb", params={}, cv_seed=42, model_seed=11,
                n_folds=5, n_features=10, feature_hash="x",
                fold_scores=[0.7, 0.7, 0.7, 0.7, 0.7],
                oof_score=bias,
                tags=["test"],
            )
            reg.register(record, oof, ho, test)

        yield reg, y_oof, y_ho


def test_pairwise_correlations(registry_with_3_models):
    reg, _, _ = registry_with_3_models
    result = pairwise_correlations(reg, tags=["test"], split="oof_probs")
    assert len(result["ids"]) == 3
    assert result["matrix"].shape == (3, 3)
    # Diagonal is 1
    assert np.allclose(np.diag(result["matrix"]), 1.0)


def test_simple_average(registry_with_3_models):
    reg, _, _ = registry_with_3_models
    blended = simple_average(reg, tags=["test"], split="test_probs")
    assert blended.shape == (100, 3)
    assert np.allclose(blended.sum(axis=1), 1.0, atol=1e-6)


def test_nelder_mead_blend(registry_with_3_models):
    reg, y_oof, y_ho = registry_with_3_models
    result = nelder_mead_blend(
        reg, y_oof, tags=["test"], regularization=0.05,
        split_for_apply="test_probs",
    )
    # Weights sum to ~1
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-5
    # Applied shape
    assert result["applied"].shape == (100, 3)


def test_stacking_meta(registry_with_3_models):
    reg, y_oof, _ = registry_with_3_models
    result = stacking_meta(reg, y_oof, tags=["test"], split_for_apply="test_probs")
    assert result["applied"].shape == (100, 3)
    assert 0.0 <= result["oof_score"] <= 1.0
