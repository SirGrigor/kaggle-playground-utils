"""blend.py — auto-blend across registered models.

Given a Registry, produce ensemble predictions via:
  - simple_average()          — equal weights
  - nelder_mead_blend()       — OOF-optimized with regularization toward uniform
  - pairwise_correlations()   — diagnostic: identify duplicate-signal models
  - stacking_meta()           — LR meta-learner on OOF probs

All functions return a normalized probability matrix.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from .registry import Registry


def _normalize(p: np.ndarray) -> np.ndarray:
    return p / p.sum(axis=1, keepdims=True)


def _load_splits(registry: Registry, tags: list[str] | None = None,
                 split: str = "holdout") -> tuple[list[str], list[np.ndarray]]:
    """Load one split (oof/holdout/test) from all matching registered models."""
    if split not in {"oof_probs", "holdout_probs", "test_probs"}:
        raise ValueError(f"split must be one of oof_probs/holdout_probs/test_probs, got {split}")
    models = registry.list_models(tags=tags)
    ids, probs = [], []
    for m in models:
        loaded = registry.load_probs(m["registry_id"])
        if split in loaded:
            ids.append(m["registry_id"])
            probs.append(loaded[split])
    return ids, probs


def pairwise_correlations(registry: Registry, tags: list[str] | None = None,
                          split: str = "oof_probs") -> dict:
    """Return pairwise correlation matrix across all registered models' probs."""
    ids, probs = _load_splits(registry, tags=tags, split=split)
    if len(probs) < 2:
        return {"ids": ids, "matrix": None}
    # Flatten each model's probs for Pearson correlation
    flat = [p.flatten() for p in probs]
    n = len(flat)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            elif j > i:
                mat[i, j] = float(np.corrcoef(flat[i], flat[j])[0, 1])
                mat[j, i] = mat[i, j]
    return {"ids": ids, "matrix": mat}


def simple_average(registry: Registry, tags: list[str] | None = None,
                   split: str = "test_probs") -> np.ndarray:
    """Equal-weight mean of registered models' probs for the given split."""
    ids, probs = _load_splits(registry, tags=tags, split=split)
    if not probs:
        raise ValueError(f"No {split} found in registry for tags={tags}")
    return _normalize(np.mean(probs, axis=0))


def nelder_mead_blend(registry: Registry,
                      y_true_oof: np.ndarray,
                      metric_fn: Callable = balanced_accuracy_score,
                      tags: list[str] | None = None,
                      regularization: float = 0.05,
                      split_for_apply: str = "test_probs",
                      max_iter: int = 500) -> dict:
    """OOF-optimized blend weights (regularized toward uniform).

    Uses OOF probs for optimization, then applies weights to the specified
    split (default test) for the returned predictions.

    Args:
        registry: where to find models
        y_true_oof: true labels aligned to OOF rows (same across all models)
        metric_fn: maximized; default balanced_accuracy_score
        regularization: penalty coefficient on deviation from uniform weights
        split_for_apply: "test_probs" or "holdout_probs"

    Returns:
        {
          "weights": dict of registry_id → weight,
          "oof_score": best OOF score,
          "applied": blended probs for split_for_apply,
        }
    """
    ids_oof, oof_list = _load_splits(registry, tags=tags, split="oof_probs")
    if len(oof_list) < 2:
        raise ValueError(f"Need ≥2 registered models' OOF probs, got {len(oof_list)}")

    n = len(oof_list)
    w0 = np.ones(n) / n  # uniform start

    def objective(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return 0.0
        w = w / w.sum()
        blended = _normalize(sum(w[i] * oof_list[i] for i in range(n)))
        score = metric_fn(y_true_oof, blended.argmax(axis=1))
        # Regularize toward uniform
        reg_penalty = regularization * np.sum((w - 1 / n) ** 2)
        return -score + reg_penalty

    res = minimize(objective, w0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "maxiter": max_iter})
    w_opt = np.clip(res.x, 0, None)
    w_opt = w_opt / w_opt.sum()

    # OOF score with the optimized weights
    blended_oof = _normalize(sum(w_opt[i] * oof_list[i] for i in range(n)))
    oof_score = metric_fn(y_true_oof, blended_oof.argmax(axis=1))

    # Apply to target split
    ids_apply, apply_list = _load_splits(registry, tags=tags, split=split_for_apply)
    if ids_apply != ids_oof:
        raise RuntimeError(f"Model set differs between OOF and {split_for_apply}")
    applied = _normalize(sum(w_opt[i] * apply_list[i] for i in range(n)))

    return {
        "weights": dict(zip(ids_oof, w_opt.tolist())),
        "oof_score": float(oof_score),
        "applied": applied,
    }


def stacking_meta(registry: Registry,
                   y_true_oof: np.ndarray,
                   tags: list[str] | None = None,
                   meta_C: float = 1.0,
                   class_weight: str = "balanced",
                   split_for_apply: str = "test_probs") -> dict:
    """LR meta-learner on concatenated OOF probs.

    Each model contributes n_classes features. Meta-learner trains on OOF,
    applies to specified split.

    Note: stacking with highly-correlated base models (ρ>0.97) often hurts
    (see S6E4 v20 result). Check pairwise_correlations() first.
    """
    ids_oof, oof_list = _load_splits(registry, tags=tags, split="oof_probs")
    if not oof_list:
        raise ValueError("No models found")

    meta_X_oof = np.hstack(oof_list)
    meta = LogisticRegression(
        solver="lbfgs", max_iter=2000, C=meta_C,
        class_weight=class_weight, random_state=42,
    )
    meta.fit(meta_X_oof, y_true_oof)
    oof_score = balanced_accuracy_score(y_true_oof, meta.predict(meta_X_oof))

    ids_apply, apply_list = _load_splits(registry, tags=tags, split=split_for_apply)
    if ids_apply != ids_oof:
        raise RuntimeError(f"Model set differs between OOF and {split_for_apply}")
    meta_X_apply = np.hstack(apply_list)
    applied = meta.predict_proba(meta_X_apply)

    return {
        "meta_model": meta,
        "ids": ids_oof,
        "oof_score": float(oof_score),
        "applied": applied,
    }
