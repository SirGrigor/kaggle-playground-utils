"""train.py — unified `train_variant()` for XGB/LGB/CatBoost/MLP.

Target (Day 4-5): single entry point that handles:
  - 5-fold (or configurable) CV
  - Class-balanced sample weights
  - Optuna class-weight tuning on OOF
  - Registry auto-save
  - Mini-test gate (10% data, 1 fold) with abort threshold
  - GPU/CPU detection

Stub for Day 1. Full implementation follows.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrainConfig:
    algo: str                               # "xgb", "lgb", "catboost", "mlp"
    params: dict[str, Any]
    cv_seed: int = 42
    model_seed: int = 11
    n_folds: int = 5
    mini_test_threshold: float = 0.0        # 0 disables
    optuna_trials: int = 200
    optuna_n_jobs: int = 4
    register_as: str | None = None
    tags: list[str] = field(default_factory=list)


def train_variant(config: TrainConfig,
                  X_tr: pd.DataFrame, y_tr: np.ndarray,
                  X_ho: pd.DataFrame | None = None, y_ho: np.ndarray | None = None,
                  X_test: pd.DataFrame | None = None,
                  registry_dir: str = "registry") -> dict:
    """Train a model variant with full pipeline.

    STUB — to be implemented Day 4-5. Raises NotImplementedError for now.

    Returns dict with:
      oof_probs, holdout_probs, test_probs, best_cw,
      fold_scores, oof_score, holdout_score, registry_id
    """
    raise NotImplementedError(
        "train_variant is scheduled for Day 4-5 of build plan. "
        "See knowledge-graph/kaggle/2026-16_toolkit-build-plan.md"
    )
