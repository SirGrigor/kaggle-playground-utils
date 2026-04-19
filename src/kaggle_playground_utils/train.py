"""train.py — unified `train_variant()` for XGB/LGB/CatBoost.

MVP implementation: XGB first. LGB/CatBoost/MLP follow same pattern.
"""
from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from .config import (
    CV_SEED, HOLDOUT_SEED, MODEL_SEED,
    OPTUNA_CLASS_WEIGHT_LOW, OPTUNA_CLASS_WEIGHT_HIGH,
)
from .registry import ModelRecord, Registry, make_registry_id


@dataclass
class TrainConfig:
    algo: str                               # "xgb", "lgb", "catboost"
    params: dict[str, Any]                  # algo-specific hyperparameters
    n_classes: int = 3
    metric_fn: Callable = balanced_accuracy_score
    cv_seed: int = CV_SEED
    model_seed: int = MODEL_SEED
    n_folds: int = 5
    mini_test_threshold: float = 0.0        # 0 disables
    mini_test_frac: float = 0.10
    optuna_trials: int = 200
    optuna_n_jobs: int = 4
    register_as: str | None = None          # overrides auto-generated registry_id
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    verbose: bool = True


def _hash_features(X: pd.DataFrame) -> str:
    """Deterministic hash of feature-matrix shape + columns + dtypes."""
    key = f"{X.shape}|{list(X.columns)}|{list(X.dtypes.astype(str))}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _normalize_probs(p: np.ndarray) -> np.ndarray:
    return p / p.sum(axis=1, keepdims=True)


def _train_xgb_fold(X_tr: pd.DataFrame, y_tr: np.ndarray,
                    X_val: pd.DataFrame, y_val: np.ndarray,
                    params: dict) -> tuple[np.ndarray, int]:
    """Train one XGB fold; return val probs + best iteration."""
    import xgboost as xgb

    sw = compute_sample_weight("balanced", y_tr)
    early_stop = params.pop("early_stopping_rounds", 500)
    m = xgb.XGBClassifier(**params, early_stopping_rounds=early_stop)
    m.fit(X_tr, y_tr, sample_weight=sw,
          eval_set=[(X_val, y_val)], verbose=False)
    return m, m.predict_proba(X_val), m.best_iteration


def _train_lgb_fold(X_tr, y_tr, X_val, y_val, params):
    """Train one LGB fold."""
    import lightgbm as lgb
    sw = compute_sample_weight("balanced", y_tr)
    # LGB-specific callback pattern
    early_stop = params.pop("early_stopping_rounds", 250)
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, sample_weight=sw,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(early_stop, verbose=False),
                     lgb.log_evaluation(0)])
    return m, m.predict_proba(X_val), m.best_iteration_


def _train_catboost_fold(X_tr, y_tr, X_val, y_val, params):
    """Train one CatBoost fold."""
    from catboost import CatBoostClassifier
    early_stop = params.pop("early_stopping_rounds", 300)
    auto_class_weights = params.pop("auto_class_weights", "Balanced")
    m = CatBoostClassifier(
        **params, auto_class_weights=auto_class_weights, verbose=False,
        early_stopping_rounds=early_stop,
    )
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    try:
        best_iter = m.get_best_iteration()
    except Exception:
        best_iter = 0
    return m, m.predict_proba(X_val), best_iter


_FOLD_TRAINERS = {
    "xgb": _train_xgb_fold,
    "lgb": _train_lgb_fold,
    "catboost": _train_catboost_fold,
}


def _predict(model, X, algo: str) -> np.ndarray:
    """Unified predict_proba across algorithms."""
    return model.predict_proba(X)


def _mini_test(config: TrainConfig, X: pd.DataFrame, y: np.ndarray) -> bool:
    """Quick sanity: train on 10% of data, 1 fold. Abort if below threshold."""
    if config.mini_test_threshold <= 0:
        return True
    sub_idx, _ = train_test_split(
        np.arange(len(y)), train_size=config.mini_test_frac,
        stratify=y, random_state=999,
    )
    X_sub, y_sub = X.iloc[sub_idx].reset_index(drop=True), y[sub_idx]
    tr_idx, val_idx = train_test_split(
        np.arange(len(y_sub)), test_size=0.20,
        stratify=y_sub, random_state=42,
    )
    trainer = _FOLD_TRAINERS[config.algo]
    params = dict(config.params)  # copy since trainer may pop keys
    t0 = time.time()
    _, val_pred, best_iter = trainer(
        X_sub.iloc[tr_idx], y_sub[tr_idx],
        X_sub.iloc[val_idx], y_sub[val_idx],
        params,
    )
    bacc = config.metric_fn(y_sub[val_idx], val_pred.argmax(axis=1))
    if config.verbose:
        print(f"  [mini-test] bacc={bacc:.5f} (threshold {config.mini_test_threshold:.2f}, "
              f"best_iter={best_iter}, {time.time()-t0:.0f}s)")
    return bacc >= config.mini_test_threshold


def _tune_class_weights(oof: np.ndarray, y: np.ndarray,
                         n_trials: int, n_jobs: int,
                         metric_fn: Callable, n_classes: int) -> tuple[np.ndarray, float]:
    """Optuna search over per-class multiplicative weights."""
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        w = np.array([
            trial.suggest_float(f"cw{i}", OPTUNA_CLASS_WEIGHT_LOW, OPTUNA_CLASS_WEIGHT_HIGH)
            for i in range(n_classes)
        ])
        adj = _normalize_probs(oof * w)
        return metric_fn(y, adj.argmax(axis=1))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
    best_cw = np.array([study.best_params[f"cw{i}"] for i in range(n_classes)])
    return best_cw, study.best_value


def train_variant(
    config: TrainConfig,
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_ho: pd.DataFrame | None = None, y_ho: np.ndarray | None = None,
    X_test: pd.DataFrame | None = None,
    registry: Registry | None = None,
) -> dict | None:
    """Train a model variant with full pipeline.

    Returns dict with:
      oof_probs, oof_probs_tuned, holdout_probs, holdout_probs_tuned,
      test_probs, test_probs_tuned, best_cw, fold_scores,
      oof_score_raw, oof_score_tuned, holdout_score_tuned, registry_id

    Or None if mini-test aborted.
    """
    t_total = time.time()

    if config.algo not in _FOLD_TRAINERS:
        raise ValueError(f"Unknown algo: {config.algo}. Supported: {list(_FOLD_TRAINERS)}")

    if config.verbose:
        print(f"\n[train_variant] algo={config.algo}, n_folds={config.n_folds}, "
              f"n_features={X_tr.shape[1]}, n_rows={len(y_tr)}")

    # Mini-test gate
    if not _mini_test(config, X_tr, y_tr):
        if config.verbose:
            print("  [mini-test] ABORT")
        return None

    # Feature hash (for registry)
    feature_hash = _hash_features(X_tr)

    # 5-fold CV
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.cv_seed)
    oof = np.zeros((len(X_tr), config.n_classes), dtype=np.float32)
    ho_sum = (np.zeros((len(X_ho), config.n_classes), dtype=np.float32)
              if X_ho is not None else None)
    test_sum = (np.zeros((len(X_test), config.n_classes), dtype=np.float32)
                if X_test is not None else None)
    fold_scores = []
    best_iters = []

    trainer = _FOLD_TRAINERS[config.algo]

    for fold, (t_idx, v_idx) in enumerate(kf.split(X_tr), start=1):
        t_fold = time.time()
        params_copy = dict(config.params)  # trainer may pop keys
        m, val_pred, best_iter = trainer(
            X_tr.iloc[t_idx], y_tr[t_idx],
            X_tr.iloc[v_idx], y_tr[v_idx],
            params_copy,
        )
        oof[v_idx] = val_pred
        if X_ho is not None:
            ho_sum += _predict(m, X_ho, config.algo) / config.n_folds
        if X_test is not None:
            test_sum += _predict(m, X_test, config.algo) / config.n_folds

        bacc = config.metric_fn(y_tr[v_idx], val_pred.argmax(axis=1))
        fold_scores.append(bacc)
        best_iters.append(best_iter)
        if config.verbose:
            print(f"  Fold {fold}: score={bacc:.5f}, best_iter={best_iter}, "
                  f"time={time.time()-t_fold:.0f}s")
        del m
        gc.collect()

    oof_score_raw = config.metric_fn(y_tr, oof.argmax(axis=1))
    if config.verbose:
        print(f"  OOF raw score: {oof_score_raw:.5f}")

    # Optuna class-weight tuning
    best_cw, oof_score_tuned = _tune_class_weights(
        oof, y_tr, config.optuna_trials, config.optuna_n_jobs,
        config.metric_fn, config.n_classes,
    )
    if config.verbose:
        print(f"  Optuna best: {oof_score_tuned:.5f}, weights={best_cw}")

    # Apply tuning
    oof_tuned = _normalize_probs(oof * best_cw)
    ho_tuned = _normalize_probs(ho_sum * best_cw) if ho_sum is not None else None
    test_tuned = _normalize_probs(test_sum * best_cw) if test_sum is not None else None

    holdout_score_tuned = None
    if ho_tuned is not None and y_ho is not None:
        holdout_score_tuned = config.metric_fn(y_ho, ho_tuned.argmax(axis=1))
        if config.verbose:
            print(f"  Holdout tuned: {holdout_score_tuned:.5f}")

    # Register
    registry_id = config.register_as or make_registry_id(
        config.algo, config.params, feature_hash, config.cv_seed, config.model_seed,
    )
    if registry is not None:
        record = ModelRecord(
            registry_id=registry_id,
            algo=config.algo,
            params=config.params,
            cv_seed=config.cv_seed,
            model_seed=config.model_seed,
            n_folds=config.n_folds,
            n_features=X_tr.shape[1],
            feature_hash=feature_hash,
            fold_scores=fold_scores,
            oof_score=float(oof_score_tuned),
            holdout_score=float(holdout_score_tuned) if holdout_score_tuned else None,
            notes=config.notes,
            tags=config.tags,
        )
        registry.register(record, oof, ho_sum, test_sum, best_cw)
        if config.verbose:
            print(f"  Registered as: {registry_id}")

    if config.verbose:
        print(f"  Total time: {(time.time()-t_total)/60:.1f} min")

    return {
        "oof_probs": oof,
        "oof_probs_tuned": oof_tuned,
        "holdout_probs": ho_sum,
        "holdout_probs_tuned": ho_tuned,
        "test_probs": test_sum,
        "test_probs_tuned": test_tuned,
        "best_cw": best_cw,
        "fold_scores": fold_scores,
        "best_iters": best_iters,
        "oof_score_raw": float(oof_score_raw),
        "oof_score_tuned": float(oof_score_tuned),
        "holdout_score_tuned": float(holdout_score_tuned) if holdout_score_tuned else None,
        "registry_id": registry_id,
        "feature_hash": feature_hash,
    }
