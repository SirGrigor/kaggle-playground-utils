"""greedy_selection.py — Greedy forward feature selection with mini-tests.

Given candidate features from signal_factory, select the subset that actually
improves a model's holdout score. Avoids the "top-K MI picks redundant features"
trap by testing each candidate's marginal contribution empirically.

Algorithm:
  1. Start from base feature set (already-working features)
  2. Candidate pool: top-N from signal_factory
  3. For each round:
     a. For each UNSELECTED candidate, train (mini or full) with base + selected + candidate
     b. Rank by holdout improvement
     c. Pick the one with biggest lift (if > threshold)
     d. Add to selected set, remove from candidates
  4. Stop when no candidate exceeds lift threshold

Mini-test mode: 10% data, 1 fold per candidate → fast but noisier.
Full-test mode: full 5-fold per candidate → precise but expensive.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from .config import CV_SEED, MODEL_SEED


def _mini_test_with_candidate(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_ho: pd.DataFrame, y_ho: np.ndarray,
    base_cols: list[str], candidate_name: str,
    candidate_tr: np.ndarray, candidate_ho: np.ndarray,
    algo: str, params: dict,
    sample_frac: float = 0.20,
    metric_fn: Callable = balanced_accuracy_score,
) -> float:
    """Train a quick model with (base + candidate) features. Return val score.

    Uses a subsample of training data + 1 fold for speed. 1-2 min per candidate.
    """
    # Subsample training data for speed
    sub_idx, _ = train_test_split(
        np.arange(len(y_tr)), train_size=sample_frac,
        stratify=y_tr, random_state=999,
    )
    X_sub = X_tr.iloc[sub_idx].reset_index(drop=True).copy()
    y_sub = y_tr[sub_idx]
    cand_sub = candidate_tr[sub_idx]

    # Inner train/val split
    inner_tr, inner_val = train_test_split(
        np.arange(len(y_sub)), test_size=0.20, stratify=y_sub, random_state=42,
    )

    # Feature matrix: base_cols + candidate
    X_tr_full = X_sub[base_cols].copy()
    X_tr_full[candidate_name] = cand_sub

    X_train = X_tr_full.iloc[inner_tr]
    X_val = X_tr_full.iloc[inner_val]
    y_train = y_sub[inner_tr]
    y_val = y_sub[inner_val]

    sw = compute_sample_weight("balanced", y_train)

    if algo == "xgb":
        import xgboost as xgb
        p = dict(params)
        early_stop = p.pop("early_stopping_rounds", 200)
        m = xgb.XGBClassifier(**p, early_stopping_rounds=early_stop)
        m.fit(X_train, y_train, sample_weight=sw,
              eval_set=[(X_val, y_val)], verbose=False)
        val_pred = m.predict_proba(X_val)
    elif algo == "lgb":
        import lightgbm as lgb
        p = dict(params)
        early_stop = p.pop("early_stopping_rounds", 100)
        m = lgb.LGBMClassifier(**p)
        m.fit(X_train, y_train, sample_weight=sw,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(early_stop, verbose=False),
                         lgb.log_evaluation(0)])
        val_pred = m.predict_proba(X_val)
    else:
        raise ValueError(f"algo {algo} not supported in greedy selection yet")

    return metric_fn(y_val, val_pred.argmax(axis=1))


def greedy_forward_selection(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_ho: pd.DataFrame, y_ho: np.ndarray,
    base_cols: list[str],
    candidates: dict[str, np.ndarray],  # name -> train-size array
    algo: str = "xgb",
    params: dict = None,
    lift_threshold: float = 0.0005,
    max_rounds: int = 10,
    sample_frac: float = 0.20,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """Greedily select candidates one at a time based on mini-test lift.

    Args:
        X_tr: training data with all base_cols (plus any other columns which
              will NOT be used as features unless in base_cols)
        y_tr: training labels
        base_cols: names of columns to always include as features
        candidates: dict of candidate_name -> array (aligned to X_tr rows)
        algo: "xgb" or "lgb"
        params: algorithm hyperparameters
        lift_threshold: minimum improvement to add a candidate (per mini-test)
        max_rounds: stop after this many rounds even if improvements found
        sample_frac: fraction of training data for each mini-test

    Returns:
        (selected_candidate_names, round_log DataFrame)
    """
    if params is None:
        params = {}

    # Run baseline: no candidates
    baseline_score = _baseline_score(X_tr, y_tr, base_cols, algo, params, sample_frac)
    if verbose:
        print(f"  Baseline (base cols only): {baseline_score:.5f}")

    selected = []
    remaining = dict(candidates)
    rounds_log = []

    for round_num in range(1, max_rounds + 1):
        if not remaining:
            break
        if verbose:
            print(f"\n  === Round {round_num}: testing {len(remaining)} candidates ===")

        # Current base = base_cols + already-selected
        current_base = base_cols + selected
        current_X = X_tr[current_base].copy()
        # Add selected candidates as columns
        for s in selected:
            if s in candidates:
                current_X[s] = candidates[s]
        current_base_cols_in_X = current_X.columns.tolist()

        # Test each remaining candidate
        lifts = {}
        for cand_name, cand_arr in remaining.items():
            t0 = time.time()
            score = _mini_test_with_candidate(
                current_X, y_tr, X_ho, y_ho,
                current_base_cols_in_X, cand_name, cand_arr, None,
                algo, params, sample_frac,
            )
            lift = score - baseline_score
            lifts[cand_name] = lift
            if verbose:
                print(f"    {cand_name:<30} score={score:.5f} lift={lift:+.5f} "
                      f"({time.time()-t0:.0f}s)")

        # Pick best
        best_cand = max(lifts, key=lifts.get)
        best_lift = lifts[best_cand]

        rounds_log.append({
            "round": round_num,
            "best_candidate": best_cand,
            "best_lift": best_lift,
            "all_lifts": lifts,
        })

        if best_lift < lift_threshold:
            if verbose:
                print(f"\n  STOP: best lift {best_lift:.5f} < threshold {lift_threshold}")
            break

        if verbose:
            print(f"\n  ✅ ROUND {round_num}: added '{best_cand}' (lift {best_lift:+.5f})")
        selected.append(best_cand)
        del remaining[best_cand]
        baseline_score = _baseline_score(
            X_tr, y_tr, base_cols + selected,
            algo, params, sample_frac,
            extra_cols_data={s: candidates[s] for s in selected},
        )

    if verbose:
        print(f"\n  FINAL: selected {len(selected)} candidates")
    return selected, pd.DataFrame(rounds_log)


def _baseline_score(X_tr, y_tr, base_cols, algo, params, sample_frac,
                    extra_cols_data=None):
    """Score a baseline model (base_cols only) on subsample."""
    sub_idx, _ = train_test_split(
        np.arange(len(y_tr)), train_size=sample_frac,
        stratify=y_tr, random_state=999,
    )
    X_sub = X_tr.iloc[sub_idx].reset_index(drop=True).copy()
    if extra_cols_data:
        for name, arr in extra_cols_data.items():
            X_sub[name] = arr[sub_idx]

    y_sub = y_tr[sub_idx]
    inner_tr, inner_val = train_test_split(
        np.arange(len(y_sub)), test_size=0.20, stratify=y_sub, random_state=42,
    )
    use_cols = base_cols + (list(extra_cols_data.keys()) if extra_cols_data else [])
    X_train = X_sub[use_cols].iloc[inner_tr]
    X_val = X_sub[use_cols].iloc[inner_val]
    y_train = y_sub[inner_tr]
    y_val = y_sub[inner_val]

    sw = compute_sample_weight("balanced", y_train)
    if algo == "xgb":
        import xgboost as xgb
        p = dict(params)
        early_stop = p.pop("early_stopping_rounds", 200)
        m = xgb.XGBClassifier(**p, early_stopping_rounds=early_stop)
        m.fit(X_train, y_train, sample_weight=sw,
              eval_set=[(X_val, y_val)], verbose=False)
        val_pred = m.predict_proba(X_val)
    else:
        raise NotImplementedError(f"algo {algo}")
    return balanced_accuracy_score(y_val, val_pred.argmax(axis=1))
