"""v32 — LGB + CatBoost on v30's feature set. Blend with v30.

Tests whether algorithm diversity on the enhanced (signal-factory) feature set
produces a real blend gain over v30 alone.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from s6e4_v14_reproduction import (
    DATA_RAW, TARGET, TARGET_MAPPING, build_v14_features, S6E4_ROOT,
)
from s6e4_v30_signal_enhanced import add_signal_factory_features
from s6e4_algo_sweep import LGB_PARAMS, CATBOOST_PARAMS
from kaggle_playground_utils import config
from kaggle_playground_utils.features import get_cat_cols
from kaggle_playground_utils.train import TrainConfig, train_variant
from kaggle_playground_utils.registry import Registry
from kaggle_playground_utils.blend import (
    simple_average, nelder_mead_blend, pairwise_correlations, _load_splits, _normalize,
)


def main():
    print("=" * 60)
    print("v32 — LGB + CatBoost on v30 features + blend")
    print("=" * 60)
    t_total = time.time()

    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    y = train[TARGET].map(TARGET_MAPPING).values
    train_raw = train.drop(columns=["id", TARGET])
    test_raw = test.drop(columns=["id"])

    print("\n[1/5] Building v30 features...")
    train_v30, test_v30 = build_v14_features(train_raw, test_raw)
    train_v30, test_v30 = add_signal_factory_features(train_v30, test_v30)
    cat_cols = get_cat_cols(train_v30)
    for c in cat_cols:
        train_v30[c] = train_v30[c].astype(str).astype("category")
        test_v30[c] = test_v30[c].astype(str).astype("category")
    cat_indices = [train_v30.columns.get_loc(c) for c in cat_cols]
    print(f"  v30 features: {train_v30.shape[1]}")

    tr_idx, ho_idx = train_test_split(
        np.arange(len(y)), test_size=config.HOLDOUT_FRAC,
        stratify=y, random_state=config.HOLDOUT_SEED,
    )
    X_tr = train_v30.iloc[tr_idx].reset_index(drop=True)
    X_ho = train_v30.iloc[ho_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_ho = y[ho_idx]

    reg = Registry(root=S6E4_ROOT / "registry")

    # LGB on v30
    print("\n[2/5] LGB on v30 features...")
    lgb_cfg = TrainConfig(
        algo="lgb", params=LGB_PARAMS, n_classes=3,
        cv_seed=config.CV_SEED, model_seed=config.MODEL_SEED, n_folds=5,
        optuna_trials=200, optuna_n_jobs=4,
        register_as="v32_lgb_on_v30",
        tags=["s6e4", "algo_on_v30", "lgb"],
        notes="LGB yunsuxiaozi-params on v30 enhanced features",
        verbose=True,
    )
    lgb_result = train_variant(lgb_cfg, X_tr, y_tr, X_ho, y_ho, test_v30, registry=reg)

    # CatBoost on v30 (fresh params, needs string cats)
    print("\n[3/5] CatBoost on v30 features...")
    X_tr_cb = X_tr.copy(); X_ho_cb = X_ho.copy(); test_cb = test_v30.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_ho_cb[c] = X_ho_cb[c].astype(str)
        test_cb[c] = test_cb[c].astype(str)
    cb_params = dict(CATBOOST_PARAMS)
    cb_params["cat_features"] = cat_indices
    cb_cfg = TrainConfig(
        algo="catboost", params=cb_params, n_classes=3,
        cv_seed=config.CV_SEED, model_seed=config.MODEL_SEED, n_folds=5,
        optuna_trials=200, optuna_n_jobs=4,
        register_as="v32_catboost_on_v30",
        tags=["s6e4", "algo_on_v30", "catboost"],
        notes="CatBoost GPU on v30 enhanced features",
        verbose=True,
    )
    cb_result = train_variant(cb_cfg, X_tr_cb, y_tr, X_ho_cb, y_ho, test_cb, registry=reg)

    # Correlation analysis (v30 + new LGB + new CatBoost)
    print("\n[4/5] Correlation analysis (v30 + algo_on_v30 models)...")
    blend_ids = ["v30_signal_enhanced", "v32_lgb_on_v30", "v32_catboost_on_v30"]
    # Load probs manually for correlation
    all_oofs = {}
    for mid in blend_ids:
        loaded = reg.load_probs(mid)
        all_oofs[mid] = _normalize(loaded["oof_probs"] * loaded["best_cw"])

    print("  Pairwise ρ (tuned OOF):")
    ids = list(all_oofs.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            corr = np.corrcoef(all_oofs[ids[i]].flatten(), all_oofs[ids[j]].flatten())[0, 1]
            print(f"    {ids[i]:<25} vs {ids[j]:<25} ρ={corr:.5f}")

    # Blend v30 + LGB + CatBoost
    print("\n[5/5] Blend analysis...")

    # Manual blend since blend.py takes tags
    oof_list = [all_oofs[mid] for mid in blend_ids]
    ho_list = []
    test_list = []
    for mid in blend_ids:
        loaded = reg.load_probs(mid)
        ho_list.append(_normalize(loaded["holdout_probs"] * loaded["best_cw"]))
        test_list.append(_normalize(loaded["test_probs"] * loaded["best_cw"]))

    # Simple average
    avg_ho = _normalize(np.mean(ho_list, axis=0))
    avg_test = _normalize(np.mean(test_list, axis=0))
    avg_bacc = balanced_accuracy_score(y_ho, avg_ho.argmax(axis=1))
    print(f"  Simple average (3 models) holdout: {avg_bacc:.5f}")

    # Nelder-Mead
    from scipy.optimize import minimize
    n = len(oof_list)
    w0 = np.ones(n) / n
    def obj(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0: return 0.0
        w = w / w.sum()
        blended = _normalize(sum(w[i] * oof_list[i] for i in range(n)))
        return -balanced_accuracy_score(y_tr, blended.argmax(axis=1)) + 0.001 * np.sum((w - 1/n)**2)
    res = minimize(obj, w0, method="Nelder-Mead", options={"xatol": 1e-4, "maxiter": 1000})
    w_opt = np.clip(res.x, 0, None); w_opt /= w_opt.sum()
    blended_ho = _normalize(sum(w_opt[i] * ho_list[i] for i in range(n)))
    blended_test = _normalize(sum(w_opt[i] * test_list[i] for i in range(n)))
    nm_bacc = balanced_accuracy_score(y_ho, blended_ho.argmax(axis=1))
    print(f"  Nelder-Mead weights: {dict(zip(blend_ids, w_opt))}")
    print(f"  Nelder-Mead holdout:  {nm_bacc:.5f}")

    # Save blend submission
    from datetime import datetime
    inv_map = {0: "Low", 1: "Medium", 2: "High"}
    preds = blended_test.argmax(axis=1)
    labels = [inv_map[p] for p in preds]
    sub = pd.DataFrame({"id": test["id"], "Irrigation_Need": labels})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = S6E4_ROOT / "submissions" / f"submission_{ts}_v32_blend.csv"
    sub.to_csv(sub_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("v32 SUMMARY")
    print("=" * 60)
    print(f"  v30 single holdout:       0.97479 (LB 0.97654)")
    print(f"  LGB on v30 holdout:       {lgb_result['holdout_score_tuned']:.5f}")
    print(f"  CatBoost on v30 holdout:  {cb_result['holdout_score_tuned']:.5f}")
    print(f"  Simple average (3):       {avg_bacc:.5f}")
    print(f"  Nelder-Mead blend:        {nm_bacc:.5f}")
    print(f"  Delta blend vs v30:       {nm_bacc - 0.97479:+.5f}")
    print(f"  Saved: {sub_path}")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
