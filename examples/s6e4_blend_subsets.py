"""Test different blend subsets to isolate which models hurt vs help.

Path B: Drop weak models from registry and re-blend.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from s6e4_v14_reproduction import S6E4_ROOT
from kaggle_playground_utils.registry import Registry
from kaggle_playground_utils.blend import _normalize, _load_splits


def blend_subset(registry, y_oof, y_ho, model_ids: list[str], label: str):
    """Blend a specific subset of registry entries."""
    # Manually filter to given ids
    oof_list, ho_list = [], []
    for mid in model_ids:
        loaded = registry.load_probs(mid)
        oof = _normalize(loaded["oof_probs"] * loaded["best_cw"])  # apply tuning
        ho = _normalize(loaded["holdout_probs"] * loaded["best_cw"])
        oof_list.append(oof)
        ho_list.append(ho)

    # Simple average
    avg_ho = _normalize(np.mean(ho_list, axis=0))
    avg_bacc = balanced_accuracy_score(y_ho, avg_ho.argmax(axis=1))

    # Nelder-Mead on OOF
    n = len(oof_list)
    w0 = np.ones(n) / n
    def obj(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0: return 0.0
        w = w / w.sum()
        blended = _normalize(sum(w[i] * oof_list[i] for i in range(n)))
        return -balanced_accuracy_score(y_oof, blended.argmax(axis=1)) + 0.001 * np.sum((w - 1/n)**2)
    res = minimize(obj, w0, method="Nelder-Mead", options={"xatol": 1e-4, "maxiter": 1000})
    w_opt = np.clip(res.x, 0, None); w_opt /= w_opt.sum()
    blended_ho = _normalize(sum(w_opt[i] * ho_list[i] for i in range(n)))
    nm_bacc = balanced_accuracy_score(y_ho, blended_ho.argmax(axis=1))

    print(f"\n  {label}: {n} models")
    for mid, w in zip(model_ids, w_opt):
        print(f"    {mid:<20} weight={w:.4f}")
    print(f"  Simple average:  {avg_bacc:.5f}")
    print(f"  Nelder-Mead:     {nm_bacc:.5f}")
    return avg_bacc, nm_bacc


def main():
    reg = Registry(root=S6E4_ROOT / "registry")
    y_tr = np.load(S6E4_ROOT / "v14" / "probs" / "y_tr.npy")
    y_ho = np.load(S6E4_ROOT / "v14" / "probs" / "y_ho.npy")

    print("=" * 60)
    print("Blend subset testing — which models help vs hurt?")
    print("=" * 60)

    # Available models
    all_xgb_seeds = ["v14_seed_11", "v14_seed_42", "v14_seed_123", "v14_seed_2026", "v14_seed_7"]

    print("\nBaseline: v14 single = 0.97423")

    # 1. All 5 XGB seeds only
    blend_subset(reg, y_tr, y_ho, all_xgb_seeds, "5 XGB seeds only")

    # 2. 5 XGB + LGB
    blend_subset(reg, y_tr, y_ho, all_xgb_seeds + ["v14_lgb"], "5 XGB + LGB")

    # 3. 5 XGB + CatBoost
    blend_subset(reg, y_tr, y_ho, all_xgb_seeds + ["v14_catboost"], "5 XGB + CatBoost")

    # 4. 5 XGB + LGB + CatBoost (full)
    blend_subset(reg, y_tr, y_ho, all_xgb_seeds + ["v14_lgb", "v14_catboost"], "5 XGB + LGB + CatBoost (all)")

    # 5. Just XGB + LGB (no seeds, no catboost)
    blend_subset(reg, y_tr, y_ho, ["v14_seed_123", "v14_lgb"], "Best XGB + LGB (2 models)")

    # 6. Best XGB + LGB + CatBoost
    blend_subset(reg, y_tr, y_ho, ["v14_seed_123", "v14_lgb", "v14_catboost"], "Best XGB + LGB + CatBoost (3 models)")


if __name__ == "__main__":
    main()
