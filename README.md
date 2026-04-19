# kaggle-playground-utils

Reusable toolkit for Kaggle Playground Series competitions.

**Status**: scaffolding phase (v0.1.0-dev). Target public release: 2026-04-30.

## Motivation

After 30+ experiments on Playground Series S6E4, I found that:
- 80% of the work is repeatable across competitions
- The gap to the top 0.98+ cluster is **infrastructure**, not insight
- Each new model variant should take 5 minutes, not 30

This toolkit standardizes the repeatable parts so every new competition starts from a known infrastructure baseline, not from scratch.

## Design principles

1. **Reproducibility**: fixed seeds via `config.py` shared across modules
2. **Registry-first**: every model auto-saves OOF/holdout/test probabilities
3. **Fail fast**: feature count assertions, mini-test gates, CUDA checks
4. **Cache expensive work**: FE hashed to disk, recomputed only when inputs change
5. **Precision-safe math**: integer-based digit extraction (IEEE 754 safe)
6. **One-variable discipline**: pre-experiment JSON template enforced

## Modules (target structure)

| Module | Purpose |
|---|---|
| `config.py` | Canonical seeds, target mappings, thresholds |
| `features.py` | Precision-safe digit extraction, formula logits, threshold booleans, one-hot helpers |
| `cache.py` | Disk cache for expensive feature engineering (hash-based invalidation) |
| `registry.py` | Model registration, OOF/holdout/test probs persistence |
| `train.py` | Unified `train_variant()` for XGB/LGB/CatBoost/MLP |
| `blend.py` | Auto-blend: simple average, Nelder-Mead, pairwise correlation, stacking |
| `probes.py` | Signal discovery: MI, clusters, residual signature, adversarial CV |
| `diagnose.py` | SHAP, learning curves, calibration plots |

## Quick start (target UX)

```python
from kaggle_playground_utils import train, blend, features

# Feature engineering (cached)
X_train_fe, X_test_fe = features.build_playground_tabular(X_train, X_test, config=...)

# Train 3 diverse models
for algo in ["xgb", "lgb", "catboost"]:
    train.train_variant(
        algo=algo,
        X_tr=X_train_fe, y_tr=y,
        X_ho=X_holdout_fe, y_ho=y_ho,
        X_test=X_test_fe,
        register_as=f"baseline_{algo}",
    )

# Auto-blend
ensemble = blend.auto_blend(registry_dir="registry/")
ensemble.submit("submission.csv")
```

## Build schedule

See [build plan](https://github.com/ilgrig/kaggle-playground-utils/blob/master/BUILD_PLAN.md) for day-by-day tracking.

## License

MIT
