# kaggle-playground-utils

Reusable toolkit for Kaggle Playground Series competitions. Built during Playground S6E4 (April 2026) where it produced a solo LB **0.97654** via an automated signal-discovery pipeline.

**Status**: working on XGBoost / LightGBM / CatBoost pipelines. Feature discovery (signal factory + greedy selection) and model registry are production-ready; `probes.py` and `diagnose.py` are still stubs.

## What's in it

| Module | Status | Purpose |
|---|---|---|
| [`config.py`](src/kaggle_playground_utils/config.py) | ✓ | Canonical CV / holdout / model seeds |
| [`features.py`](src/kaggle_playground_utils/features.py) | ✓ | Precision-safe digit extraction (IEEE 754 hardened), threshold booleans, formula logits, categorical one-hots |
| [`cache.py`](src/kaggle_playground_utils/cache.py) | ✓ | Hash-based FE cache (DataFrame-aware, function-source versioned) |
| [`registry.py`](src/kaggle_playground_utils/registry.py) | ✓ | Auto-save OOF / holdout / test probabilities, `best_cw`, metadata per model |
| [`train.py`](src/kaggle_playground_utils/train.py) | ✓ | `train_variant()` for XGB / LGB / CatBoost: 5-fold CV, Optuna class-weight tuning, mini-test gate |
| [`blend.py`](src/kaggle_playground_utils/blend.py) | ✓ | Simple-average, Nelder-Mead, pairwise-correlation blending (applies per-model `best_cw` by default) |
| [`signal_factory.py`](src/kaggle_playground_utils/signal_factory.py) | ✓ | 9 transformation families, MI ranking, orthogonality dedup — see the [tutorial notebook](notebooks/s6e4_signal_factory_tutorial.ipynb) |
| [`greedy_selection.py`](src/kaggle_playground_utils/greedy_selection.py) | beta | Forward feature selection with mini-test lift gate (the "top-K by MI is wrong" fix) |
| `probes.py` | stub | Planned: fast data probes (target-leak detection, class-conditional histograms) |
| `diagnose.py` | stub | Planned: SHAP, learning curves, calibration plots |

## Design principles

1. **Reproducibility** — fixed seeds via `config.py` shared across modules
2. **Registry-first** — every model auto-saves OOF / holdout / test probabilities
3. **Fail fast** — feature count assertions, mini-test gates, CUDA checks
4. **Cache expensive work** — FE hashed to disk, recomputed only when inputs change
5. **Precision-safe math** — integer-based digit extraction (IEEE 754 safe)

## Quick start

```bash
uv pip install -e .
# or
pip install -e .
```

```python
from kaggle_playground_utils.signal_factory import discover_signals
from kaggle_playground_utils.train import TrainConfig, train_variant
from kaggle_playground_utils.registry import Registry

# Discover candidate features
result = discover_signals(
    train_df=train_raw, y=y,
    numeric_cols=NUMERIC, sample_size=50000,
    top_n=20, max_corr_with_existing=0.80,
    existing_features=existing,  # dict of name -> array, for dedup
)

# Train + auto-register
cfg = TrainConfig(
    algo="xgb", params=XGB_PARAMS, n_classes=3,
    optuna_trials=200, register_as="my_model",
    tags=["s6e4", "xgb"],
)
reg = Registry(root=Path("registry"))
res = train_variant(cfg, X_tr, y_tr, X_ho, y_ho, X_test, registry=reg)
```

## Examples

All [`examples/*.py`](examples/) are S6E4-era reference scripts:
- `s6e4_v14_reproduction.py` — reproduces a formula-logit LR baseline (LB 0.97542)
- `s6e4_v30_signal_enhanced.py` — v14 + 5 factory-discovered features → **solo LB 0.97654**
- `s6e4_v31_factory_v2.py` through `s6e4_v34_greedy_round1_diagnostic.py` — iterations documenting factory saturation + why greedy selection corrects the top-K-by-MI failure mode

## Tutorial notebook

[`notebooks/s6e4_signal_factory_tutorial.ipynb`](notebooks/s6e4_signal_factory_tutorial.ipynb) — self-contained walkthrough of the signal factory methodology. Designed for the Kaggle Code tab; also renders directly on GitHub.

## License

MIT — see [LICENSE](LICENSE).
