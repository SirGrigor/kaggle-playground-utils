# kaggle-playground-utils

Reusable toolkit for Kaggle Playground Series competitions. Built incrementally:

- **v0.1** (S6E4, April 2026): signal factory + registry + training pipeline. Solo LB **0.97654** on Playground S6E4.
- **v0.2** (S6E5, May 2026): public-notebook harvesting + curated public-mix blending. Public LB **0.95423**, rank **~28 / 1488** on Playground S6E5.

The two generations are complementary — v0.1 focuses on building a strong solo model, v0.2 on aggregating the public ecosystem's predictions into a winning blend.

## What's in it

### v0.2 modules (S6E5 additions)

| Module | Purpose |
|---|---|
| [`harvesting.py`](src/kaggle_playground_utils/harvesting.py) | Bulk-scan top-N public Kaggle notebooks via CLI, validate outputs, produce manifest with claimed_lb / claimed_oof_auc / verdict per kernel |
| [`curated_blend.py`](src/kaggle_playground_utils/curated_blend.py) | **L19 winning lever**: top-K LB-curated public mix beats all-N average. Includes rank-space blending, ratio sweep with predicted_lb scoring |
| [`drive.py`](src/kaggle_playground_utils/drive.py) | Colab Drive ↔ /content sync with merge semantics (handles the partial-restore case that bit us on S6E5) |
| [`environment.py`](src/kaggle_playground_utils/environment.py) | Colab vs local detection, Kaggle CLI auth setup from Colab Secrets / env vars / kaggle.json (with auto-JSON-unwrap) |

### v0.1 modules (S6E4 toolkit)

| Module | Status | Purpose |
|---|---|---|
| [`config.py`](src/kaggle_playground_utils/config.py) | ✓ | Canonical CV / holdout / model seeds |
| [`features.py`](src/kaggle_playground_utils/features.py) | ✓ | Precision-safe digit extraction (IEEE 754 hardened), threshold booleans, formula logits, categorical one-hots |
| [`cache.py`](src/kaggle_playground_utils/cache.py) | ✓ | Hash-based FE cache (DataFrame-aware, function-source versioned) |
| [`registry.py`](src/kaggle_playground_utils/registry.py) | ✓ | Auto-save OOF / holdout / test probabilities, `best_cw`, metadata per model |
| [`train.py`](src/kaggle_playground_utils/train.py) | ✓ | `train_variant()` for XGB / LGB / CatBoost: 5-fold CV, Optuna class-weight tuning, mini-test gate |
| [`blend.py`](src/kaggle_playground_utils/blend.py) | ✓ | Simple-average, Nelder-Mead, pairwise-correlation blending |
| [`signal_factory.py`](src/kaggle_playground_utils/signal_factory.py) | ✓ | 9 transformation families, MI ranking, orthogonality dedup |
| [`greedy_selection.py`](src/kaggle_playground_utils/greedy_selection.py) | beta | Forward feature selection with mini-test lift gate |
| `probes.py` | stub | Planned: fast data probes |
| `diagnose.py` | stub | Planned: SHAP, learning curves, calibration plots |

## Install

```bash
# From GitHub (recommended — pins version)
uv pip install "git+https://github.com/SirGrigor/kaggle-playground-utils@v0.2.0"

# Local dev
git clone https://github.com/SirGrigor/kaggle-playground-utils
cd kaggle-playground-utils
uv pip install -e ".[dev]"
```

## Quick start — v0.2 curated public-mix flow

The S6E5 winning recipe:

```python
from pathlib import Path
import numpy as np

from kaggle_playground_utils import (
    setup_kaggle_auth, detect_environment,
    bulk_harvest, load_manifest,
    load_public_subset, ratio_sweep,
)

# 1. Auth Kaggle CLI (works on local or Colab Secrets)
setup_kaggle_auth()

# 2. Harvest top 50 public notebooks for your competition
test_ids = np.array([...])  # your test row ids
harvest_dir = Path("harvest/v18")
entries = bulk_harvest(
    comp="playground-series-s6e5",
    top_n=50,
    out_dir=harvest_dir,
    test_ids=test_ids,
)

# 3. Filter to top-LB curated publics (the L19 lever)
manifest = load_manifest(harvest_dir / "manifest.json")
publics = load_public_subset(
    manifest,
    claimed_lb_threshold=0.954,   # only keep claimed_lb >= 0.954
)
print(f"Curated pool: {len(publics)} submissions")

# 4. Sweep K × ratio against your baseline
baseline_test_predictions = np.load("probs/v17/test.npy")
results = ratio_sweep(
    baseline_test=baseline_test_predictions,
    public_subs=publics,
    test_ids=test_ids,
    baseline_lb_estimate=0.9538,  # your baseline's known LB
    K_values=[3, 4, 5, 6],
    ratios=[0.30, 0.50, 0.70, 0.90, 1.00],
)

# Sorted by predicted LB — submit top results to Kaggle
for r in results[:5]:
    print(f"{r.name}: predicted LB {r.predicted_lb:.5f}, ratio {r.ratio}, K {r.K}")
```

## Quick start — v0.1 signal factory flow

For solo-model building (the S6E4 path):

```python
from kaggle_playground_utils.signal_factory import discover_signals
from kaggle_playground_utils.train import TrainConfig, train_variant
from kaggle_playground_utils.registry import Registry

result = discover_signals(
    train_df=train_raw, y=y,
    numeric_cols=NUMERIC, sample_size=50000,
    top_n=20, max_corr_with_existing=0.80,
    existing_features=existing,
)

cfg = TrainConfig(algo="xgb", params=XGB_PARAMS, n_classes=3, ...)
reg = Registry(root=Path("registry"))
res = train_variant(cfg, X_tr, y_tr, X_ho, y_ho, X_test, registry=reg)
```

## Design principles

1. **Reproducibility** — fixed seeds via `config.py`, manifest-driven harvest registry
2. **Registry-first** — every model auto-saves OOF / holdout / test probabilities (v0.1) or per-version probs/ folders (v0.2)
3. **Fail fast** — feature count assertions, mini-test gates, CUDA checks, **pre-flight validation** for the v0.2 pipeline
4. **Cache expensive work** — FE hashed to disk; harvest results idempotent (skip-existing flag)
5. **Precision-safe math** — integer-based digit extraction (IEEE 754 safe)
6. **Cross-environment** — same scripts work on local + Colab; Drive sync helpers handle persistence

## Environment compatibility

The toolkit is designed to work identically on local machines and on Colab. The `environment.py` module detects which environment you're in; `drive.py` handles Colab Drive ↔ /content sync transparently.

**Local**: probs/harvest/submissions live in your working dir.

**Colab**: Same paths in `/content`, with `drive.restore_from_drive()` called at script start to populate from Drive (persistent backup), and `drive.sync_to_drive()` called at end to push fresh artifacts back.

Both paths are tested. The drive module has 8 unit tests; environment has 9.

## Examples

All [`examples/*.py`](examples/) are S6E4-era reference scripts using v0.1 modules:
- `s6e4_v14_reproduction.py` — formula-logit LR baseline (LB 0.97542)
- `s6e4_v30_signal_enhanced.py` — v14 + 5 factory-discovered features → **solo LB 0.97654**
- `s6e4_v31_factory_v2.py` through `s6e4_v34_greedy_round1_diagnostic.py` — iterations

For v0.2 examples, see the [playground-s6e5 project](https://github.com/SirGrigor/playground-s6e5) — particularly `notebooks/25_pipeline.py` (end-to-end pipeline) and `notebooks/26_curated_explorer.py` (standalone curated-mix tool).

## Tests

```bash
pytest tests/                       # all tests
pytest tests/test_curated_blend.py  # v0.2 curated math (24 tests)
pytest tests/test_harvesting.py     # v0.2 parsers + validation (24 tests)
pytest tests/test_environment.py    # v0.2 cross-env helpers (9 tests)
pytest tests/test_drive.py          # v0.2 Drive sync (8 tests)
```

v0.2 modules have 65 tests, all passing.

## Lessons captured

The S6E5 post-mortem at `~/knowledge-graph/kaggle/2026-19_s6e5-postmortem.md` documents the L19 lesson (curated > all-N public mix) and the empirical curve we mapped on Playground S6E5. The S6E4 post-mortem at `~/knowledge-graph/kaggle/2026-16_s6e4-postmortem.md` covers L1-L18.

## License

MIT — see [LICENSE](LICENSE).
