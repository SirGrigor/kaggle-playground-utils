"""Build the S6E4 signal factory tutorial notebook.

Run: uv run python notebooks/_build_notebook.py
Output: notebooks/s6e4_signal_factory_tutorial.ipynb
"""
import json
import hashlib
from pathlib import Path
from textwrap import dedent


def _id(src: str) -> str:
    """Deterministic cell id from content — stable across rebuilds."""
    return hashlib.sha1(src.encode()).hexdigest()[:12]


def md(src: str) -> dict:
    body = dedent(src).strip("\n")
    return {
        "cell_type": "markdown",
        "id": _id(body),
        "metadata": {},
        "source": body,
    }


def code(src: str) -> dict:
    body = dedent(src).strip("\n")
    return {
        "cell_type": "code",
        "id": _id(body),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": body,
    }


CELLS = []

CELLS.append(md("""
# Signal Factory — Automated Feature Discovery via MI + Orthogonality

**Playground Series S6E4 — Predicting Irrigation Need**

This notebook shares a **reusable methodology**, not a final solution.

---

## What you get

A small self-contained toolkit that:
1. Generates ~1,500–3,000 candidate features from raw numerics (9 transformation families)
2. Ranks them by mutual information with the target
3. **Deduplicates against features you already have** — so it doesn't hand back slight variants of what you've engineered by hand

Runs in ~5 minutes on 50k sampled rows. No external dependencies beyond `numpy`, `pandas`, `sklearn`.

## What you DON'T get

- Not a top-tier LB solution — I closed my S6E4 at **LB 0.97654** (pure-solo best, no public-kernel blending). Below the 98%+ tier.
- Not a magic feature generator — this tool systematizes hypotheses you'd test by hand, ~500× faster.

## Honest observation that motivated this

I tried "pick top-K by mutual information" on S6E4 several times. Each time, the top-K were **redundant variants of the same underlying pattern** — 6 sin/cos frequencies of the same column, or 3 thresholds at nearby percentiles. Adding them all made things **worse**, not better.

That failure is why this notebook has two layers:
1. **Orthogonality dedup during discovery** (Section 3) — don't surface candidates that duplicate your existing features
2. **Greedy forward selection after discovery** (Section 6) — don't assume top-K by MI are mutually orthogonal either

If you find this useful, or spot a bug, or have improvement ideas — comments very welcome. I'd especially love feedback from anyone who's built something similar.
"""))

CELLS.append(md("""
## TL;DR — what to take away

1. **MI is a good FIRST filter** for candidate features, but **not a good selection criterion** when candidates correlate with each other.
2. **Always dedup candidates against features you already have** — otherwise the factory "discovers" features your hand-crafted pipeline already encodes.
3. **For final selection: use greedy forward selection with a marginal-lift gate.** Don't just take top-K — measure marginal contribution instead. Section 6 shows the skeleton.

Total code in this notebook: ~200 lines of pure Python / numpy / sklearn.
"""))

CELLS.append(md("""
## Why this tool exists

When you start a Playground comp, Phase 1 is always: *which feature transformations help?* You try things like:

- `log(x)`, `sqrt(x)`, `x²`, `1/x`, `x^0.3` ...
- Quantile bins at 5, 10, 20, 50 levels
- `x > threshold` booleans across percentiles
- Pairwise ratios, products, differences
- Maybe `sin(x * 0.1)` if you're feeling exotic
- Distance to cluster centroids at k ∈ {3, 5, 8}

Testing each of these by hand — one transformation per version, retrain, compare CV — is ~15 minutes per hypothesis. Over 30 hypotheses: **7.5 hours of grind**, and you're not even sure you covered the right search space.

This factory tests **~1,700 hypotheses in 5 minutes** using mutual information as a fast proxy.

**Important caveat**: MI doesn't guarantee a feature will HELP a tree ensemble — just that it shares statistical information with the target. Tree models often already recover that signal from raw columns via their own splits. The factory is a **starting point** — you still validate via actual CV.
"""))

CELLS.append(code("""
import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
"""))

CELLS.append(code("""
import os

# Auto-detect data path — works on Kaggle (with competition attached) or locally
def find_data_dir():
    candidates = [
        "/kaggle/input/playground-series-s6e4",
        "/kaggle/input/playground-series-season-6-episode-4",
    ]
    for p in candidates:
        if os.path.exists(f"{p}/train.csv"):
            return p
    # Fallback: search /kaggle/input recursively for train.csv
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "train.csv" in files:
                return root
    raise FileNotFoundError(
        "Competition data not found. On Kaggle: attach 'Playground Series S6E4' "
        "via '+ Add data' in the sidebar. Locally: set DATA_DIR manually."
    )

DATA_DIR = find_data_dir()
print(f"Using DATA_DIR = {DATA_DIR}")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
print(f"Train shape: {train.shape}")

TARGET = "Irrigation_Need"
TARGET_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
y = train[TARGET].map(TARGET_MAPPING).values

NUMERIC = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CATEGORICAL = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]

train_features = train.drop(columns=["id", TARGET])
print(f"Numeric columns: {len(NUMERIC)}")
print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
"""))

CELLS.append(md("""
## Section 1 — Nine transformation families

Each family is a function `(df, numeric_cols) -> dict[name, (array, description)]`. Adding a new family is one more function + one entry in `BUILDERS`.

| Family | Generates | Why it helps |
|---|---|---|
| `arithmetic` | log, sqrt, square, cube, inv, inv², pow^0.3/0.7, log2, exp(−x) | Common monotone transforms; `inv_sq` + `log2` handle heavy-tailed numerics |
| `binning` | quantile bins at 5/10/20/50 resolutions | Coarse categorical features from continuous; helps trees at low leaf counts |
| `thresholds` | `x < p` at 100 percentiles | Hunt for single-cutoff rules common in physical-process data |
| `pairwise` | products, ratios, diffs, abs-diffs | Interactions trees miss without explicit features |
| `rank` | percentile rank | Monotone but distribution-free — defeats outliers |
| `distance` | `|x − mean|`, `|x − median|` | Extremeness signal; helps when target depends on deviation |
| `mod` | `int(x*100) mod m` for 13 divisors | Generator-artifact detection in synthetic Playground data |
| `exotic` | sin/cos at 7 frequencies | Periodic signal detection (rare but critical when present) |
| `cluster_distance` | distance to KMeans centroids at k=3,5,8 | Sub-population structure trees can't derive from axis-aligned splits |

Total for 11 numeric columns: **~1,700 candidates**.
"""))

CELLS.append(code("""
# Nine builder functions — each returns dict[name, (array, description)]

def _build_arithmetic(df: pd.DataFrame, numeric: list) -> dict:
    out = {}
    for c in numeric:
        v = df[c].values.astype(np.float64)
        out[f"log1p_{c}"] = (np.log1p(np.abs(v)), f"log(1+|{c}|)")
        out[f"sqrt_{c}"] = (np.sqrt(np.abs(v)), f"sqrt(|{c}|)")
        out[f"square_{c}"] = (v ** 2, f"{c}²")
        out[f"cube_{c}"] = (v ** 3, f"{c}³")
        out[f"inv_{c}"] = (1.0 / (np.abs(v) + 1), f"1/(|{c}|+1)")
        out[f"inv_sq_{c}"] = (1.0 / (v ** 2 + 1), f"1/({c}²+1)")
        out[f"pow03_{c}"] = (np.abs(v) ** 0.3, f"|{c}|^0.3")
        out[f"pow07_{c}"] = (np.abs(v) ** 0.7, f"|{c}|^0.7")
        out[f"log2_{c}"] = (np.log2(np.abs(v) + 1), f"log2(|{c}|+1)")
        out[f"exp_small_{c}"] = (np.exp(-0.1 * np.abs(v)), f"exp(-0.1*|{c}|)")
    return out


def _build_binning(df: pd.DataFrame, numeric: list) -> dict:
    out = {}
    for c in numeric:
        v = df[c].values
        for n_bins in [5, 10, 20, 50]:
            try:
                bins = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
                if len(bins) < 3:
                    continue
                out[f"qbin{n_bins}_{c}"] = (
                    np.digitize(v, bins[1:-1]),
                    f"{c} quantile-binned ({n_bins})",
                )
            except Exception:
                continue
    return out


def _build_thresholds(df: pd.DataFrame, numeric: list, n_scan: int = 100) -> dict:
    out = {}
    for c in numeric:
        v = df[c].values
        percentiles = np.linspace(2, 98, n_scan)
        thresholds = np.quantile(v, percentiles / 100)
        for pct, thr in zip(percentiles, thresholds):
            out[f"lt_{c}_p{int(pct)}"] = (
                (v < thr).astype(int),
                f"{c} < {thr:.3f} (p{int(pct)})",
            )
    return out


def _build_pairwise(df: pd.DataFrame, numeric: list) -> dict:
    out = {}
    cols = numeric[:min(len(numeric), 8)]  # cap combinatorial cost
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            v1 = df[c1].values.astype(np.float64)
            v2 = df[c2].values.astype(np.float64)
            out[f"prod_{c1}_x_{c2}"] = (v1 * v2, f"{c1} × {c2}")
            out[f"ratio_{c1}_by_{c2}"] = (v1 / (np.abs(v2) + 0.1), f"{c1} / {c2}")
            out[f"diff_{c1}_{c2}"] = (v1 - v2, f"{c1} − {c2}")
            out[f"absdiff_{c1}_{c2}"] = (np.abs(v1 - v2), f"|{c1} − {c2}|")
    return out


def _build_rank(df: pd.DataFrame, numeric: list) -> dict:
    return {
        f"rankpct_{c}": (pd.Series(df[c].values).rank(pct=True).values, f"pct-rank of {c}")
        for c in numeric
    }


def _build_distance(df: pd.DataFrame, numeric: list) -> dict:
    out = {}
    for c in numeric:
        v = df[c].values.astype(np.float64)
        out[f"dist_mean_{c}"] = (np.abs(v - v.mean()), f"|{c} − mean({c})|")
        out[f"dist_median_{c}"] = (np.abs(v - np.median(v)), f"|{c} − median({c})|")
    return out


def _build_mod(df: pd.DataFrame, numeric: list) -> dict:
    out = {}
    for c in numeric:
        v_int = (df[c].values * 100).astype(np.int64)
        for m in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 50, 100]:
            out[f"mod{m}_{c}"] = (v_int % m, f"int({c}*100) mod {m}")
    return out


def _build_exotic(df: pd.DataFrame, numeric: list) -> dict:
    out = {}
    frequencies = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    for c in numeric:
        v = df[c].values.astype(np.float64)
        for f in frequencies:
            out[f"sin{f}_{c}"] = (np.sin(v * f), f"sin({c}*{f})")
            out[f"cos{f}_{c}"] = (np.cos(v * f), f"cos({c}*{f})")
    return out


def _build_cluster_distance(df: pd.DataFrame, numeric: list) -> dict:
    if len(numeric) < 2:
        return {}
    X_scaled = StandardScaler().fit_transform(df[numeric].fillna(0).values)
    out = {}
    for n_clusters in [3, 5, 8]:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5).fit(X_scaled)
        for k in range(n_clusters):
            dist = np.linalg.norm(X_scaled - km.cluster_centers_[k], axis=1)
            out[f"dist_km{n_clusters}_c{k}"] = (dist, f"distance to cluster {k}/{n_clusters}")
    return out


BUILDERS = [
    ("arithmetic", _build_arithmetic),
    ("binning", _build_binning),
    ("thresholds", _build_thresholds),
    ("pairwise", _build_pairwise),
    ("rank", _build_rank),
    ("distance", _build_distance),
    ("mod", _build_mod),
    ("exotic", _build_exotic),
    ("cluster_distance", _build_cluster_distance),
]
print(f"Loaded {len(BUILDERS)} transformation families.")
"""))

CELLS.append(md("""
## Section 2 — Mutual information ranking

For fast first-pass filtering, we use `sklearn.feature_selection.mutual_info_classif`. It:

- Handles **non-linear** relationships (unlike Pearson correlation)
- Returns a single score per feature (no pairwise explosion)
- Runs on 50k sampled rows in ~30 seconds

**Caveat**: MI scores are *relative*, not absolute. `MI > 0.01` = signal exists; `MI > 0.1` = strong. Don't treat absolute values as probability.

**Why sample instead of running on full train?** MI computation scales roughly `O(N_features × N_samples)` — 50k rows is enough for reliable ranking and 10× faster than running on 500k.
"""))

CELLS.append(md("""
## Section 3 — Orthogonality deduplication

**This is the part I wish I'd added on day 1.**

The factory generates ~1,700 candidates. Many are **highly correlated** with each other:

- `square(Rainfall)` and `inv_sq(Rainfall)` — both monotone on |Rainfall|
- `lt_x_p70` and `lt_x_p72` — nearly identical booleans
- `sin(x*0.05)` and `sin(x*0.01)` — both slow-oscillation variants

If you take top-K by MI without dedup, you tend to get **K variants of the same cluster**, not K orthogonal features. Adding them to a tree ensemble is redundant at best, noise at worst.

**Fix**: compare each candidate against a pool of **existing features** (base features + candidates previously accepted). Drop anything with `|corr| > 0.80`.

In my S6E4 runs, this cut 1,700 candidates down to 20–30 genuine orthogonal signals.

### Critical gotcha — sample-index alignment

When you sample data for MI, you **must use the same row indices** for both candidates and existing features. I had a bug for a week where existing features were computed on one random sample and candidates on another — correlations came out as noise (~0.01) when the real correlation was 0.97.

The function below handles this correctly. If you adapt the code, make sure you preserve the `idx` alignment.
"""))

CELLS.append(md("""
## Section 4 — The full `discover_signals()` function

Ties everything together: generation → MI ranking → orthogonality dedup → top-N return.
"""))

CELLS.append(code("""
def discover_signals(
    train_df: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list,
    sample_size: int = 50000,
    top_n: int = 20,
    min_mi: float = 0.005,
    max_corr_with_existing: float = 0.80,
    existing_features: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    \"\"\"Generate + rank + dedup candidate features.

    Returns DataFrame with columns: feature_name, mi, description, family, max_corr_with_existing.
    Sorted by mi descending, head(top_n).
    \"\"\"
    # Sample for MI speed — keep idx for dedup alignment
    n = len(train_df)
    if sample_size < n:
        idx = np.random.RandomState(42).choice(n, sample_size, replace=False)
        df_sample = train_df.iloc[idx].reset_index(drop=True)
        y_sample = y[idx]
    else:
        idx = np.arange(n)
        df_sample = train_df
        y_sample = y

    # Generate all candidates
    if verbose:
        print(f"Generating candidates from {len(numeric_cols)} numeric cols...")
    all_candidates = {}
    for family, builder in BUILDERS:
        t0 = time.time()
        sig = builder(df_sample, numeric_cols)
        for name, (arr, desc) in sig.items():
            all_candidates[name] = {"array": arr, "description": desc, "family": family}
        if verbose:
            print(f"  {family}: {len(sig)} ({time.time()-t0:.1f}s)")
    if verbose:
        print(f"\\nTotal candidates: {len(all_candidates)}")

    # Batch MI computation
    if verbose:
        print("Computing mutual information...")
    feat_names = list(all_candidates.keys())
    X = np.column_stack([all_candidates[n]["array"] for n in feat_names])
    t0 = time.time()
    mi_scores = mutual_info_classif(X, y_sample, random_state=42, n_jobs=-1)
    if verbose:
        print(f"  MI: {time.time()-t0:.1f}s")

    # Filter by min MI
    records = [
        {
            "feature_name": name,
            "mi": float(mi),
            "description": all_candidates[name]["description"],
            "family": all_candidates[name]["family"],
            "max_corr_with_existing": 0.0,
        }
        for name, mi in zip(feat_names, mi_scores)
        if mi >= min_mi
    ]

    # Dedup against existing features — USE SAME idx
    if existing_features and records:
        existing_sample = {}
        for name, arr in existing_features.items():
            if len(arr) == n:
                existing_sample[name] = arr[idx]
            elif len(arr) == sample_size:
                existing_sample[name] = arr
        for r in records:
            cand_arr = all_candidates[r["feature_name"]]["array"]
            max_corr = 0.0
            for earr in existing_sample.values():
                try:
                    c = abs(np.corrcoef(cand_arr, earr)[0, 1])
                    if not np.isnan(c) and c > max_corr:
                        max_corr = c
                except Exception:
                    continue
            r["max_corr_with_existing"] = float(max_corr)
        records = [r for r in records if r["max_corr_with_existing"] < max_corr_with_existing]

    result = pd.DataFrame(records).sort_values("mi", ascending=False).reset_index(drop=True)
    return result.head(top_n)
"""))

CELLS.append(md("""
## Section 5 — Demo on S6E4

Pure discovery on raw numeric columns — no existing features passed, so no orthogonality dedup yet (first run always has nothing to dedup against).
"""))

CELLS.append(code("""
result = discover_signals(
    train_df=train_features,
    y=y,
    numeric_cols=NUMERIC,
    sample_size=50000,
    top_n=20,
    min_mi=0.005,
    verbose=True,
)

print("\\nTop 20 candidates by MI:")
pd.set_option("display.max_colwidth", 60)
print(result.to_string(index=False))
"""))

CELLS.append(md("""
### Interpreting the output

Columns:
- `feature_name` — unique ID you'll use when adding the feature to your pipeline
- `mi` — mutual information with `y`; higher = more signal
- `family` — which builder generated it
- `description` — human-readable formula
- `max_corr_with_existing` — 0.0 here because we didn't pass `existing_features`

**To dedup against features you already built** (e.g., after Phase 1 hand-engineering), pass them as a dict:

```python
existing = {
    "log1p_Rainfall_mm": np.log1p(train_raw["Rainfall_mm"].values),
    "my_custom_ratio": my_array,
    # ...
}
result = discover_signals(..., existing_features=existing, max_corr_with_existing=0.80)
```

The function will drop candidates with `|corr| > 0.80` against ANY of your existing features.
"""))

CELLS.append(md("""
## Section 6 — The honest failure: top-K by MI picks redundant features

When I first used this factory on S6E4 (earlier version, without dedup-vs-each-other), I took the top 7 by MI and added them all to my model. Result: **−0.00035 on CV** — actively hurt.

Looking at the picks, they were:
```
sin(Soil_Moisture * 2.0)
cos(Soil_Moisture * 1.0)
cos(Soil_Moisture * 0.5)
sin(Soil_Moisture * 1.0)
sin(Soil_Moisture * 0.5)
cos(Soil_Moisture * 0.1)
sin(Soil_Moisture * 0.2)
```

**Six sin/cos variants of the same column.** They all encode the same underlying nonlinearity in `Soil_Moisture`. The tree only needed ONE; the other six were redundant at best, adversarial noise at worst.

**The orthogonality dedup in Section 3 fixes dedup against EXISTING features, but not dedup among candidates themselves.** The tool for that is **greedy forward selection**: add candidates one-at-a-time, each time measuring marginal lift on held-out data. A candidate that duplicates what's already in gets a lift near zero and is rejected.

Skeleton below. Plug in your own `eval_fn` (e.g., 1-fold XGB on a subsample for speed, or full 5-fold CV for precision).
"""))

CELLS.append(code("""
def greedy_forward_selection(
    X_base: pd.DataFrame,
    y: np.ndarray,
    candidates: dict,          # name -> np.ndarray (aligned to X_base rows)
    eval_fn: Callable,         # (X, y) -> scalar score to maximize
    lift_threshold: float = 0.0005,
    max_rounds: int = 10,
    verbose: bool = True,
) -> list:
    \"\"\"Add features one-at-a-time by marginal lift. Stop when no candidate crosses threshold.\"\"\"
    selected = []
    remaining = dict(candidates)
    baseline = eval_fn(X_base, y)

    for round_num in range(1, max_rounds + 1):
        if not remaining:
            break
        best_name, best_lift = None, -np.inf
        for name, arr in remaining.items():
            X_try = X_base.copy()
            X_try[name] = arr
            score = eval_fn(X_try, y)
            lift = score - baseline
            if verbose:
                print(f"  round {round_num}: {name:<30s} lift={lift:+.5f}")
            if lift > best_lift:
                best_name, best_lift = name, lift

        if best_lift < lift_threshold:
            if verbose:
                print(f"  round {round_num}: no candidate crossed threshold "
                      f"({best_lift:+.5f} < {lift_threshold}). STOP.")
            break

        if verbose:
            print(f"  round {round_num}: ACCEPT {best_name} (lift {best_lift:+.5f})\\n")
        X_base = X_base.copy()
        X_base[best_name] = remaining.pop(best_name)
        selected.append(best_name)
        baseline = eval_fn(X_base, y)

    return selected


# Example eval_fn — replace with your own pipeline
# def eval_fn(X, y):
#     # e.g. 1-fold XGB on a 20% subsample for speed
#     from sklearn.model_selection import train_test_split
#     import xgboost as xgb
#     idx = np.random.RandomState(42).choice(len(X), len(X) // 5, replace=False)
#     X_s, y_s = X.iloc[idx], y[idx]
#     tr, va = train_test_split(np.arange(len(y_s)), test_size=0.2, stratify=y_s, random_state=42)
#     m = xgb.XGBClassifier(n_estimators=500, max_depth=4, early_stopping_rounds=50)
#     m.fit(X_s.iloc[tr], y_s[tr], eval_set=[(X_s.iloc[va], y_s[va])], verbose=False)
#     from sklearn.metrics import balanced_accuracy_score
#     return balanced_accuracy_score(y_s[va], m.predict(X_s.iloc[va]))

print("greedy_forward_selection() defined. Plug in your own eval_fn (CV score or 1-fold mini-test).")
"""))

CELLS.append(md("""
## Takeaways

1. **MI + orthogonality dedup finds 20–30 orthogonal signals from ~1,700 candidates** in ~5 min on a typical tabular Playground dataset.
2. **Don't take top-K from MI directly** — use greedy forward selection with a marginal-lift gate.
3. **Dedup matters on two axes**: against your existing features (Section 3) AND among candidates themselves (Section 6).
4. Biggest bug-in-waiting: **sample-index alignment** when comparing candidates to existing features. Use the same `idx` for both. Ask me how I know.

## Open questions I'd love feedback on

- Am I missing a transformation family you find regularly useful? (Target-encoding? Polynomial interactions of order 3+?)
- Is there a better fast proxy than MI for "candidate is worth testing"? I've considered LOFO+permutation but those are too slow for 1,700 candidates.
- Is the greedy mini-test lift threshold (0.0005 in balanced-accuracy terms) reasonable for S6E4-size datasets? My mini-test noise floor feels like ~±0.0005.
- Has anyone stress-tested greedy-selection on competitions where top-K-by-MI clearly fails? I only have S6E4 as a clean negative example.

## Links

- Full toolkit on GitHub: [`kaggle-playground-utils`](https://github.com/SirGrigor/kaggle-playground-utils) — includes the train/predict side I didn't share here, plus registry, blend utilities, and caching.
- [S6E4 post-mortem](https://github.com/SirGrigor/kaggle-playground-utils/blob/master/docs/s6e4-postmortem.md) — why I closed at LB 0.97654 without pushing further (spoiler: compute parity with grandmaster-employer DGX access).

Thanks for reading. Comments + fork-forks welcome.
"""))


notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


out_path = Path(__file__).parent / "s6e4_signal_factory_tutorial.ipynb"
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Wrote {out_path} ({len(CELLS)} cells, {out_path.stat().st_size} bytes)")
