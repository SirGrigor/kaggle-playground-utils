# %% [md] Header
# # EDA Dashboard — Signal Discovery + FE Hypothesis Generator
#
# Template notebook. Set `TRAIN_PATH`, `TEST_PATH`, `TARGET`, `EXCLUDE_COLS` below
# and Run All. Works locally or in Colab.

# %% [1] Parameters
TRAIN_PATH = "../../../playground-s6e3/data/raw/train.csv"
TEST_PATH = "../../../playground-s6e3/data/raw/test.csv"
TARGET = "Churn"              # column name in train
EXCLUDE_COLS = ["id"]         # columns to ignore everywhere
TASK = "classification"       # or "regression"

# Display parameters
TOP_N_UNIVARIATE_PLOTS = 6    # show plots for top-N features by univariate signal
TOP_N_INTERACTION_PLOTS = 3   # show heatmaps for top-N interaction pairs
SHAP_SAMPLE = 5000            # rows used for SHAP (controls runtime)

# %% [2] Imports + path setup
import sys
from pathlib import Path

# Make the utils package importable from a sibling directory
ROOT = Path.cwd()
for _ in range(5):
    if (ROOT / "src" / "kaggle_playground_utils").exists():
        sys.path.insert(0, str(ROOT / "src"))
        break
    ROOT = ROOT.parent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kaggle_playground_utils import probes, diagnose, hypothesis

print(f"kaggle_playground_utils loaded from {ROOT / 'src'}")

# %% [3] Load data
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH) if Path(TEST_PATH).exists() else None

print(f"train: {train.shape}")
if test is not None:
    print(f"test:  {test.shape}")
print(f"target: {TARGET}")
print(f"target dist:\n{train[TARGET].value_counts(normalize=True)}")

# %% [4] LAYER 1 — Univariate scan
print("\n" + "=" * 60)
print("LAYER 1: UNIVARIATE SIGNAL SCAN")
print("=" * 60)

uni = probes.univariate_scan(train, TARGET, exclude=EXCLUDE_COLS)
print(uni.to_string())

# %% [5] Top univariate plots
print(f"\nPlotting top {TOP_N_UNIVARIATE_PLOTS} features by univariate signal:")
for feat in uni.head(TOP_N_UNIVARIATE_PLOTS)["feature"]:
    if pd.api.types.is_numeric_dtype(train[feat]) and train[feat].nunique() > 10:
        r = probes.numeric_decile_target_rate(train, feat, TARGET)
    else:
        r = probes.categorical_target_rate(train, feat, TARGET)
    r["plot_fn"]()
    plt.show()

# %% [6] LAYER 2 — Correlation heatmap
print("\n" + "=" * 60)
print("LAYER 2: CORRELATION HEATMAP (numeric features)")
print("=" * 60)

corr = probes.correlation_heatmap(train.drop(columns=EXCLUDE_COLS + [TARGET], errors="ignore"))
corr["plot_fn"]()
plt.show()

if not corr["collinear_pairs"].empty:
    print(f"\nCollinear pairs (|r| > 0.8):")
    print(corr["collinear_pairs"].to_string())
else:
    print("\nNo strongly collinear numeric pairs.")

# %% [7] Mutual information (catches non-linear signal)
print("\n" + "=" * 60)
print("MUTUAL INFORMATION (non-linear signal detector)")
print("=" * 60)

mi = probes.mutual_info_scan(train, TARGET, exclude=EXCLUDE_COLS, task=TASK)
print(mi["data"].to_string())
mi["plot_fn"]()
plt.show()

# %% [8] LAYER 2 — Pairwise interaction scan
print("\n" + "=" * 60)
print("INTERACTION SCAN (top features pairwise)")
print("=" * 60)

inter = probes.interaction_scan(train, TARGET, top_k_features=8, exclude=EXCLUDE_COLS)
print(inter.head(15).to_string())

print(f"\nTop {TOP_N_INTERACTION_PLOTS} interaction heatmaps:")
for _, row in inter.head(TOP_N_INTERACTION_PLOTS).iterrows():
    r = probes.interaction_heatmap(train, row["feat1"], row["feat2"], TARGET)
    r["plot_fn"]()
    plt.show()

# %% [9] LAYER 4 — Adversarial validation
if test is not None:
    print("\n" + "=" * 60)
    print("LAYER 4: ADVERSARIAL VALIDATION")
    print("=" * 60)

    adv = probes.adversarial_validation(train, test, target=TARGET)
    print(f"Adversarial AUC: {adv['auc']:.4f}  →  {adv['verdict']}")
    if adv["auc"] > 0.55:
        print(f"\nTop adversarial features (likely to distort CV):")
        print(adv["importances"].head(10).to_string())
    adv["plot_fn"]()
    plt.show()
else:
    adv = None
    print("\nNo test set provided — skipping adversarial validation.")

# %% [10] LAYER 3 — SHAP model-based insights
print("\n" + "=" * 60)
print("LAYER 3: SHAP (model-based insights)")
print("=" * 60)

X = train.drop(columns=[c for c in EXCLUDE_COLS + [TARGET] if c in train.columns])
if train[TARGET].dtype == "object":
    counts = train[TARGET].value_counts()
    pos_label = counts.idxmin()
    y = (train[TARGET] == pos_label).astype(int).values
else:
    y = train[TARGET].values

print("Training quick XGB baseline for SHAP analysis...")
model, X_enc, _ = diagnose.quick_baseline(X, y)

print("\nSHAP summary (top 20 features):")
shap_sum = diagnose.shap_summary(model, X_enc, max_display=20, sample=SHAP_SAMPLE)
shap_sum["plot_fn"]()
plt.show()
print(shap_sum["importance"].head(15).to_string())

# %% [11] SHAP dependence for top-5 features
print("\nSHAP dependence plots (top 5 features, auto-colored by strongest interactor):")
for feat in shap_sum["importance"].head(5)["feature"]:
    dep = diagnose.shap_dependence(model, X_enc, feat, interaction="auto", sample=SHAP_SAMPLE)
    dep["plot_fn"]()
    plt.show()

# %% [12] SHAP interaction detection
print("\n" + "=" * 60)
print("SHAP INTERACTION PAIRS (top 10)")
print("=" * 60)

try:
    shap_int = diagnose.shap_interactions(model, X_enc, top_k=10, sample=2000)
    print(shap_int["pairs"].to_string())
    shap_int["plot_fn"]()
    plt.show()
except Exception as e:
    print(f"SHAP interaction analysis failed: {e}")

# %% [13] Final: auto-generated FE hypotheses
print("\n" + "=" * 60)
print("AUTO-GENERATED FE HYPOTHESIS LIST")
print("=" * 60)

findings = hypothesis.generate_hypotheses(
    train, TARGET, test=test, exclude=EXCLUDE_COLS,
    top_k_interactions=10, top_k_features_for_interaction=8,
)
print(hypothesis.format_hypotheses_summary(findings))

# %% [14] Save hypothesis list to disk
import json
out_dir = Path.cwd() / "dashboard_output"
out_dir.mkdir(exist_ok=True)

h = findings["hypotheses"]
if not h.empty:
    h.to_csv(out_dir / "fe_hypotheses.csv", index=False)
    print(f"\nSaved: {out_dir / 'fe_hypotheses.csv'}")

findings["univariate"].to_csv(out_dir / "univariate_scan.csv", index=False)
findings["interaction_scan"].to_csv(out_dir / "interaction_scan.csv", index=False)
findings["mutual_info"]["data"].to_csv(out_dir / "mutual_info.csv", index=False)
if adv is not None:
    findings["adversarial"]["importances"].to_csv(out_dir / "adversarial_importance.csv", index=False)

print("All findings saved to dashboard_output/")
