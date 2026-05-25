"""Auto-generate FE hypotheses from probes + diagnose findings.

The output is a ranked list of actionable suggestions, each with:
  - rank, priority, feature(s), finding, suggested_action, expected_lift_guess
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from . import probes, diagnose


def generate_hypotheses(
    train: pd.DataFrame,
    target: str,
    test: Optional[pd.DataFrame] = None,
    exclude: Optional[Sequence[str]] = None,
    top_k_interactions: int = 10,
    top_k_features_for_interaction: int = 8,
) -> dict:
    """Run full probe + diagnose pipeline. Returns a dict of findings + a hypothesis
    DataFrame ready for human review.
    """
    exclude = set(exclude or [])
    exclude.add(target)

    findings = {}
    hypotheses = []

    # --- Layer 1: univariate -----------------------------------------
    uni = probes.univariate_scan(train, target, exclude=exclude)
    findings["univariate"] = uni

    # Numeric non-monotonic → binning hypothesis
    for _, row in uni[uni["kind"] == "numeric"].iterrows():
        if row["monotonic"] is False and row["signal_score"] > 0.05:
            hypotheses.append({
                "priority": "HIGH",
                "type": "numeric_transform",
                "features": [row["feature"]],
                "finding": f"{row['feature']} has non-monotonic target rate (signal={row['signal_score']:.3f})",
                "action": f"Quantile-bin {row['feature']} (10-20 bins) then target-encode the bins",
                "expected_lift": "+0.0005 to +0.002",
            })
        elif row["monotonic"] and row["signal_score"] > 0.1:
            hypotheses.append({
                "priority": "MEDIUM",
                "type": "numeric_transform",
                "features": [row["feature"]],
                "finding": f"{row['feature']} monotonic with strong signal ({row['signal_score']:.3f})",
                "action": f"Try log or quantile transform; also consider {row['feature']} buckets as categorical",
                "expected_lift": "+0.0003 to +0.001",
            })

    # Categorical with high signal → target encoding hypothesis
    for _, row in uni[uni["kind"] == "categorical"].iterrows():
        if row["n_cats"] and row["n_cats"] > 2 and row["signal_score"] > 0.05:
            hypotheses.append({
                "priority": "HIGH" if row["signal_score"] > 0.15 else "MEDIUM",
                "type": "encoding",
                "features": [row["feature"]],
                "finding": f"{row['feature']} has {row['n_cats']} categories with signal={row['signal_score']:.3f}",
                "action": f"KFold target-encode {row['feature']} (smoothing=10)",
                "expected_lift": "+0.0005 to +0.002",
            })

    # --- Layer 2: bivariate interactions -----------------------------
    inter = probes.interaction_scan(
        train, target,
        top_k_features=top_k_features_for_interaction,
        exclude=exclude,
    )
    findings["interaction_scan"] = inter

    # Top-k interaction pairs → concat hypothesis
    for _, row in inter.head(top_k_interactions).iterrows():
        if row["interaction_score"] > 0.01:  # only meaningful interactions
            hypotheses.append({
                "priority": "HIGH" if row["interaction_score"] > 0.03 else "MEDIUM",
                "type": "interaction",
                "features": [row["feat1"], row["feat2"]],
                "finding": f"{row['feat1']} × {row['feat2']} interaction_score={row['interaction_score']:.4f}",
                "action": f"Create concat feature '{row['feat1']}_{row['feat2']}' and target-encode",
                "expected_lift": "+0.0005 to +0.0015",
            })

    # --- Mutual information check -----------------------------------
    mi = probes.mutual_info_scan(train, target, exclude=exclude)
    findings["mutual_info"] = mi

    # Features with high MI but low univariate signal → hidden non-linearity
    mi_df = mi["data"].set_index("feature")
    uni_df = uni.set_index("feature")
    joined = mi_df.join(uni_df, how="inner")
    for feat, row in joined.iterrows():
        # High MI + low univariate signal = non-linear signal
        if row["mi"] > joined["mi"].quantile(0.7) and row["signal_score"] < joined["signal_score"].quantile(0.3):
            hypotheses.append({
                "priority": "MEDIUM",
                "type": "nonlinear_signal",
                "features": [feat],
                "finding": f"{feat} has high MI ({row['mi']:.4f}) but low linear signal",
                "action": f"Check SHAP dependence for {feat} — likely non-linear or interaction-heavy",
                "expected_lift": "+0.0003 to +0.001",
            })

    # --- Layer 4: adversarial validation -----------------------------
    if test is not None:
        adv = probes.adversarial_validation(train, test, target=target, exclude=exclude)
        findings["adversarial"] = adv

        if adv["auc"] > 0.65:
            top_shift = adv["importances"].head(5)["feature"].tolist()
            hypotheses.append({
                "priority": "HIGH",
                "type": "distribution_shift",
                "features": top_shift,
                "finding": f"Train/test distribution SHIFT detected (adversarial AUC={adv['auc']:.3f})",
                "action": f"Investigate features {top_shift}. Consider dropping or downweighting.",
                "expected_lift": "prevents LB drop from overfitting to train",
            })
        elif adv["auc"] > 0.55:
            top_shift = adv["importances"].head(3)["feature"].tolist()
            hypotheses.append({
                "priority": "MEDIUM",
                "type": "distribution_shift",
                "features": top_shift,
                "finding": f"Mild shift (AUC={adv['auc']:.3f})",
                "action": f"Watch features {top_shift}",
                "expected_lift": "N/A (sanity check)",
            })

    # --- SHAP-based hypotheses --------------------------------------
    try:
        X = train.drop(columns=[c for c in exclude if c in train.columns])
        y = train[target].values if target in train.columns else None
        if y is not None and train[target].dtype != "object":
            # Only if target is already numeric (skip for string targets)
            y_enc = y
        elif y is not None:
            # Binary string target — encode to 0/1 (assume positive class is minority)
            counts = pd.Series(y).value_counts()
            pos = counts.idxmin()
            y_enc = (pd.Series(y) == pos).astype(int).values
        else:
            y_enc = None

        if y_enc is not None:
            model, X_enc, _ = diagnose.quick_baseline(X, y_enc)
            findings["shap_model"] = model

            shap_summary = diagnose.shap_summary(model, X_enc)
            findings["shap_summary"] = shap_summary

            # Top features from SHAP that don't match univariate top-5 → non-obvious signal
            shap_top5 = set(shap_summary["importance"].head(5)["feature"].tolist())
            uni_top5 = set(uni.head(5)["feature"].tolist())
            shap_only = shap_top5 - uni_top5
            for feat in shap_only:
                hypotheses.append({
                    "priority": "MEDIUM",
                    "type": "shap_insight",
                    "features": [feat],
                    "finding": f"{feat} ranks high in SHAP but low in univariate scan — model uses it in interactions",
                    "action": f"Run shap_dependence for {feat} to see functional form",
                    "expected_lift": "informational (confirms where interactions live)",
                })
    except Exception as e:
        findings["shap_error"] = str(e)

    # --- Compile hypotheses -----------------------------------------
    hyp_df = pd.DataFrame(hypotheses)
    if not hyp_df.empty:
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        hyp_df["_order"] = hyp_df["priority"].map(priority_order)
        hyp_df = hyp_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        hyp_df.insert(0, "rank", range(1, len(hyp_df) + 1))

    findings["hypotheses"] = hyp_df
    return findings


def format_hypotheses_summary(findings: dict) -> str:
    """Human-readable summary of hypotheses ready for stdout or writeup."""
    h = findings.get("hypotheses")
    if h is None or h.empty:
        return "No hypotheses generated (insufficient signal or error)."
    lines = ["=" * 70, "FE HYPOTHESIS LIST", "=" * 70]
    for _, row in h.iterrows():
        lines.append(f"\n[{row['rank']:2d}] [{row['priority']}] {row['type']}")
        lines.append(f"    Features: {row['features']}")
        lines.append(f"    Finding:  {row['finding']}")
        lines.append(f"    Action:   {row['action']}")
        lines.append(f"    Lift:     {row['expected_lift']}")
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
