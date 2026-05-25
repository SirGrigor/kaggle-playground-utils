"""Reusable model-comparison visualizations (metric-agnostic).

Promoted from playground-s6e5/src/viz.py (2026-05-25) and generalized: the
score-neutral schema, family taken from the model dict (no per-competition
version-name heuristics), and a pluggable scorer in the loader.

Each function accepts a ``models`` dict keyed by version name. Standard shape::

    {
        "v3": {
            "oof_pred":      np.ndarray,
            "holdout_pred":  np.ndarray,
            "test_pred":     np.ndarray,
            "fold_scores":   list[float],   # per-fold metric
            "holdout_score": float,
            "family":        str,           # "LGB" / "XGB" / "CatBoost" / "MLP" / ...
        },
        ...
    }

Usage::

    from kaggle_playground_utils.viz import rho_heatmap, score_rho_scatter, fold_score_boxplot
    rho_heatmap(models, on="holdout", save_path="reports/viz/rho_matrix.png")
    fold_score_boxplot(models, score_label="RMSE", lower_is_better=True,
                       save_path="reports/viz/fold_scores.png")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata


# Family → color mapping (extend as needed; unknown families fall back to grey).
FAMILY_COLORS = {
    "LGB":       "#2ca02c",
    "XGB":       "#9467bd",
    "CatBoost":  "#ff7f0e",
    "MLP":       "#1f77b4",
    "RealMLP":   "#1f77b4",
    "FTT":       "#d62728",
    "TabM":      "#8c564b",
    "TabPFN":    "#e377c2",
    "KNN":       "#17becf",
    "stack":     "#000000",
    "anchor":    "#7f7f7f",
    "Unknown":   "#cccccc",
}


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _family(models: dict[str, dict[str, Any]], name: str) -> str:
    return models[name].get("family", "Unknown")


def rho_heatmap(
    models: dict[str, dict[str, Any]],
    anchor_pred: np.ndarray | None = None,
    anchor_name: str = "anchor",
    on: str = "holdout",
    save_path: str | None = None,
    figsize: tuple = (8, 7),
    redundant_rho: float = 0.97,
    diverse_rho: float = 0.95,
) -> plt.Figure:
    """Rank-correlation (ρ) matrix of all models (+ an optional anchor).

    Args:
        models: {name: {..._pred, ...}}
        anchor_pred: optional reference predictions (e.g. current best blend);
            added as a row/col when ``on == 'test'``.
        on: which prediction array to correlate ('holdout' or 'test').
        redundant_rho/diverse_rho: annotation thresholds for the title legend.
    """
    pred_key = f"{on}_pred"
    names = list(models.keys())
    preds = [models[n][pred_key] for n in names]

    if anchor_pred is not None and on == "test":
        names.append(anchor_name)
        preds.append(anchor_pred)

    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = _rho(preds[i], preds[j])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0.92, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    for i in range(n):
        for j in range(n):
            color = "white" if M[i, j] > redundant_rho else "black"
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_title(f"ρ correlation matrix ({on} predictions)\n"
                 f"red = redundant (ρ ≥ {redundant_rho}), blue = diverse (ρ < {diverse_rho})",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label="rank correlation (ρ)", shrink=0.7)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved: {save_path}")
    return fig


def score_rho_scatter(
    models: dict[str, dict[str, Any]],
    anchor_pred: np.ndarray,
    anchor_score: float,
    score_label: str = "score",
    lower_is_better: bool = False,
    diverse_rho: float = 0.95,
    save_path: str | None = None,
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """Scatter each model at (ρ vs anchor, holdout score).

    The "winnable zone" is the region of better-than-anchor score AND low ρ
    (a blend there is mathematically likely to lift). Direction respects
    ``lower_is_better``.
    """
    names = list(models.keys())
    fams = [_family(models, n) for n in names]
    scores = [models[n]["holdout_score"] for n in names]
    rhos = [_rho(models[n]["test_pred"], anchor_pred) for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    for name, fam, sc, rho in zip(names, fams, scores, rhos):
        color = FAMILY_COLORS.get(fam, FAMILY_COLORS["Unknown"])
        ax.scatter(rho, sc, s=180, c=color, edgecolors="black", linewidths=1.2, alpha=0.85, zorder=3)
        ax.annotate(name, (rho, sc), xytext=(7, 7), textcoords="offset points", fontsize=9, zorder=4)

    ax.scatter([1.0], [anchor_score], s=300, c=FAMILY_COLORS["anchor"], marker="*",
               edgecolors="black", linewidths=1.5, zorder=3, label=f"anchor ({score_label} {anchor_score:.5f})")
    ax.axhline(anchor_score, color=FAMILY_COLORS["anchor"], ls="--", lw=1, alpha=0.5)
    ax.axvline(diverse_rho, color="red", ls="--", lw=1, alpha=0.4, label=f"ρ={diverse_rho} (diversity threshold)")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Better-than-anchor band depends on metric direction.
    band = [ylim[0], anchor_score] if lower_is_better else [anchor_score, ylim[1]]
    ax.fill_between([xlim[0], diverse_rho], band[0], band[1],
                    color="green", alpha=0.08, zorder=1, label="winnable zone (blend likely lifts)")

    from matplotlib.patches import Patch
    used_fams = sorted(set(fams))
    family_legend = [Patch(facecolor=FAMILY_COLORS.get(f, "#ccc"), edgecolor="black", label=f) for f in used_fams]
    leg1 = ax.legend(handles=family_legend, loc="lower left", title="family", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("ρ vs anchor (test predictions)", fontsize=11)
    ax.set_ylabel(f"Holdout {score_label}", fontsize=11)
    ax.set_title("Where each model sits on the paradigm map\n"
                 "(empty winnable zone ⇒ no blend lifts past the anchor)", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved: {save_path}")
    return fig


def fold_score_boxplot(
    models: dict[str, dict[str, Any]],
    score_label: str = "score",
    lower_is_better: bool = False,
    save_path: str | None = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Per-fold score distribution as a boxplot, sorted by median (best first)."""
    names = list(models.keys())
    fams = [_family(models, n) for n in names]
    fold_scores_list = [list(models[n]["fold_scores"]) for n in names]

    medians = [np.median(fs) for fs in fold_scores_list]
    order = np.argsort(medians)  # ascending
    if not lower_is_better:
        order = order[::-1]      # best (highest) first
    names = [names[i] for i in order]
    fams = [fams[i] for i in order]
    fold_scores_list = [fold_scores_list[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(fold_scores_list, labels=names, patch_artist=True, widths=0.6, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "yellow",
                               "markeredgecolor": "black", "markersize": 7})
    for patch, fam in zip(bp["boxes"], fams):
        patch.set_facecolor(FAMILY_COLORS.get(fam, FAMILY_COLORS["Unknown"]))
        patch.set_alpha(0.7)
    for i, fs in enumerate(fold_scores_list):
        ax.scatter([i + 1] * len(fs), fs, c="black", s=20, alpha=0.5, zorder=3)

    ax.set_xlabel("model", fontsize=11)
    ax.set_ylabel(f"Fold {score_label}", fontsize=11)
    ax.set_title(f"Per-fold {score_label} distribution (sorted, best first)\n"
                 "boxes = IQR, diamond = mean, dots = individual folds", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved: {save_path}")
    return fig


def load_models_from_probs(
    probs_dir: Path | str,
    versions: list[str],
    y_holdout: np.ndarray,
    scorer: Callable[[np.ndarray, np.ndarray], float],
    families: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read oof/holdout/test arrays from ``probs/<version>/`` for each version.

    Layout (written by training scripts)::
        probs/{version}/oof.npy
        probs/{version}/holdout.npy
        probs/{version}/test.npy

    Args:
        scorer: ``scorer(y_true, y_pred) -> float`` (e.g. ``roc_auc_score`` or a
            negative-RMSE callable). Replaces the old hardcoded AUC.
        families: optional {version: family} map for coloring.
    """
    probs_dir = Path(probs_dir)
    families = families or {}
    models: dict[str, dict] = {}
    for v in versions:
        d = probs_dir / v
        oof_path, holdout_path, test_path = d / "oof.npy", d / "holdout.npy", d / "test.npy"
        if not (oof_path.exists() and holdout_path.exists() and test_path.exists()):
            print(f"  ⚠ incomplete or missing probs for {v} ({d})")
            continue
        oof, h, t = np.load(oof_path), np.load(holdout_path), np.load(test_path)
        models[v] = {
            "oof_pred": oof,
            "holdout_pred": h,
            "test_pred": t,
            "holdout_score": float(scorer(y_holdout, h)),
            "fold_scores": [],   # caller fills if fold breakdown is available
            "family": families.get(v, "Unknown"),
        }
    return models
