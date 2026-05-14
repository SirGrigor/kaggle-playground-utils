"""Curated public-mix blending — L19 from S6E5.

When stacking public submissions, **filter aggressively by claimed LB**.
Empirically (S6E5 2026-05-14):
  - Mean of all-33 public submissions, 20% mix → +0.00002 LB over baseline
  - Mean of top-4-LB-curated, 70% mix → +0.00017 LB over baseline

The filter is the value. This module:
  - rank_norm: scale-invariant rank normalization
  - load_public_subset: filter manifest by claimed_lb (PublicSubmission objs)
  - top_k_curated_mean: take top-K by claimed_lb → rank-space mean
  - curated_mix: (1-r)*baseline + r*public_mean in rank space
  - ratio_sweep: generate K × ratio grid with predicted_lb scoring
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def rank_norm(x: np.ndarray) -> np.ndarray:
    """Convert raw values to uniform [0,1] ranks (scale-invariant for blending)."""
    return (rankdata(x, method="average") - 1) / max(len(x) - 1, 1)


def curated_mix(
    baseline_rank: np.ndarray,
    public_mean_rank: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """Linear interpolation in rank space: (1-ratio)*baseline + ratio*public_mean.

    Args:
        baseline_rank: rank-normalized predictions from baseline model.
        public_mean_rank: rank-normalized mean of curated public submissions.
        ratio: weight on public mean. 0 = pure baseline, 1 = pure public.

    Empirical peak (S6E5): ratio ≈ 0.70 for top-4 curated.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    return (1.0 - ratio) * baseline_rank + ratio * public_mean_rank


# ---------------------------------------------------------------------------
# Public submission loading + filtering
# ---------------------------------------------------------------------------


@dataclass
class PublicSubmission:
    """One harvested public submission with metadata."""
    tag: str
    slug: str
    claimed_lb: float | None
    test_path: Path
    votes: int = 0


def load_public_subset(
    manifest: list[dict],
    claimed_lb_threshold: float | None = None,
    only_verdict: str = "INCLUDE-TEST",
    repo_root: Path | None = None,
) -> list[PublicSubmission]:
    """Filter manifest entries by quality. Returns PublicSubmission list sorted by LB desc.

    Args:
        manifest: parsed harvest manifest contents (list of dicts).
        claimed_lb_threshold: drop entries with claimed_lb below this. None = accept all.
        only_verdict: filter to this verdict. Default INCLUDE-TEST.
        repo_root: base path for resolving relative `files_found.submission` paths.
            If None, paths are taken as-is (absolute, or relative to cwd).
    """
    out: list[PublicSubmission] = []
    for entry in manifest:
        if entry.get("verdict") != only_verdict:
            continue
        claimed_lb = entry.get("claimed_lb")
        if claimed_lb_threshold is not None:
            if claimed_lb is None or claimed_lb < claimed_lb_threshold:
                continue
        files = entry.get("files_found") or {}
        sub_path = files.get("submission")
        if sub_path is None:
            continue
        full_path = (repo_root / sub_path) if repo_root else Path(sub_path)
        if not full_path.exists():
            continue
        out.append(PublicSubmission(
            tag=entry.get("tag", ""),
            slug=entry.get("slug", ""),
            claimed_lb=claimed_lb,
            test_path=full_path,
            votes=int(entry.get("votes", 0)),
        ))
    out.sort(
        key=lambda p: (
            -(p.claimed_lb if p.claimed_lb is not None else -1.0),
            -p.votes,
        )
    )
    return out


def top_k_curated_mean(
    public_subs: list[PublicSubmission],
    K: int,
    test_ids: np.ndarray,
    id_col: str = "id",
) -> tuple[np.ndarray, list[PublicSubmission]]:
    """Take top-K by claimed_lb, return (mean_rank, selected_subs)."""
    if not public_subs:
        raise ValueError("public_subs is empty — nothing to curate")
    K = min(K, len(public_subs))
    selected = public_subs[:K]
    rank_matrix = np.empty((K, len(test_ids)))
    for i, sub in enumerate(selected):
        df = pd.read_csv(sub.test_path)
        pred_col = [c for c in df.columns if c != id_col][0]
        s = df.set_index(id_col)[pred_col].loc[test_ids]
        rank_matrix[i] = rank_norm(s.to_numpy())
    return rank_matrix.mean(axis=0), list(selected)


# ---------------------------------------------------------------------------
# Ratio sweep + predicted LB scoring
# ---------------------------------------------------------------------------


@dataclass
class RatioSweepResult:
    """One (K, ratio) variant from the sweep."""
    name: str
    K: int
    ratio: float
    public_avg_claimed_lb: float
    selected_tags: list[str]
    predicted_lb: float
    rho_with_baseline: float
    test_predictions: np.ndarray = field(repr=False)


def predicted_lb_score(
    baseline_lb: float,
    public_avg_lb: float,
    ratio: float,
    rho_baseline_public: float,
    diversity_bonus_scale: float = 0.01,
) -> float:
    """Heuristic predicted LB for a curated mix variant.

    Combines:
      - Linear interpolation: (1-r)*baseline_lb + r*public_avg_lb
      - Diversity bonus: peaks at r=0.5, scales with (1 - ρ^10)
        (acknowledges that mixed often beats pure even at high ρ)

    Calibrated against S6E5 v18.007 → v19.006 data:
      - baseline=0.95406, public_avg=0.95417, ρ ≈ 0.99
      - r=0.7 measured LB = 0.95423 (bonus ≈ 0.0001)
    """
    if np.isnan(public_avg_lb):
        public_avg_lb = baseline_lb
    linear = (1.0 - ratio) * baseline_lb + ratio * public_avg_lb
    diversity_bonus = (
        4.0 * ratio * (1.0 - ratio)  # peaks at r=0.5, 0 at extremes
        * (1.0 - rho_baseline_public ** 10)  # 0 at ρ=1, ~0.1 at ρ=0.99
        * diversity_bonus_scale
    )
    return linear + diversity_bonus


def ratio_sweep(
    baseline_test: np.ndarray,
    public_subs: list[PublicSubmission],
    test_ids: np.ndarray,
    baseline_lb_estimate: float,
    K_values: Iterable[int] = (3, 4, 5, 6),
    ratios: Iterable[float] = (0.20, 0.30, 0.50, 0.70, 0.90, 1.00),
    name_prefix: str = "curated",
    id_col: str = "id",
) -> list[RatioSweepResult]:
    """Generate K × ratio grid of curated-mix candidates.

    Returns list sorted by descending predicted_lb.
    """
    baseline_rank = rank_norm(baseline_test)
    results: list[RatioSweepResult] = []
    for K in K_values:
        if K > len(public_subs):
            continue
        public_mean, selected = top_k_curated_mean(public_subs, K, test_ids, id_col=id_col)
        public_avg_lb = float(np.mean([
            s.claimed_lb for s in selected if s.claimed_lb is not None
        ])) if any(s.claimed_lb is not None for s in selected) else float("nan")
        rho_baseline_public = float(np.corrcoef(baseline_rank, public_mean)[0, 1])
        for ratio in ratios:
            mix = curated_mix(baseline_rank, public_mean, ratio)
            rho_with_baseline = float(np.corrcoef(mix, baseline_rank)[0, 1])
            predicted = predicted_lb_score(
                baseline_lb=baseline_lb_estimate,
                public_avg_lb=public_avg_lb,
                ratio=ratio,
                rho_baseline_public=rho_baseline_public,
            )
            results.append(RatioSweepResult(
                name=f"{name_prefix}_k{K}_r{int(ratio*100):02d}",
                K=K, ratio=ratio,
                public_avg_claimed_lb=public_avg_lb,
                selected_tags=[s.tag for s in selected],
                predicted_lb=predicted,
                rho_with_baseline=rho_with_baseline,
                test_predictions=mix,
            ))
    results.sort(key=lambda r: -r.predicted_lb)
    return results
