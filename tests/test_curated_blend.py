"""Unit tests for curated_blend module."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_playground_utils.curated_blend import (
    PublicSubmission,
    curated_mix,
    load_public_subset,
    predicted_lb_score,
    rank_norm,
    ratio_sweep,
    top_k_curated_mean,
)


# -------------------------- rank_norm --------------------------

def test_rank_norm_shape_and_range():
    x = np.array([0.1, 0.5, 0.9, 0.3])
    r = rank_norm(x)
    assert r.shape == x.shape
    assert r.min() == 0.0
    assert r.max() == 1.0


def test_rank_norm_is_order_preserving():
    x = np.array([3.0, 1.0, 2.0, 5.0])
    r = rank_norm(x)
    # Same rank order: indices 1<2<0<3 → ranks should follow
    assert r[1] < r[2] < r[0] < r[3]


def test_rank_norm_handles_ties():
    x = np.array([1.0, 1.0, 2.0, 3.0])
    r = rank_norm(x)
    assert r[0] == r[1]  # ties get same rank


def test_rank_norm_single_element():
    r = rank_norm(np.array([5.0]))
    assert r.shape == (1,)
    assert r[0] == 0.0  # 0 rank with len-1 = 0/max(1-1,1) = 0


# -------------------------- curated_mix --------------------------

def test_curated_mix_at_ratio_0_returns_baseline():
    a = np.array([0.0, 0.5, 1.0])
    b = np.array([1.0, 0.5, 0.0])
    np.testing.assert_array_equal(curated_mix(a, b, ratio=0.0), a)


def test_curated_mix_at_ratio_1_returns_public():
    a = np.array([0.0, 0.5, 1.0])
    b = np.array([1.0, 0.5, 0.0])
    np.testing.assert_array_equal(curated_mix(a, b, ratio=1.0), b)


def test_curated_mix_at_ratio_05_is_midpoint():
    a = np.array([0.0, 0.5, 1.0])
    b = np.array([1.0, 0.5, 0.0])
    expected = np.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(curated_mix(a, b, ratio=0.5), expected)


def test_curated_mix_rejects_invalid_ratio():
    a = np.array([0.5])
    with pytest.raises(ValueError):
        curated_mix(a, a, ratio=-0.1)
    with pytest.raises(ValueError):
        curated_mix(a, a, ratio=1.1)


# -------------------------- predicted_lb_score --------------------------

def test_predicted_lb_score_linear_at_ratio_0():
    """At ratio=0, predicted LB == baseline_lb (no bonus)."""
    score = predicted_lb_score(
        baseline_lb=0.954, public_avg_lb=0.955, ratio=0.0,
        rho_baseline_public=0.95,
    )
    assert score == pytest.approx(0.954, abs=1e-10)


def test_predicted_lb_score_linear_at_ratio_1():
    """At ratio=1, predicted LB == public_avg_lb (no bonus — diversity term zero)."""
    score = predicted_lb_score(
        baseline_lb=0.954, public_avg_lb=0.955, ratio=1.0,
        rho_baseline_public=0.95,
    )
    assert score == pytest.approx(0.955, abs=1e-10)


def test_predicted_lb_score_diversity_bonus_peaks_at_05():
    """The diversity bonus is symmetric around ratio=0.5."""
    s_03 = predicted_lb_score(0.95, 0.95, ratio=0.3, rho_baseline_public=0.95)
    s_05 = predicted_lb_score(0.95, 0.95, ratio=0.5, rho_baseline_public=0.95)
    s_07 = predicted_lb_score(0.95, 0.95, ratio=0.7, rho_baseline_public=0.95)
    # Bonus peaks at 0.5 (where linear is constant b/c baseline=public)
    assert s_05 > s_03
    assert s_05 > s_07


def test_predicted_lb_score_no_bonus_at_rho_1():
    """When ρ=1, the diversity bonus collapses to 0."""
    score = predicted_lb_score(
        baseline_lb=0.95, public_avg_lb=0.95, ratio=0.5,
        rho_baseline_public=1.0,
    )
    assert score == pytest.approx(0.95, abs=1e-10)


def test_predicted_lb_score_handles_nan_public():
    """NaN public_avg_lb falls back to baseline."""
    score = predicted_lb_score(
        baseline_lb=0.95, public_avg_lb=float("nan"), ratio=0.5,
        rho_baseline_public=0.95,
    )
    # Should equal baseline + small diversity bonus
    assert score >= 0.95
    assert score < 0.96


# -------------------------- load_public_subset --------------------------

def _make_manifest(entries: list[dict]) -> list[dict]:
    """Build a synthetic manifest."""
    return entries


def test_load_public_subset_filters_by_verdict():
    m = _make_manifest([
        {"verdict": "INCLUDE-TEST", "tag": "a", "claimed_lb": 0.95,
         "files_found": {"submission": "fake_a.csv"}, "votes": 10},
        {"verdict": "EDA", "tag": "b", "claimed_lb": None,
         "files_found": {"submission": None}, "votes": 5},
    ])
    # Won't load because file doesn't exist on disk
    result = load_public_subset(m, claimed_lb_threshold=None)
    assert result == []  # no files exist


def test_load_public_subset_filters_by_claimed_lb(tmp_path):
    # Create a fake submission file
    fake_csv = tmp_path / "fake.csv"
    pd.DataFrame({"id": [1, 2], "PitNextLap": [0.1, 0.9]}).to_csv(fake_csv, index=False)
    m = [
        {"verdict": "INCLUDE-TEST", "tag": "high", "slug": "user/h",
         "claimed_lb": 0.96,
         "files_found": {"submission": str(fake_csv.relative_to(tmp_path))},
         "votes": 50},
        {"verdict": "INCLUDE-TEST", "tag": "low", "slug": "user/l",
         "claimed_lb": 0.90,
         "files_found": {"submission": str(fake_csv.relative_to(tmp_path))},
         "votes": 20},
    ]
    result = load_public_subset(m, claimed_lb_threshold=0.94, repo_root=tmp_path)
    assert len(result) == 1
    assert result[0].tag == "high"


def test_load_public_subset_sorts_by_lb_desc(tmp_path):
    fake_csv = tmp_path / "f.csv"
    pd.DataFrame({"id": [1], "p": [0.5]}).to_csv(fake_csv, index=False)
    m = [
        {"verdict": "INCLUDE-TEST", "tag": "mid", "slug": "u/m",
         "claimed_lb": 0.94, "files_found": {"submission": "f.csv"}, "votes": 10},
        {"verdict": "INCLUDE-TEST", "tag": "high", "slug": "u/h",
         "claimed_lb": 0.96, "files_found": {"submission": "f.csv"}, "votes": 5},
        {"verdict": "INCLUDE-TEST", "tag": "low", "slug": "u/l",
         "claimed_lb": 0.92, "files_found": {"submission": "f.csv"}, "votes": 100},
    ]
    result = load_public_subset(m, repo_root=tmp_path)
    assert [r.tag for r in result] == ["high", "mid", "low"]


# -------------------------- top_k_curated_mean + ratio_sweep --------------------------


@pytest.fixture
def synthetic_public_subs(tmp_path):
    """Create 5 synthetic public submissions on disk."""
    test_ids = np.array([1, 2, 3, 4, 5])
    subs = []
    rng = np.random.default_rng(42)
    for i, lb in enumerate([0.96, 0.95, 0.94, 0.93, 0.92]):
        csv_path = tmp_path / f"sub_{i}.csv"
        preds = rng.uniform(0, 1, len(test_ids))
        pd.DataFrame({"id": test_ids, "p": preds}).to_csv(csv_path, index=False)
        subs.append(PublicSubmission(
            tag=f"sub_{i}", slug=f"user/sub_{i}",
            claimed_lb=lb, test_path=csv_path, votes=10,
        ))
    return subs, test_ids


def test_top_k_curated_mean_returns_correct_shape(synthetic_public_subs):
    subs, test_ids = synthetic_public_subs
    mean, selected = top_k_curated_mean(subs, K=3, test_ids=test_ids)
    assert mean.shape == (len(test_ids),)
    assert len(selected) == 3
    # Should be the top-3 by claimed_lb (already sorted)
    assert selected[0].claimed_lb == 0.96


def test_top_k_curated_mean_K_larger_than_pool(synthetic_public_subs):
    subs, test_ids = synthetic_public_subs
    mean, selected = top_k_curated_mean(subs, K=100, test_ids=test_ids)
    assert len(selected) == len(subs)


def test_top_k_curated_mean_empty_raises():
    with pytest.raises(ValueError):
        top_k_curated_mean([], K=3, test_ids=np.array([1, 2]))


def test_ratio_sweep_produces_expected_count(synthetic_public_subs):
    subs, test_ids = synthetic_public_subs
    baseline = np.array([0.5, 0.3, 0.7, 0.1, 0.9])
    results = ratio_sweep(
        baseline_test=baseline,
        public_subs=subs,
        test_ids=test_ids,
        baseline_lb_estimate=0.95,
        K_values=[2, 3],
        ratios=[0.3, 0.5, 0.7],
    )
    # 2 K values × 3 ratios = 6 results
    assert len(results) == 6
    # Results sorted by predicted_lb desc
    for i in range(len(results) - 1):
        assert results[i].predicted_lb >= results[i + 1].predicted_lb


def test_ratio_sweep_skips_K_larger_than_pool(synthetic_public_subs):
    subs, test_ids = synthetic_public_subs
    baseline = np.array([0.5, 0.3, 0.7, 0.1, 0.9])
    results = ratio_sweep(
        baseline_test=baseline,
        public_subs=subs,  # has 5 subs
        test_ids=test_ids,
        baseline_lb_estimate=0.95,
        K_values=[10, 20],  # both > 5
        ratios=[0.5],
    )
    assert results == []
