"""Tests for post-processing utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from kaggle_playground_utils import bias_tune


def test_bias_tune_improves_known_shift():
    """Construct a 3-class OOF where argmax is biased and a bias shift provably
    raises balanced_accuracy.

    Class 2 is rare but its probabilities are systematically under-scored, so
    plain argmax never predicts it (recall=0). Adding positive bias to class 2
    must recover those recalls and raise balanced_accuracy.
    """
    rng = np.random.default_rng(0)
    n = 600
    # Two common classes, one rare class.
    y = np.concatenate([np.zeros(280, int), np.ones(280, int), np.full(40, 2)])
    proba = np.zeros((n, 3))
    for i, label in enumerate(y):
        base = rng.dirichlet(np.ones(3) * 2.0)
        # bump the true class moderately
        base[label] += 0.25
        # systematically deflate class 2 so argmax under-predicts it
        base[2] *= 0.35
        proba[i] = base / base.sum()

    base_pred = np.argmax(proba, axis=1)
    base_score = balanced_accuracy_score(y, base_pred)

    bias, tuned_score = bias_tune(proba, y, balanced_accuracy_score)

    assert tuned_score >= base_score
    assert tuned_score > base_score  # the construction guarantees a real gain
    # class 2 should get a positive bias relative to the deflated baseline
    assert bias[2] > bias[0]
    assert bias.shape == (3,)


def test_bias_tune_zero_bias_when_optimal():
    """If argmax is already optimal, tuned score must not be worse."""
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y] * 0.8 + 0.1  # confident, correct
    bias, score = bias_tune(proba, y, balanced_accuracy_score)
    assert score == 1.0
