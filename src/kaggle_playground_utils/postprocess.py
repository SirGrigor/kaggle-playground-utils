"""Post-processing utilities for Kaggle Playground tabular competitions.

Monotone, leakage-free transforms applied to OOF / test probabilities after
model training. Designed for metrics like balanced_accuracy where the optimal
decision threshold per class differs from naive argmax.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "bias_tune",
]


def bias_tune(
    oof_proba: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    step_ladder: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002),
    n_passes: int = 3,
) -> tuple[np.ndarray, float]:
    """Coordinate-descent on per-class log-odds biases to maximize ``metric_fn``.

    Prediction rule: ``pred = argmax(log(proba) + bias)``. For each step in the
    ladder we sweep ``n_passes`` over the classes, greedily moving each class's
    bias by +/- step whenever it improves the metric. A coarse-to-fine ladder
    avoids local optima. Returns ``(best_bias, best_score)``.

    For metrics like balanced_accuracy this is a monotone post-processing step
    that typically adds ~+0.001..0.003 by correcting class-imbalance bias in the
    argmax decision boundary.
    """
    proba = np.asarray(oof_proba, dtype=float)
    y_true = np.asarray(y_true)
    n_classes = proba.shape[1]
    log_proba = np.log(np.clip(proba, 1e-12, None))

    bias = np.zeros(n_classes, dtype=float)

    def score(b: np.ndarray) -> float:
        pred = np.argmax(log_proba + b, axis=1)
        return float(metric_fn(y_true, pred))

    best_score = score(bias)

    for step in step_ladder:
        for _ in range(n_passes):
            improved = False
            for k in range(n_classes):
                for delta in (step, -step):
                    cand = bias.copy()
                    cand[k] += delta
                    s = score(cand)
                    if s > best_score:
                        bias = cand
                        best_score = s
                        improved = True
            if not improved:
                break

    return bias, best_score
