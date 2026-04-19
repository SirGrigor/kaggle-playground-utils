"""diagnose.py — post-training diagnostics.

Target (Day 7): SHAP wrapper, learning curves, calibration plots.

Stub for Day 1.
"""
from __future__ import annotations


def shap_summary(model, X_sample):
    raise NotImplementedError("Day 7")


def learning_curve(X, y, model_factory, sizes=[0.1, 0.25, 0.5, 0.75, 1.0]):
    raise NotImplementedError("Day 7")


def calibration_plot(probs, y_true, n_bins=10):
    raise NotImplementedError("Day 7")
