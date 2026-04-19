"""config.py — canonical seeds and shared constants.

Enforces the 'same seeds across models for blending compatibility' lesson
(L1 of training-configuration-lessons). Import these from every training script.
"""
from __future__ import annotations

# ---- Seeds (override per-project via env vars or explicit args if needed) ----
CV_SEED = 42        # KFold / StratifiedKFold random_state
HOLDOUT_SEED = 42   # train_test_split random_state
MODEL_SEED = 11     # XGB/LGB/CatBoost/PyTorch random_state

# ---- Holdout fraction (when used) ----
HOLDOUT_FRAC = 0.20

# ---- Common bounds for Optuna class-weight search ----
OPTUNA_CLASS_WEIGHT_LOW = 0.5
OPTUNA_CLASS_WEIGHT_HIGH = 3.0

# ---- Metric names ----
METRIC_BALANCED_ACCURACY = "balanced_accuracy"
METRIC_MACRO_F1 = "macro_f1"
METRIC_LOG_LOSS = "log_loss"
