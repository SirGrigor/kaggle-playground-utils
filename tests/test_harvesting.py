"""Unit tests for harvesting module — parsers + categorization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_playground_utils.harvesting import (
    categorize_entry,
    extract_claimed_lb,
    extract_claimed_oof_auc,
    slug_to_tag,
    validate_submission,
)


# -------------------------- slug_to_tag --------------------------


def test_slug_to_tag_basic():
    assert slug_to_tag("user/kernel-name") == "user_kernel-name"


def test_slug_to_tag_no_spaces():
    assert slug_to_tag("user name/kernel") == "user_name_kernel"


# -------------------------- extract_claimed_lb --------------------------


def test_extract_lb_from_title_with_decimal():
    assert extract_claimed_lb("Solution 0.95419 v2") == 0.95419


def test_extract_lb_from_title_short_decimal():
    assert extract_claimed_lb("F1 | Blender | 0.954") == 0.954


def test_extract_lb_from_title_inside_filename():
    assert extract_claimed_lb("kernel-0.95130-final") == 0.9513


def test_extract_lb_multiple_matches_picks_max():
    # When multiple plausible LB values appear, take the max
    assert extract_claimed_lb("from 0.9412 improved to 0.9550") == 0.955


def test_extract_lb_no_match():
    assert extract_claimed_lb("Predicting F1 Pit Stops") is None


def test_extract_lb_filters_implausible():
    # 0.1 and 1.5 are not LB-like (must be 0.90-0.99)
    assert extract_claimed_lb("section 0.1 figure 1.5") is None


# -------------------------- extract_claimed_oof_auc --------------------------


def test_extract_oof_from_plain_text_log(tmp_path):
    log = tmp_path / "plain.log"
    log.write_text("Some output\nOOF AUC: 0.95368\nMore output\n")
    assert extract_claimed_oof_auc(log) == 0.95368


def test_extract_oof_from_json_stream_log(tmp_path):
    """Kaggle logs are sometimes JSON-stream — list of events."""
    log = tmp_path / "json.log"
    events = [
        {"stream_name": "stdout", "data": "Training fold 1\n"},
        {"stream_name": "stdout", "data": "OOF AUC: 0.94250\n"},
        {"stream_name": "stderr", "data": "warning text\n"},
    ]
    log.write_text(json.dumps(events))
    assert extract_claimed_oof_auc(log) == 0.9425


def test_extract_oof_multiple_matches_picks_max(tmp_path):
    log = tmp_path / "multi.log"
    # Note: regex requires 4-5 decimal digits (typical OOF AUC precision)
    log.write_text("Fold OOF AUC: 0.9234\nFinal OOF AUC: 0.95421\n")
    assert extract_claimed_oof_auc(log) == 0.95421


def test_extract_oof_no_log_path():
    assert extract_claimed_oof_auc(None) is None


def test_extract_oof_missing_file(tmp_path):
    assert extract_claimed_oof_auc(tmp_path / "missing.log") is None


def test_extract_oof_case_insensitive(tmp_path):
    log = tmp_path / "case.log"
    log.write_text("oof_auc: 0.93214\n")
    assert extract_claimed_oof_auc(log) == 0.93214


# -------------------------- validate_submission --------------------------


def test_validate_submission_happy_path(tmp_path):
    test_ids = np.array([1, 2, 3, 4, 5])
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"id": test_ids, "PitNextLap": [0.1, 0.5, 0.9, 0.3, 0.7]}).to_csv(sub, index=False)
    diag = validate_submission(sub, test_ids)
    assert diag["valid"] is True
    assert diag["pred_col"] == "PitNextLap"
    assert diag["n_rows"] == 5


def test_validate_submission_wrong_row_count(tmp_path):
    test_ids = np.array([1, 2, 3, 4, 5])
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"id": [1, 2, 3], "p": [0.1, 0.5, 0.9]}).to_csv(sub, index=False)
    diag = validate_submission(sub, test_ids)
    assert diag["valid"] is False
    assert any("row count" in i for i in diag["issues"])


def test_validate_submission_id_mismatch(tmp_path):
    test_ids = np.array([1, 2, 3])
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"id": [4, 5, 6], "p": [0.1, 0.5, 0.9]}).to_csv(sub, index=False)
    diag = validate_submission(sub, test_ids)
    assert diag["valid"] is False
    assert any("id set mismatch" in i for i in diag["issues"])


def test_validate_submission_missing_id_column(tmp_path):
    test_ids = np.array([1, 2])
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"foo": [1, 2], "bar": [0.5, 0.5]}).to_csv(sub, index=False)
    diag = validate_submission(sub, test_ids)
    assert diag["valid"] is False
    assert any("missing 'id'" in i for i in diag["issues"])


def test_validate_submission_nan_predictions(tmp_path):
    test_ids = np.array([1, 2, 3])
    sub = tmp_path / "sub.csv"
    pd.DataFrame({"id": [1, 2, 3], "p": [0.1, float("nan"), 0.5]}).to_csv(sub, index=False)
    diag = validate_submission(sub, test_ids)
    assert diag["valid"] is False
    assert any("nan" in i.lower() for i in diag["issues"])


# -------------------------- categorize_entry --------------------------


def test_categorize_download_fail():
    entry = {"download_failed": True}
    assert categorize_entry(entry) == "DOWNLOAD-FAIL"


def test_categorize_no_predictions_is_eda():
    entry = {
        "download_failed": False,
        "submission_diag": {"valid": False},
        "oof_diag": {"valid": False},
    }
    assert categorize_entry(entry) == "EDA"


def test_categorize_include_test():
    entry = {
        "download_failed": False,
        "submission_diag": {"valid": True},
        "oof_diag": {"valid": False},
        "claimed_lb": 0.954,
        "claimed_oof_auc": None,
        "rho_with_anchor": 0.95,
    }
    assert categorize_entry(entry) == "INCLUDE-TEST"


def test_categorize_include_oof():
    entry = {
        "download_failed": False,
        "submission_diag": {"valid": True},
        "oof_diag": {"valid": True},
        "claimed_lb": 0.954,
        "claimed_oof_auc": 0.955,
        "rho_with_anchor": 0.97,
    }
    assert categorize_entry(entry) == "INCLUDE-OOF"


def test_categorize_leaky_when_oof_far_above_lb():
    entry = {
        "download_failed": False,
        "submission_diag": {"valid": True},
        "oof_diag": {"valid": True},
        "claimed_lb": 0.95,
        "claimed_oof_auc": 0.97,  # oof - lb = 0.02 > 0.005 threshold
        "rho_with_anchor": 0.95,
    }
    assert categorize_entry(entry) == "LEAKY"


def test_categorize_weak_low_lb():
    entry = {
        "download_failed": False,
        "submission_diag": {"valid": True},
        "oof_diag": {"valid": False},
        "claimed_lb": 0.93,  # < 0.945 threshold
        "claimed_oof_auc": None,
        "rho_with_anchor": 0.85,
    }
    assert categorize_entry(entry) == "WEAK"


def test_categorize_redundant_high_rho():
    entry = {
        "download_failed": False,
        "submission_diag": {"valid": True},
        "oof_diag": {"valid": False},
        "claimed_lb": 0.954,
        "claimed_oof_auc": None,
        "rho_with_anchor": 0.999,  # > 0.995 threshold
    }
    assert categorize_entry(entry) == "REDUNDANT"
