"""Tests for features.py — especially the IEEE 754 digit bug (L10)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kaggle_playground_utils.features import (
    digit_features,
    threshold_booleans,
    get_cat_cols,
    categorical_one_hot,
    formula_logits,
)


class TestDigitFeatures:
    """Test precision-safe digit extraction.

    These cases caused 60%+ silent errors in S6E4 v1-v9 with naive `v // 0.01 % 10`.
    """

    def test_known_failure_values(self):
        """Values where naive floor-division on floats fails."""
        # From L10 debugging: 32.58, 56.61, 49.67, 59.14, 24.70, 48.61, 53.01, 41.91
        df = pd.DataFrame({"x": [32.58, 56.61, 49.67, 59.14, 24.70, 48.61, 53.01, 41.91]})
        out = digit_features(df, ["x"], k_range=range(-4, 4))

        # k=-2 (hundredths) correct values:
        expected_k_neg_2 = [8, 1, 7, 4, 0, 1, 1, 1]
        for i, exp in enumerate(expected_k_neg_2):
            got = int(out.iloc[i]["x_digit-2"])
            assert got == exp, f"Row {i} (v={df.iloc[i]['x']}): expected digit-2={exp}, got {got}"

    def test_k_zero_uses_floor_not_round(self):
        """For k=0 (ones place), 32.58 → 2 (floor), not 3 (round)."""
        df = pd.DataFrame({"x": [32.58]})
        out = digit_features(df, ["x"], k_range=range(0, 1))
        assert int(out.iloc[0]["x_digit0"]) == 2, \
            "k=0 must use floor (32.58 → 32 → 2), not round (33 → 3)"

    def test_k_negative_one_tenths(self):
        """Tenths digit of 32.58 is 5, not 6."""
        df = pd.DataFrame({"x": [32.58]})
        out = digit_features(df, ["x"], k_range=range(-1, 0))
        assert int(out.iloc[0]["x_digit-1"]) == 5

    def test_multi_digit_integer(self):
        """For 1234.56: k=-2→6, k=-1→5, k=0→4, k=1→3, k=2→2, k=3→1."""
        df = pd.DataFrame({"x": [1234.56]})
        out = digit_features(df, ["x"], k_range=range(-2, 4))
        expected = {-2: 6, -1: 5, 0: 4, 1: 3, 2: 2, 3: 1}
        for k, exp in expected.items():
            got = int(out.iloc[0][f"x_digit{k}"])
            assert got == exp, f"k={k}: expected {exp}, got {got}"

    def test_zero_value(self):
        """0.00 → all digits 0."""
        df = pd.DataFrame({"x": [0.00]})
        out = digit_features(df, ["x"], k_range=range(-4, 4))
        for k in range(-4, 4):
            assert int(out.iloc[0][f"x_digit{k}"]) == 0

    def test_exact_100(self):
        """100.00 → hundreds digit 1, else 0."""
        df = pd.DataFrame({"x": [100.00]})
        out = digit_features(df, ["x"], k_range=range(-2, 4))
        assert int(out.iloc[0]["x_digit2"]) == 1  # hundreds
        for k in [-2, -1, 0, 1, 3]:
            assert int(out.iloc[0][f"x_digit{k}"]) == 0


class TestThresholdBooleans:
    def test_default_lt(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        out = threshold_booleans(df, {"a": 2.5})
        assert list(out["a_lt_2.5"]) == [1, 1, 0, 0]

    def test_gt_direction(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        out = threshold_booleans(df, {"a": 2.5}, direction={"a": "gt"})
        assert list(out["a_gt_2.5"]) == [0, 0, 1, 1]


class TestCategoricalOneHot:
    def test_deterministic_columns(self):
        """Columns must match even if some values missing from data."""
        df = pd.DataFrame({"c": ["A", "B", "A"]})
        out = categorical_one_hot(df, "c", values=["A", "B", "C"])
        assert "c_A" in out.columns
        assert "c_B" in out.columns
        assert "c_C" in out.columns  # missing from data but column exists
        assert list(out["c_A"]) == [1, 0, 1]
        assert list(out["c_C"]) == [0, 0, 0]


class TestGetCatCols:
    def test_object_dtype(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        assert get_cat_cols(df) == ["a"]

    def test_mixed_dtypes(self):
        df = pd.DataFrame({
            "a": pd.Series(["x", "y"], dtype="string"),
            "b": [1, 2],
            "c": pd.Categorical(["p", "q"]),
        })
        cols = get_cat_cols(df)
        assert "a" in cols and "c" in cols and "b" not in cols
