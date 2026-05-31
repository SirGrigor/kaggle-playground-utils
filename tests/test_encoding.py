"""Tests for safe_label_encode, decimal_round_by_magnitude, drop_uniform_in_test."""

from __future__ import annotations

import pandas as pd

from kaggle_playground_utils import (
    decimal_round_by_magnitude,
    drop_uniform_in_test,
    safe_label_encode,
)


def test_safe_label_encode_rare_to_other():
    train = pd.DataFrame({"c": ["a"] * 6 + ["b"] * 6 + ["rare"] * 2})
    test = pd.DataFrame({"c": ["a", "b", "rare", "unseen"]})
    tr_enc, te_enc = safe_label_encode(train, test, ["c"], min_count=5)

    # 'rare' (count 2 < 5) folds into the "other" bucket -> code 0
    rare_code = tr_enc["c"][train["c"] == "rare"].iloc[0]
    assert rare_code == 0
    # 'a' and 'b' are kept with distinct nonzero codes
    a_code = tr_enc["c"][train["c"] == "a"].iloc[0]
    b_code = tr_enc["c"][train["c"] == "b"].iloc[0]
    assert a_code != 0 and b_code != 0 and a_code != b_code

    # test: unseen -> 0, rare -> 0, a/b -> their train codes
    assert te_enc["c"].iloc[3] == 0  # unseen
    assert te_enc["c"].iloc[2] == 0  # rare
    assert te_enc["c"].iloc[0] == a_code
    assert te_enc["c"].iloc[1] == b_code


def test_decimal_round_by_magnitude():
    df = pd.DataFrame(
        {
            "small": [1.23456, 9.99999],     # max < 10 -> 3 dp
            "mid": [12.34567, 99.99999],     # max < 100 -> 2 dp
            "big": [123.4567, 5000.6789],    # >= 100 -> 1 dp
        }
    )
    out = decimal_round_by_magnitude(df, ["small", "mid", "big"])
    assert out["small"].iloc[0] == 1.235
    assert out["mid"].iloc[0] == 12.35
    assert out["big"].iloc[0] == 123.5


def test_drop_uniform_in_test():
    train = pd.DataFrame({"a": [1, 2, 3], "const": [9, 9, 9], "b": [4, 5, 6]})
    test = pd.DataFrame({"a": [1, 2, 1], "const": [7, 7, 7], "b": [1, 1, 2]})
    tr_k, te_k, kept = drop_uniform_in_test(train, test)
    assert "const" not in kept
    assert kept == ["a", "b"]
    assert list(tr_k.columns) == ["a", "b"]
    assert list(te_k.columns) == ["a", "b"]
