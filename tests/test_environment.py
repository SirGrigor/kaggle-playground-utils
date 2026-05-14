"""Unit tests for environment module."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from kaggle_playground_utils.environment import (
    _maybe_unwrap_json_token,
    detect_environment,
    setup_kaggle_auth,
)


# -------------------------- _maybe_unwrap_json_token --------------------------


def test_unwrap_plain_key_passthrough():
    """Raw API key string should pass through unchanged."""
    key = "abc123def456ghi789"
    assert _maybe_unwrap_json_token(key) == key


def test_unwrap_json_extracts_key_field():
    """If user pasted full kaggle.json, extract just the key."""
    json_token = '{"username": "user1", "key": "actualkey123"}'
    assert _maybe_unwrap_json_token(json_token) == "actualkey123"


def test_unwrap_json_with_whitespace():
    """Handles surrounding whitespace."""
    json_token = '  {"username": "u", "key": "k"}  '
    assert _maybe_unwrap_json_token(json_token) == "k"


def test_unwrap_malformed_json_passthrough():
    """If string starts with { but isn't valid JSON, fall through as raw key."""
    weird = "{not really json"
    assert _maybe_unwrap_json_token(weird) == weird


def test_unwrap_json_without_key_field():
    """JSON without 'key' field is unwrapped to itself (no change)."""
    json_no_key = '{"foo": "bar"}'
    # Doesn't have 'key' so returns original
    result = _maybe_unwrap_json_token(json_no_key)
    assert result == json_no_key


# -------------------------- detect_environment --------------------------


def test_detect_environment_returns_valid_value():
    """Either 'colab' or 'local' — always one of the two."""
    env = detect_environment()
    assert env in ("colab", "local")


# -------------------------- setup_kaggle_auth --------------------------


def test_setup_kaggle_auth_with_explicit_creds(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    ok = setup_kaggle_auth(username="alice", api_token="rawkey123", from_colab_userdata=False)
    assert ok is True
    assert os.environ["KAGGLE_USERNAME"] == "alice"
    assert os.environ["KAGGLE_KEY"] == "rawkey123"


def test_setup_kaggle_auth_unwraps_json_token(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    token = '{"username": "ignored", "key": "extractedkey"}'
    ok = setup_kaggle_auth(username="alice", api_token=token, from_colab_userdata=False)
    assert ok is True
    assert os.environ["KAGGLE_KEY"] == "extractedkey"


def test_setup_kaggle_auth_picks_up_existing_env(monkeypatch):
    """If KAGGLE_USERNAME and KAGGLE_KEY are already in env, returns True without re-setting."""
    monkeypatch.setenv("KAGGLE_USERNAME", "preset")
    monkeypatch.setenv("KAGGLE_KEY", "preset_key")
    # No explicit args, no Colab — should detect existing env
    ok = setup_kaggle_auth(from_colab_userdata=False)
    assert ok is True


def test_setup_kaggle_auth_no_credentials_anywhere(monkeypatch, tmp_path):
    """When nothing's available, returns False."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    # Point HOME to a clean dir so .kaggle/kaggle.json check fails
    monkeypatch.setenv("HOME", str(tmp_path))
    ok = setup_kaggle_auth(from_colab_userdata=False)
    assert ok is False
