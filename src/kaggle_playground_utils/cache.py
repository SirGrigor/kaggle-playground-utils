"""cache.py — disk-based cache for expensive feature engineering.

Target (Day 3): hash-based invalidation so the same inputs produce the same
cached result. If inputs change, cache auto-invalidates.

Key use case: S6E4 FE took 9 minutes per experiment. With cache, 9 seconds.

Usage:
    @cache_features(cache_dir="data/cache")
    def build_features(train, test, config):
        # expensive
        return train_fe, test_fe

    # First call: computes + saves
    # Second call with same args: loads from disk

Hash key derived from:
  - Function name
  - Positional arg hashes (DataFrame shape+column hashes; scalar values direct)
  - Keyword arg hashes (dict contents)
  - Function source (to invalidate if you change the FE code)
"""
from __future__ import annotations

import hashlib
import inspect
import pickle
from functools import wraps
from pathlib import Path
from typing import Callable

import pandas as pd


def _hash_df(df: pd.DataFrame) -> str:
    """Cheap DataFrame hash: shape + column names + first/last row + dtypes."""
    parts = [
        f"shape={df.shape}",
        f"cols={list(df.columns)}",
        f"dtypes={list(df.dtypes.astype(str))}",
    ]
    if len(df) > 0:
        parts.append(f"first={df.iloc[0].to_dict()}")
        parts.append(f"last={df.iloc[-1].to_dict()}")
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _hash_arg(x) -> str:
    if isinstance(x, pd.DataFrame):
        return _hash_df(x)
    if isinstance(x, (str, int, float, bool, type(None))):
        return str(x)
    if isinstance(x, (list, tuple, set)):
        return hashlib.sha256(str(x).encode()).hexdigest()[:16]
    if isinstance(x, dict):
        return hashlib.sha256(str(sorted(x.items())).encode()).hexdigest()[:16]
    try:
        return hashlib.sha256(pickle.dumps(x)).hexdigest()[:16]
    except Exception:
        return "unhashable_" + str(type(x).__name__)


def cache_features(cache_dir: str | Path = "data/cache",
                   version: str = "v1") -> Callable:
    """Decorator: cache function output keyed by argument hashes + function source.

    Args:
        cache_dir: directory for cached outputs
        version: bump to invalidate all existing caches (e.g., when upstream data changes)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func: Callable) -> Callable:
        func_source_hash = hashlib.sha256(
            inspect.getsource(func).encode()
        ).hexdigest()[:12]

        @wraps(func)
        def wrapper(*args, **kwargs):
            key_parts = [
                func.__name__,
                version,
                func_source_hash,
                *[_hash_arg(a) for a in args],
                *[f"{k}={_hash_arg(v)}" for k, v in sorted(kwargs.items())],
            ]
            key = hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:24]
            cache_file = cache_dir / f"{func.__name__}_{key}.pkl"

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            return result

        wrapper.cache_dir = cache_dir
        wrapper.clear_cache = lambda: [p.unlink() for p in cache_dir.glob(f"{func.__name__}_*.pkl")]
        return wrapper

    return decorator
