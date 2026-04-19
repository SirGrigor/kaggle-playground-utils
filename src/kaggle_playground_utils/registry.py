"""registry.py — model registration and probs persistence.

Target (Day 3): every trained model auto-saves OOF/holdout/test probs
indexed by a registry_id derived from the training config hash.

Enables retroactive blending: the blender just iterates over registry/*.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelRecord:
    """Metadata for a single trained model."""
    registry_id: str
    algo: str                              # "xgb", "lgb", "catboost", "mlp", etc.
    params: dict[str, Any]
    cv_seed: int
    model_seed: int
    n_folds: int
    n_features: int
    feature_hash: str                      # identifies the feature set used
    fold_scores: list[float]
    oof_score: float
    holdout_score: float | None = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def make_registry_id(algo: str, params: dict, feature_hash: str,
                     cv_seed: int, model_seed: int) -> str:
    """Deterministic ID: hash of config parts."""
    key = json.dumps({
        "algo": algo,
        "params": params,
        "feature_hash": feature_hash,
        "cv_seed": cv_seed,
        "model_seed": model_seed,
    }, sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{algo}_{h}"


class Registry:
    """Flat-file model registry."""

    def __init__(self, root: str | Path = "registry"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def register(self, record: ModelRecord,
                 oof_probs: np.ndarray,
                 holdout_probs: np.ndarray | None = None,
                 test_probs: np.ndarray | None = None,
                 best_cw: np.ndarray | None = None) -> Path:
        """Save all artifacts for a model under registry/<registry_id>/."""
        if not record.created_at:
            record.created_at = datetime.now().isoformat()
        model_dir = self.root / record.registry_id
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)
        np.save(model_dir / "oof_probs.npy", oof_probs)
        if holdout_probs is not None:
            np.save(model_dir / "holdout_probs.npy", holdout_probs)
        if test_probs is not None:
            np.save(model_dir / "test_probs.npy", test_probs)
        if best_cw is not None:
            np.save(model_dir / "best_cw.npy", best_cw)
        self._update_index(record)
        return model_dir

    def _update_index(self, record: ModelRecord) -> None:
        idx = self._load_index()
        # Replace existing entry with same registry_id
        idx = [r for r in idx if r["registry_id"] != record.registry_id]
        idx.append(record.to_dict())
        with open(self.index_path, "w") as f:
            json.dump(idx, f, indent=2, default=str)

    def _load_index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        with open(self.index_path) as f:
            return json.load(f)

    def list_models(self, tags: list[str] | None = None) -> list[dict]:
        """Return metadata for all registered models, optionally filtered by tag."""
        idx = self._load_index()
        if tags:
            idx = [r for r in idx if any(t in r.get("tags", []) for t in tags)]
        return idx

    def load_probs(self, registry_id: str) -> dict[str, np.ndarray]:
        """Load probs files for a single model."""
        model_dir = self.root / registry_id
        out = {}
        for name in ["oof_probs", "holdout_probs", "test_probs", "best_cw"]:
            path = model_dir / f"{name}.npy"
            if path.exists():
                out[name] = np.load(path)
        return out
