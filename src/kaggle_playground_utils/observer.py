"""Experiment observer — hypothesis-before-result discipline, metric-agnostic.

Promoted from playground-s6e5/src/observer.py (2026-05-25) and generalized so the
same diary discipline works for *any* metric and *any* competition, not just
binary-AUC classification. The S6E5 copy stays put; this is the canonical shared
version.

Two things were generalized out of the original:

1. **No competition coupling.** The original imported ``.config.ROOT``. Here you
   call :func:`configure` once (typically from your project's ``config.py``) to
   set the project root + metric direction.
2. **Metric-neutral.** Fields are ``oof_score_*`` / ``holdout_score`` (not
   ``*_auc``), and the auto-flag detectors respect :class:`MetricSpec` so they
   work for lower-is-better metrics (RMSE, MAE, logloss) as well as
   higher-is-better (AUC, accuracy). ``predicted_delta`` / ``actual_delta`` are
   always expressed as *improvement* (positive = better) regardless of metric
   direction, so the calibration signal stays sign-consistent across comps.

Usage::

    from kaggle_playground_utils.observer import Experiment, MetricSpec, configure

    # once, e.g. in your project config.py:
    configure(
        root=ROOT,
        metric=MetricSpec(name="rmse", greater_is_better=False,
                          leak_gap=0.05, regression_drop=0.01,
                          fold_collapse_drop=0.10, fold_instability_std=0.05),
    )

    exp = Experiment.start(
        version="v3",
        parent="v2",
        hypothesis="DTW typewell-alignment features lift holdout by ~0.1 RMSE",
        predicted_delta=0.1,           # always 'expected improvement' (>0 = better)
        confidence="medium",
        feature_changes=["+ dtw_shift", "+ dtw_cost"],
        cloud_or_local="local",
    )
    # ... train + eval ...
    exp.record(
        oof_score_mean=9.41,
        oof_score_per_fold=[9.38, 9.45, 9.40, 9.43, 9.39],
        holdout_score=9.44,
        runtime_sec=183,
    )
    exp.commit()

``Experiment.start()`` enforces a non-empty hypothesis + predicted_delta.
``exp.commit()`` runs the 7 auto-flag detectors before appending to
``experiments.jsonl`` (the append-only source of truth). ``diary`` renders the
human-readable markdown views from that file.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import stdev
from typing import Any


# --------------------------------------------------------------------------- config
@dataclass
class MetricSpec:
    """Describes the competition metric so the auto-flags fire correctly.

    Thresholds are expressed in *metric units*. The defaults are tuned for
    binary AUC (~0.5–1.0 range); override them for any other metric. For
    example RMSE in the ~9 range wants much larger absolute thresholds.
    """
    name: str = "auc"
    greater_is_better: bool = True
    # auto-flag thresholds (metric units)
    fold_collapse_drop: float = 0.01      # a fold this much worse than the mean
    leak_gap: float = 0.005               # |oof_mean - holdout| above this
    regression_drop: float = 0.001        # improvement below -this vs parent
    fold_instability_std: float = 0.005   # std across folds above this

    def improvement(self, score: float, baseline: float) -> float:
        """Improvement of ``score`` over ``baseline`` (positive = better)."""
        return (score - baseline) if self.greater_is_better else (baseline - score)


@dataclass
class _ObserverConfig:
    root: Path
    metric: MetricSpec
    docs: Path

    @property
    def jsonl_path(self) -> Path:
        return self.root / "experiments.jsonl"


_CFG: _ObserverConfig | None = None


def configure(root: str | Path, metric: MetricSpec | None = None,
              docs: str | Path | None = None) -> None:
    """Set the project root + metric. Call once per project (e.g. in config.py)."""
    global _CFG
    root = Path(root).resolve()
    _CFG = _ObserverConfig(
        root=root,
        metric=metric or MetricSpec(),
        docs=Path(docs).resolve() if docs else root / "docs",
    )


def _cfg() -> _ObserverConfig:
    if _CFG is None:
        # Sensible fallback: cwd + default (AUC) metric. Projects should call
        # configure() explicitly, but this keeps ad-hoc use from crashing.
        configure(Path.cwd())
    assert _CFG is not None
    return _CFG


def jsonl_path() -> Path:
    return _cfg().jsonl_path


# --------------------------------------------------------------------------- helpers
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_cfg().root), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _load_jsonl() -> list[dict]:
    path = jsonl_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _find_entry(version: str) -> dict | None:
    for entry in _load_jsonl():
        if entry.get("version") == version:
            return entry
    return None


# --------------------------------------------------------------------------- experiment
@dataclass
class Experiment:
    # required pre-run
    version: str
    parent: str | None
    hypothesis: str
    predicted_delta: float                # expected improvement (>0 = better)
    confidence: str
    feature_changes: list[str]
    config_changes: dict[str, Any]
    pipeline_changes: list[str]
    cloud_or_local: str

    # auto-captured
    created_at: str = field(default_factory=_now_iso)
    git_sha: str | None = field(default_factory=_git_sha)

    # post-run (record())
    completed_at: str | None = None
    oof_score_mean: float | None = None
    oof_score_per_fold: list[float] | None = None
    holdout_score: float | None = None
    runtime_sec: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # post-commit (auto-fill)
    metric_name: str | None = None
    greater_is_better: bool | None = None
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    actual_delta: float | None = None     # actual improvement vs parent (>0 = better)
    parent_holdout_score: float | None = None

    @classmethod
    def start(
        cls,
        *,
        version: str,
        parent: str | None,
        hypothesis: str,
        predicted_delta: float,
        confidence: str = "medium",
        feature_changes: list[str] | None = None,
        config_changes: dict[str, Any] | None = None,
        pipeline_changes: list[str] | None = None,
        cloud_or_local: str = "local",
    ) -> "Experiment":
        if not hypothesis or not hypothesis.strip():
            raise ValueError("Experiment.start() requires a non-empty hypothesis.")
        if predicted_delta is None:
            raise ValueError("Experiment.start() requires predicted_delta (use 0.0 if truly none).")
        _COERCE = {"medium-high": "medium", "high-medium": "medium",
                   "low-medium": "low", "medium-low": "low",
                   "very high": "high", "very low": "low"}
        if confidence in _COERCE:
            print(f"  [observer] coercing confidence {confidence!r} → {_COERCE[confidence]!r}")
            confidence = _COERCE[confidence]
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(f"confidence must be low/medium/high (got {confidence!r}); "
                             f"recognized coercions: {sorted(_COERCE)}")
        if _find_entry(version) is not None:
            raise ValueError(
                f"Experiment {version!r} already exists in {jsonl_path().name}. "
                "Choose a new version name."
            )
        return cls(
            version=version,
            parent=parent,
            hypothesis=hypothesis.strip(),
            predicted_delta=float(predicted_delta),
            confidence=confidence,
            feature_changes=feature_changes or [],
            config_changes=config_changes or {},
            pipeline_changes=pipeline_changes or [],
            cloud_or_local=cloud_or_local,
        )

    def record(
        self,
        *,
        oof_score_mean: float,
        oof_score_per_fold: list[float],
        holdout_score: float,
        runtime_sec: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.oof_score_mean = float(oof_score_mean)
        self.oof_score_per_fold = [float(x) for x in oof_score_per_fold]
        self.holdout_score = float(holdout_score)
        self.runtime_sec = float(runtime_sec)
        self.completed_at = _now_iso()
        if extra:
            self.extra.update(extra)

    def note(self, text: str) -> None:
        if not text.strip():
            return
        self.notes.append(f"[{_now_iso()}] {text.strip()}")

    def _autoflag(self) -> None:
        f = self.flags
        m = _cfg().metric
        self.metric_name = m.name
        self.greater_is_better = m.greater_is_better

        if self.oof_score_per_fold is None or self.holdout_score is None or self.oof_score_mean is None:
            return

        # 1. Fold collapse — one fold far worse than the mean (direction-aware)
        if self.oof_score_per_fold:
            if m.greater_is_better:
                worst = min(self.oof_score_per_fold)
                if worst < self.oof_score_mean - m.fold_collapse_drop:
                    f.append(f"fold_collapse(worst={worst:.5f}, mean={self.oof_score_mean:.5f})")
            else:
                worst = max(self.oof_score_per_fold)
                if worst > self.oof_score_mean + m.fold_collapse_drop:
                    f.append(f"fold_collapse(worst={worst:.5f}, mean={self.oof_score_mean:.5f})")

        # 2. Methodology leak — oof and holdout should roughly agree (abs gap)
        gap = abs(self.oof_score_mean - self.holdout_score)
        if gap > m.leak_gap:
            f.append(f"methodology_leak(|oof-holdout|={gap:.5f})")

        # 3. Silent regression vs parent (improvement below -threshold)
        if self.parent:
            parent_entry = _find_entry(self.parent)
            if parent_entry and parent_entry.get("holdout_score") is not None:
                self.parent_holdout_score = float(parent_entry["holdout_score"])
                self.actual_delta = m.improvement(self.holdout_score, self.parent_holdout_score)
                if self.actual_delta < -m.regression_drop:
                    f.append(f"silent_regression(Δimprove={self.actual_delta:+.5f} vs {self.parent})")

        # 4. Fold instability
        if len(self.oof_score_per_fold) >= 2:
            fold_std = stdev(self.oof_score_per_fold)
            if fold_std > m.fold_instability_std:
                f.append(f"fold_instability(std={fold_std:.5f})")

        # 5/6. Prediction calibration (improvement vs predicted improvement)
        if self.actual_delta is not None and self.predicted_delta:
            pred = self.predicted_delta
            act = self.actual_delta
            if (pred > 0 and act > 0) or (pred < 0 and act < 0):
                abs_ratio = abs(act) / abs(pred)
                if abs_ratio < 0.5:
                    f.append(f"prediction_undershot(actual={act:+.5f} vs pred={pred:+.5f}, ratio={abs_ratio:.2f})")
                elif abs_ratio > 2.0:
                    f.append(f"prediction_overshot(actual={act:+.5f} vs pred={pred:+.5f}, ratio={abs_ratio:.2f})")
            elif pred != 0 and act != 0:
                f.append(f"prediction_sign_mismatch(actual={act:+.5f} vs pred={pred:+.5f})")

        # 7. Multiple changes → attribution ambiguous
        n_changes = (
            len(self.feature_changes)
            + len(self.pipeline_changes)
            + len(self.config_changes)
        )
        if n_changes > 1:
            f.append(f"multiple_changes(n={n_changes}) — attribution ambiguous, consider ablation")

    def commit(self) -> None:
        if self.oof_score_mean is None or self.holdout_score is None:
            raise RuntimeError(
                "Experiment.commit() requires .record() to have been called first."
            )
        self._autoflag()
        path = jsonl_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fp:
            fp.write(json.dumps(asdict(self), ensure_ascii=False) + "\n")


def add_note(version: str, text: str) -> None:
    """Append a human note to an existing experiment (rewrites the jsonl)."""
    if not text.strip():
        return
    entries = _load_jsonl()
    found = False
    for entry in entries:
        if entry.get("version") == version:
            entry.setdefault("notes", []).append(f"[{_now_iso()}] {text.strip()}")
            found = True
            break
    if not found:
        raise ValueError(f"No experiment {version!r} in {jsonl_path().name}.")
    with jsonl_path().open("w") as fp:
        for entry in entries:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
