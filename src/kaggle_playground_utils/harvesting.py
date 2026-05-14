"""Bulk public-notebook harvester — Kaggle CLI wrapper + manifest builder.

For a given competition, lists top-N kernels by votes, downloads each kernel's
saved outputs, validates the submission/oof CSVs, extracts metadata, and
produces a structured manifest.

Designed to be used as:
  1. Run `bulk_harvest(comp, top_n=50, out_dir=...)` to populate harvest dir
  2. Read the produced manifest.json with `load_manifest()`
  3. Pass entries to `curated_blend.load_public_subset()` for filtering

Key features (carried over from S6E5):
  - Auto-extract `claimed_lb` from kernel title via regex
  - Auto-extract `claimed_oof_auc` from log (handles JSON-stream and plain text)
  - Validate submission against test row count + id alignment
  - Verdict categorization: INCLUDE-OOF / INCLUDE-TEST / EDA / LEAKY / WEAK / etc.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


LB_REGEX = re.compile(r"\b0\.\d{3,5}\b")

OOF_AUC_REGEXES = [
    re.compile(r"OOF AUC[^0-9]*(\d\.\d{4,5})", re.IGNORECASE),
    re.compile(r"oof_auc[^0-9]*(\d\.\d{4,5})", re.IGNORECASE),
    re.compile(r"Overall OOF[^0-9]*(\d\.\d{4,5})", re.IGNORECASE),
    re.compile(r"CV[^0-9]+score[^0-9]*(\d\.\d{4,5})", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Kaggle CLI helpers
# ---------------------------------------------------------------------------


def list_top_kernels(comp: str, n: int, timeout: int = 60) -> list[dict[str, str]]:
    """Run `kaggle kernels list -v` and parse CSV output.

    Args:
        comp: Kaggle competition slug, e.g., 'playground-series-s6e5'.
        n: number of top-voted kernels to list.

    Raises:
        RuntimeError if kaggle CLI fails (typically auth).
    """
    result = subprocess.run(
        ["kaggle", "kernels", "list", "--competition", comp,
         "--sort-by", "voteCount", "--page-size", str(n), "-v"],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        msg_parts = []
        if result.stderr.strip():
            msg_parts.append(f"stderr: {result.stderr.strip()}")
        if result.stdout.strip():
            msg_parts.append(f"stdout: {result.stdout.strip()}")
        raise RuntimeError("kaggle CLI failed — " + " | ".join(msg_parts))
    lines = [l for l in result.stdout.splitlines() if not l.startswith("Warning")]
    return list(csv.DictReader(lines))


def slug_to_tag(slug: str) -> str:
    """user/kernel-name → user_kernel-name (path-safe)."""
    return slug.replace("/", "_").replace(" ", "_")


def download_kernel(slug: str, out_dir: Path, timeout: int = 180) -> tuple[bool, str]:
    """Download all saved outputs of a public kernel. Returns (success, error_msg)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "output", slug, "-p", str(out_dir)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            err = (result.stderr or "").strip().splitlines()
            return False, err[-1] if err else "unknown CLI error"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"


# ---------------------------------------------------------------------------
# File detection within kernel output dir
# ---------------------------------------------------------------------------


def find_files(out_dir: Path) -> dict[str, Path | None]:
    """Locate oof, submission, log files in a kernel's downloaded outputs."""
    files: dict[str, Path | None] = {"oof": None, "submission": None, "log": None}

    oof_candidates = list(out_dir.rglob("*oof*.csv")) + list(out_dir.rglob("*OOF*.csv"))
    if oof_candidates:
        files["oof"] = max(oof_candidates, key=lambda p: p.stat().st_size)

    sub_candidates = list(out_dir.rglob("submission*.csv")) + list(out_dir.rglob("Submission*.csv"))
    if not sub_candidates:
        # Fallback: largest non-OOF csv
        all_csvs = list(out_dir.rglob("*.csv"))
        sub_candidates = [c for c in all_csvs if "oof" not in c.name.lower() and c != files["oof"]]
    if sub_candidates:
        files["submission"] = max(sub_candidates, key=lambda p: p.stat().st_size)

    log_candidates = list(out_dir.rglob("*.log"))
    if log_candidates:
        files["log"] = log_candidates[0]

    return files


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def extract_claimed_lb(title: str) -> float | None:
    """Find AUC-like number in kernel title. Returns highest 0.90-0.99 match."""
    matches = [float(m) for m in LB_REGEX.findall(title)]
    plausible = [v for v in matches if 0.90 <= v <= 0.99]
    return max(plausible) if plausible else None


def extract_claimed_oof_auc(log_path: Path | None) -> float | None:
    """Parse kernel log for printed OOF AUC. Handles JSON-stream + plain text."""
    if log_path is None or not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None
    # Kaggle logs are sometimes JSON-stream — concatenate stdout
    if text.lstrip().startswith("["):
        try:
            events = json.loads(text)
            text = "\n".join(e.get("data", "") for e in events if isinstance(e, dict))
        except Exception:
            pass
    candidates: list[float] = []
    for rgx in OOF_AUC_REGEXES:
        candidates += [float(m) for m in rgx.findall(text)]
    plausible = [v for v in candidates if 0.85 <= v <= 0.99]
    return max(plausible) if plausible else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_submission(sub_path: Path, test_ids: np.ndarray, id_col: str = "id") -> dict[str, Any]:
    """Check submission shape, id alignment, plausible values."""
    diag: dict[str, Any] = {"valid": False, "issues": []}
    try:
        df = pd.read_csv(sub_path)
    except Exception as e:
        diag["issues"].append(f"read failed: {e}")
        return diag
    if id_col not in df.columns:
        diag["issues"].append(f"missing '{id_col}' col")
        return diag
    pred_cols = [c for c in df.columns if c != id_col]
    if len(pred_cols) != 1:
        diag["issues"].append(f"expected 1 pred col, got {pred_cols}")
        return diag
    if len(df) != len(test_ids):
        diag["issues"].append(f"row count {len(df)} != {len(test_ids)}")
        return diag
    if set(df[id_col].to_numpy()) != set(test_ids):
        diag["issues"].append("id set mismatch")
        return diag
    preds = df[pred_cols[0]].to_numpy()
    if np.isnan(preds).any() or not np.isfinite(preds).all():
        diag["issues"].append("nan or inf predictions")
        return diag
    diag.update({
        "valid": True, "pred_col": pred_cols[0],
        "n_rows": len(df), "mean": float(preds.mean()),
        "min": float(preds.min()), "max": float(preds.max()),
    })
    return diag


def validate_oof_on_pool(
    oof_path: Path,
    pool_ids: np.ndarray,
    pool_y: np.ndarray,
    id_col: str = "id",
) -> dict[str, Any]:
    """Validate OOF predictions can be aligned to our training pool. Computes AUC."""
    from sklearn.metrics import roc_auc_score

    diag: dict[str, Any] = {"valid": False, "issues": []}
    try:
        df = pd.read_csv(oof_path)
    except Exception as e:
        diag["issues"].append(f"read failed: {e}")
        return diag
    if id_col not in df.columns:
        diag["issues"].append(f"missing '{id_col}' col")
        return diag
    pred_cols = [c for c in df.columns if c != id_col]
    if len(pred_cols) != 1:
        diag["issues"].append(f"expected 1 pred col, got {pred_cols}")
        return diag
    pool_id_set = set(int(x) for x in pool_ids)
    df_pool = df[df[id_col].isin(pool_id_set)].sort_values(id_col).reset_index(drop=True)
    if len(df_pool) < len(pool_ids) * 0.95:
        diag["issues"].append(f"only {len(df_pool)}/{len(pool_ids)} pool ids in OOF")
        return diag
    pool_id_to_y = dict(zip([int(x) for x in pool_ids], pool_y, strict=True))
    aligned_y = np.array([pool_id_to_y[int(i)] for i in df_pool[id_col]])
    aligned_pred = df_pool[pred_cols[0]].to_numpy()
    if np.isnan(aligned_pred).any():
        diag["issues"].append("nan predictions")
        return diag
    diag.update({
        "valid": True, "pred_col": pred_cols[0],
        "n_rows_in_pool": len(df_pool), "n_rows_total": len(df),
        "auc_on_our_pool": float(roc_auc_score(aligned_y, aligned_pred)),
    })
    return diag


# ---------------------------------------------------------------------------
# Verdict categorization
# ---------------------------------------------------------------------------


def categorize_entry(
    entry: dict,
    rho_redundant_threshold: float = 0.995,
    leakage_oof_minus_lb_threshold: float = 0.005,
    weak_lb_threshold: float = 0.945,
) -> str:
    """Heuristic verdict from accumulated entry data."""
    if entry.get("download_failed"):
        return "DOWNLOAD-FAIL"

    sub_valid = entry.get("submission_diag", {}).get("valid", False)
    oof_valid = entry.get("oof_diag", {}).get("valid", False)
    if not sub_valid and not oof_valid:
        return "EDA"

    claimed_oof = entry.get("claimed_oof_auc")
    claimed_lb = entry.get("claimed_lb")
    rho = entry.get("rho_with_anchor")

    if claimed_oof is not None and claimed_lb is not None:
        if claimed_oof - claimed_lb > leakage_oof_minus_lb_threshold:
            return "LEAKY"
    if claimed_lb is not None and claimed_lb < weak_lb_threshold:
        return "WEAK"
    if rho is not None and rho > rho_redundant_threshold:
        return "REDUNDANT"

    if oof_valid:
        return "INCLUDE-OOF"
    if sub_valid:
        return "INCLUDE-TEST"
    return "UNCLASSIFIED"


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def harvest_one(
    slug: str, title: str, votes: int,
    out_dir: Path,
    test_ids: np.ndarray,
    pool_ids: np.ndarray | None = None,
    pool_y: np.ndarray | None = None,
    anchor_test: np.ndarray | None = None,
    skip_existing: bool = False,
    sleep_after_download: float = 0.5,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """End-to-end harvest pipeline for one kernel."""
    tag = slug_to_tag(slug)
    kernel_dir = out_dir / tag

    entry: dict[str, Any] = {
        "slug": slug, "title": title, "votes": int(votes), "tag": tag,
        "claimed_lb": extract_claimed_lb(title),
        "download_failed": False, "download_err": None,
        "verdict": "UNCLASSIFIED",
    }

    if not (skip_existing and kernel_dir.exists() and any(kernel_dir.iterdir())):
        ok, err = download_kernel(slug, kernel_dir)
        if not ok:
            entry["download_failed"] = True
            entry["download_err"] = err
            entry["verdict"] = "DOWNLOAD-FAIL"
            return entry
        time.sleep(sleep_after_download)

    files = find_files(kernel_dir)
    base = repo_root or out_dir.parent
    entry["files_found"] = {
        k: (str(v.relative_to(base)) if v and base in v.parents else (str(v) if v else None))
        for k, v in files.items()
    }
    entry["claimed_oof_auc"] = extract_claimed_oof_auc(files["log"])

    if files["submission"]:
        entry["submission_diag"] = validate_submission(files["submission"], test_ids)
    else:
        entry["submission_diag"] = {"valid": False, "issues": ["no submission file"]}

    if files["oof"] and pool_ids is not None and pool_y is not None:
        entry["oof_diag"] = validate_oof_on_pool(files["oof"], pool_ids, pool_y)
    else:
        entry["oof_diag"] = {"valid": False, "issues": ["no OOF file or no pool labels"]}

    if anchor_test is not None and entry["submission_diag"]["valid"]:
        sub_df = pd.read_csv(files["submission"]).set_index("id").loc[test_ids]
        sub_pred = sub_df[entry["submission_diag"]["pred_col"]].to_numpy()
        from scipy.stats import rankdata
        r1 = rankdata(sub_pred, method="average")
        r2 = rankdata(anchor_test, method="average")
        entry["rho_with_anchor"] = float(np.corrcoef(r1, r2)[0, 1])

    entry["verdict"] = categorize_entry(entry)
    return entry


def bulk_harvest(
    comp: str,
    top_n: int,
    out_dir: Path,
    test_ids: np.ndarray,
    pool_ids: np.ndarray | None = None,
    pool_y: np.ndarray | None = None,
    anchor_test: np.ndarray | None = None,
    skip_existing: bool = True,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Harvest the top-N public notebooks of a competition.

    Returns the full manifest (list of entries with verdict + diagnostics).
    Side effects: writes kernel outputs to {out_dir}/{slug_tag}/ and
    {out_dir}/manifest.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kernels = list_top_kernels(comp, top_n)

    entries: list[dict[str, Any]] = []
    for i, k in enumerate(kernels, 1):
        slug = k["ref"]
        title = k["title"]
        votes = int(k.get("totalVotes", "0"))
        if progress_callback:
            progress_callback(i, len(kernels), slug, "starting")
        try:
            entry = harvest_one(
                slug, title, votes, out_dir,
                test_ids, pool_ids, pool_y, anchor_test,
                skip_existing=skip_existing,
                repo_root=repo_root,
            )
            entries.append(entry)
            if progress_callback:
                progress_callback(i, len(kernels), slug, entry["verdict"])
        except Exception as exc:
            entries.append({
                "slug": slug, "title": title, "votes": votes,
                "verdict": "EXCEPTION", "error": str(exc),
            })
            if progress_callback:
                progress_callback(i, len(kernels), slug, f"EXCEPTION: {exc}")

    (out_dir / "manifest.json").write_text(json.dumps(entries, indent=2, default=str))
    return entries


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load a harvest manifest from disk."""
    return json.loads(manifest_path.read_text())
