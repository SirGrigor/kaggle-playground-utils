"""Task-aware evaluation reporting with a strict primary/diagnostic split + the
error-matrix graphic UI (ported from S6E5, made multiclass-capable for S6E4).

ONE primary metric drives selection; everything threshold-dependent is read-only
diagnostic, fenced and labelled with the threshold it used. This surfaces *where*
a model deviates (per-class recall, confusion cells, calibration) so a regression
in one class can't hide behind a flat global average.

Entry point — `report(y_true, y_pred, task="auto", ...)`:
  - binary      → primary AUC; diagnostics = confusion@thr (base-rate-matched),
                  pos-class P/R/F1/support, calibration (Brier + logloss);
                  optional ROC/PR/reliability + 2x2 confusion plots.
  - multiclass  → primary macro-AUC + logloss; per-class P/R/F1 + micro/macro/
                  weighted; optional KxK confusion-matrix heatmap (the graphic UI
                  with the critical off-diagonal cells highlighted).
  - regression  → primary RMSE; MAE, R²; residual + per-segment diagnostics.

CONVENTION: `report(...)` is the training chokepoint. Call it once per variant
right after the OOF is built — it prints the primary line, the per-class
deviations, and (plots=True) writes the confusion graphic so you SEE where the
model is failing, not just the scalar.

Design choices (carried from S6E5):
  * micro/macro/weighted emitted ONLY for true multiclass.
  * binary threshold defaults to base-rate-matched (NOT 0.5).
  * expensive plots gated behind plots=True.
  * `segment=` adds per-slice metrics — where error concentrates beats any average.
"""
from __future__ import annotations

import numpy as np


# ── base-rate-matched threshold (ported from signal_hunt) ─────────────────────
def rate_matched_threshold(p: np.ndarray, base_rate: float) -> float:
    """Threshold at the (1-base_rate)-th percentile so predicted-positive count
    matches y's positive count. Robust to uncalibrated scores."""
    return float(np.quantile(np.asarray(p, dtype=float), 1.0 - base_rate))


def auc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_prob))


# ── task detection ────────────────────────────────────────────────────────────
def detect_task(y_true, y_pred) -> str:
    """binary | multiclass | regression, from label/prediction shapes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2 and y_pred.shape[1] > 2:
        return "multiclass"
    uniq = np.unique(y_true[~np.isnan(y_true)] if y_true.dtype.kind == "f" else y_true)
    n = len(uniq)
    if n > 20 or (y_true.dtype.kind == "f" and not np.all(np.equal(np.mod(uniq, 1), 0))):
        return "regression"
    if n <= 2:
        return "binary"
    return "multiclass"


# ── helpers ───────────────────────────────────────────────────────────────────
def _fence(title: str, lines: list[str]) -> str:
    body = "\n".join(f"    {ln}" for ln in lines)
    return f"  -- {title} --\n{body}"


def _segment_metric(y_true, y_pred, segment, metric_fn) -> dict:
    out = {}
    seg = np.asarray(segment)
    for g in np.unique(seg):
        m = seg == g
        if m.sum() >= 5 and len(np.unique(np.asarray(y_true)[m])) > 1:
            try:
                out[str(g)] = float(metric_fn(np.asarray(y_true)[m],
                                              np.asarray(y_pred)[m]))
            except Exception:
                pass
    return out


# ── binary confusion at base-rate-matched threshold ───────────────────────────
def confusion_at_threshold(y, p, *, base_rate=None, threshold=None) -> dict:
    """Split rows into TP/TN/FP/FN at a base-rate-matched threshold (binary)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if threshold is None:
        br = float(y.mean()) if base_rate is None else float(base_rate)
        threshold = rate_matched_threshold(p, br)
    pred = (p > threshold).astype(int)
    tp = (pred == 1) & (y == 1)
    tn = (pred == 0) & (y == 0)
    fp = (pred == 1) & (y == 0)
    fn = (pred == 0) & (y == 1)
    counts = {k: int(m.sum()) for k, m in
              (("TP", tp), ("TN", tn), ("FP", fp), ("FN", fn))}
    precision = counts["TP"] / max(counts["TP"] + counts["FP"], 1)
    recall = counts["TP"] / max(counts["TP"] + counts["FN"], 1)
    return {
        "threshold": float(threshold), "base_rate": float(y.mean()), "n": len(y),
        "counts": counts, "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "masks": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def plot_confusion_binary(conf: dict, save_path, *, title="Confusion matrix") -> None:
    """2x2 heatmap with counts + percentages under each cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = conf["counts"]
    grid = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
    names = [["TN", "FP"], ["FN", "TP"]]
    total = max(conf["n"], 1)
    for i in range(2):
        for j in range(2):
            nm = names[i][j]
            txt = f"{nm}\n{int(grid[i, j]):,}\n({grid[i, j]/total*100:.1f}%)"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black",
                    fontsize=10)
    ax.set_title(f"{title}\nthr={conf['threshold']:.4f}  "
                 f"P={conf['precision']:.3f} R={conf['recall']:.3f} F1={conf['f1']:.3f}")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(save_path, dpi=130); plt.close(fig)


def plot_confusion_multiclass(y_true, pred, save_path, *, class_names=None,
                              title="Confusion matrix", normalize="true") -> None:
    """KxK confusion-matrix heatmap. The graphic UI: diagonal = correct, the
    bright OFF-diagonal cells are the critical deviations (which class is bleeding
    into which). normalize='true' shows per-true-class recall on the diagonal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    labels = np.unique(np.concatenate([y_true, pred]))
    cm = confusion_matrix(y_true, pred, labels=labels)
    raw = cm.copy()
    if normalize == "true":
        with np.errstate(all="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    names = class_names or [str(x) for x in labels]
    k = len(labels)
    fig, ax = plt.subplots(figsize=(1.6 * k + 3, 1.4 * k + 2.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize == "true" else None)
    ax.set_xticks(range(k)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(k)); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(k):
        for j in range(k):
            frac = cm[i, j]
            cell = (f"{frac:.2f}\n{raw[i, j]:,}" if normalize == "true"
                    else f"{raw[i, j]:,}")
            ax.text(j, i, cell, ha="center", va="center",
                    color="white" if frac > thresh else "black", fontsize=9)
    # per-class recall (diagonal) called out in the title — the deviation summary
    recalls = np.diag(raw) / np.clip(raw.sum(axis=1), 1, None)
    rec_txt = "  ".join(f"{names[i]}={recalls[i]:.3f}" for i in range(k))
    ax.set_title(f"{title}\nper-class recall: {rec_txt}", fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(save_path, dpi=130); plt.close(fig)


# ── binary report ─────────────────────────────────────────────────────────────
def _binary_report(y_true, p, *, threshold, label, plots, plot_dir,
                   segment, segment_name, verbose) -> dict:
    from sklearn.metrics import (precision_recall_fscore_support,
                                 brier_score_loss, log_loss, roc_auc_score)
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    primary = auc(y_true, p)
    try:
        brier = float(brier_score_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7)))
        ll = float(log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7)))
    except Exception:
        brier = ll = float("nan")
    conf = confusion_at_threshold(
        y_true, p, threshold=None if threshold == "base_rate" else float(threshold))
    thr = conf["threshold"]
    pred = (p > thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    out = {
        "task": "binary", "primary_metric": "auc", "auc": primary,
        "brier": brier, "logloss": ll, "threshold": thr,
        "precision_pos": float(pr), "recall_pos": float(rc), "f1_pos": float(f1),
        "support_pos": int((y_true == 1).sum()), "n": len(y_true),
        "confusion": conf["counts"],
    }
    if segment is not None:
        out["auc_by_segment"] = _segment_metric(y_true, p, segment, roc_auc_score)
    if verbose:
        pfx = f"[{label}] " if label else ""
        print(f"{pfx}PRIMARY  auc={primary:.5f}")
        print(_fence(f"diagnostic (threshold={thr:.4f}, base-rate-matched)", [
            f"precision={pr:.4f}  recall={rc:.4f}  f1={f1:.4f}  support(pos)={out['support_pos']:,}",
            f"confusion  TP={conf['counts']['TP']:,} FP={conf['counts']['FP']:,} "
            f"FN={conf['counts']['FN']:,} TN={conf['counts']['TN']:,}",
        ]))
        print(_fence("calibration (AUC-blind)", [f"brier={brier:.5f}  logloss={ll:.5f}"]))
        if out.get("auc_by_segment"):
            worst = sorted(out["auc_by_segment"].items(), key=lambda kv: kv[1])[:3]
            print(_fence(f"auc by {segment_name} (worst 3)",
                         [f"{g}: {v:.4f}" for g, v in worst]))
    if plots:
        from pathlib import Path
        d = Path(plot_dir or "."); d.mkdir(parents=True, exist_ok=True)
        stub = (label or "eval").replace(" ", "_")
        plot_confusion_binary(conf, d / f"{stub}_confusion.png", title=label or "confusion")
        out["confusion_plot"] = str(d / f"{stub}_confusion.png")
    return out


# ── multiclass report (+ confusion graphic) ───────────────────────────────────
def _multiclass_report(y_true, proba, *, label, plots, plot_dir,
                       class_names, segment, segment_name, verbose) -> dict:
    from sklearn.metrics import (classification_report, log_loss,
                                 roc_auc_score, balanced_accuracy_score)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)
    rep = classification_report(y_true, pred, output_dict=True, zero_division=0)
    try:
        macro_auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc = float("nan")
    try:
        ll = float(log_loss(y_true, proba))
    except Exception:
        ll = float("nan")
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    per_class = {k: v for k, v in rep.items()
                 if k not in ("accuracy", "macro avg", "weighted avg")}
    out = {"task": "multiclass", "primary_metric": "macro_auc",
           "macro_auc": macro_auc, "logloss": ll,
           "balanced_accuracy": bal_acc,
           "micro_f1": rep.get("accuracy"),
           "macro_f1": rep["macro avg"]["f1-score"],
           "weighted_f1": rep["weighted avg"]["f1-score"],
           "per_class": per_class}
    if segment is not None:
        out["balacc_by_segment"] = _segment_metric(
            y_true, pred, segment, balanced_accuracy_score)
    if verbose:
        pfx = f"[{label}] " if label else ""
        print(f"{pfx}PRIMARY  macro_auc={macro_auc:.5f}  logloss={ll:.5f}  "
              f"balanced_acc={bal_acc:.5f}")
        # per-class recall/precision deviations — the critical bit
        names = class_names or sorted(per_class.keys())
        lines = []
        for cls in per_class:
            nm = (class_names[int(cls)] if class_names and str(cls).isdigit()
                  and int(cls) < len(class_names) else str(cls))
            v = per_class[cls]
            lines.append(f"{nm:>10}: recall={v['recall']:.4f}  "
                         f"precision={v['precision']:.4f}  f1={v['f1-score']:.4f}  "
                         f"support={int(v['support']):,}")
        print(_fence("per-class (deviation watch — weakest recall is the cap)", lines))
        print(_fence("averages (multiclass — meaningful here)", [
            f"micro/acc={out['micro_f1']:.4f}  macro_f1={out['macro_f1']:.4f}  "
            f"weighted_f1={out['weighted_f1']:.4f}"]))
        if out.get("balacc_by_segment"):
            worst = sorted(out["balacc_by_segment"].items(), key=lambda kv: kv[1])[:3]
            print(_fence(f"balanced-acc by {segment_name} (worst 3)",
                         [f"{g}: {v:.4f}" for g, v in worst]))
    if plots:
        from pathlib import Path
        d = Path(plot_dir or "."); d.mkdir(parents=True, exist_ok=True)
        stub = (label or "eval").replace(" ", "_")
        plot_confusion_multiclass(y_true, pred, d / f"{stub}_confusion.png",
                                  class_names=class_names, title=label or "confusion")
        out["confusion_plot"] = str(d / f"{stub}_confusion.png")
    return out


# ── regression report ─────────────────────────────────────────────────────────
def _regression_report(y_true, yp, *, label, plots, plot_dir,
                       segment, segment_name, verbose) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    yp = np.asarray(yp, dtype=float)
    resid = yp - y_true
    rmse = float(np.sqrt(mean_squared_error(y_true, yp)))
    out = {"task": "regression", "primary_metric": "rmse", "rmse": rmse,
           "mae": float(mean_absolute_error(y_true, yp)),
           "r2": float(r2_score(y_true, yp)),
           "resid_mean": float(resid.mean()), "resid_std": float(resid.std()),
           "n": len(y_true)}
    if segment is not None:
        out["rmse_by_segment"] = _segment_metric(
            y_true, yp, segment, lambda a, b: np.sqrt(mean_squared_error(a, b)))
    if verbose:
        pfx = f"[{label}] " if label else ""
        print(f"{pfx}PRIMARY  rmse={rmse:.5f}")
        print(_fence("diagnostic", [
            f"mae={out['mae']:.5f}  r2={out['r2']:.5f}",
            f"residual mean={out['resid_mean']:+.5f}  std={out['resid_std']:.5f} "
            f"(bias check: mean~0 is unbiased)"]))
        if out.get("rmse_by_segment"):
            worst = sorted(out["rmse_by_segment"].items(), key=lambda kv: -kv[1])[:3]
            print(_fence(f"rmse by {segment_name} (worst 3)",
                         [f"{g}: {v:.5f}" for g, v in worst]))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        d = Path(plot_dir or "."); d.mkdir(parents=True, exist_ok=True)
        stub = (label or "eval").replace(" ", "_")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].scatter(yp, resid, s=3, alpha=0.3); ax[0].axhline(0, color="k", lw=0.6)
        ax[0].set_title("residual vs predicted"); ax[0].set_xlabel("predicted")
        ax[0].set_ylabel("residual")
        ax[1].hist(resid, bins=60); ax[1].set_title("residual distribution")
        fig.tight_layout(); fig.savefig(d / f"{stub}_residuals.png", dpi=120); plt.close(fig)
    return out


# ── public entry point ────────────────────────────────────────────────────────
def report(y_true, y_pred, *, task: str = "auto", threshold="base_rate",
           label: str = "", plots: bool = False, plot_dir=None,
           class_names=None, segment=None, segment_name: str = "segment",
           verbose: bool = True) -> dict:
    """Task-aware evaluation. Prints a primary line + fenced diagnostics; returns
    a dict. The training chokepoint — call once per variant after OOF is built.

    y_pred: probabilities (binary), (n,k) proba (multiclass), or values (regression).
    plots:  ROC/PR/reliability/confusion (binary), KxK confusion heatmap
            (multiclass), or residual plots (regression) under plot_dir.
    class_names: optional label names for multiclass confusion graphic.
    segment: optional per-row group labels for per-slice metrics.
    """
    if task == "auto":
        task = detect_task(y_true, y_pred)
    if task == "binary":
        return _binary_report(y_true, y_pred, threshold=threshold, label=label,
                              plots=plots, plot_dir=plot_dir, segment=segment,
                              segment_name=segment_name, verbose=verbose)
    if task == "multiclass":
        return _multiclass_report(y_true, y_pred, label=label, plots=plots,
                                  plot_dir=plot_dir, class_names=class_names,
                                  segment=segment, segment_name=segment_name,
                                  verbose=verbose)
    if task == "regression":
        return _regression_report(y_true, y_pred, label=label, plots=plots,
                                  plot_dir=plot_dir, segment=segment,
                                  segment_name=segment_name, verbose=verbose)
    raise ValueError(f"unknown task {task!r}")
