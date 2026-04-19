"""Re-run just the blend analysis after fixing blend.py (tuned-probs aware).

Uses the 5 already-registered seed models from the seed sweep."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from s6e4_v14_reproduction import S6E4_ROOT
from kaggle_playground_utils.registry import Registry
from kaggle_playground_utils.blend import (
    pairwise_correlations, simple_average, nelder_mead_blend, _load_splits,
)


def main():
    reg = Registry(root=S6E4_ROOT / "registry")
    y_tr = np.load(S6E4_ROOT / "v14" / "probs" / "y_tr.npy")
    y_ho = np.load(S6E4_ROOT / "v14" / "probs" / "y_ho.npy")

    print("=" * 60)
    print("RE-BLEND (after fix: apply_tuning=True by default)")
    print("=" * 60)

    # Verify tuned probs load correctly
    print("\n[1/3] Verifying tuned-prob loading...")
    ids_raw, probs_raw = _load_splits(reg, tags=["seed_sweep"], split="holdout_probs",
                                       apply_tuning=False)
    ids_tuned, probs_tuned = _load_splits(reg, tags=["seed_sweep"], split="holdout_probs",
                                           apply_tuning=True)
    print(f"  Loaded {len(ids_tuned)} models")
    for i, id in enumerate(ids_tuned):
        raw_bacc = balanced_accuracy_score(y_ho, probs_raw[i].argmax(axis=1))
        tuned_bacc = balanced_accuracy_score(y_ho, probs_tuned[i].argmax(axis=1))
        print(f"  {id:<20} raw={raw_bacc:.5f}  tuned={tuned_bacc:.5f}  (lift +{tuned_bacc-raw_bacc:.5f})")

    # Correlations (on tuned probs)
    print("\n[2/3] Pairwise correlations (tuned):")
    corr = pairwise_correlations(reg, tags=["seed_sweep"], split="oof_probs")
    for i in range(len(corr["ids"])):
        for j in range(i+1, len(corr["ids"])):
            print(f"  {corr['ids'][i]} vs {corr['ids'][j]}: ρ = {corr['matrix'][i,j]:.5f}")

    # Blends
    print("\n[3/3] Blend results (applying each model's tuned probs):")

    avg_ho = simple_average(reg, tags=["seed_sweep"], split="holdout_probs")
    avg_bacc = balanced_accuracy_score(y_ho, avg_ho.argmax(axis=1))
    print(f"  Simple average:           {avg_bacc:.5f}")

    # Nelder-Mead on tuned OOF (use tuned probs for OOF too)
    blend = nelder_mead_blend(
        reg, y_tr, tags=["seed_sweep"], regularization=0.01,
        split_for_apply="holdout_probs",
    )
    blend_bacc = balanced_accuracy_score(y_ho, blend["applied"].argmax(axis=1))
    print(f"  Nelder-Mead (OOF-tuned):  {blend_bacc:.5f}")
    print(f"  Weights: {blend['weights']}")

    # v14 baseline
    print(f"\n  v14 baseline:             0.97423")
    print(f"  Delta (simple avg):       {avg_bacc - 0.97423:+.5f}")
    print(f"  Delta (nelder-mead):      {blend_bacc - 0.97423:+.5f}")


if __name__ == "__main__":
    main()
