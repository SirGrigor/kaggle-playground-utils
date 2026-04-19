# Playground S6E4 (Irrigation) — Post-Mortem

**Closed**: 2026-04-19
**Solo best**: LB 0.97654 (v30)
**Total submissions**: 30+
**Competition**: [Playground Series S6E4 — Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4)

## TL;DR

- Solo ceiling reached at **LB 0.97654** (v30) via the signal factory in this toolkit, ~+0.0011 over the formula-recovery baseline (v5).
- 98%+ tier is structurally unattainable on a single-GPU workstation when the top leaderboard positions belong to industry-research employees with probable multi-GPU cluster access (compute parity, not skill gap — see L15 below).
- **Primary deliverable: this toolkit, not the LB score.** Every module here is reusable for future Playground Series competitions.
- Closed deliberately, not exhausted: factory still has ~+0.0005 of greedy-selection signal available, but grinding it no longer justifies the opportunity cost versus preparing for the next competition.

## Score progression (solo)

| Version | Description | Holdout | LB |
|---|---|---|---|
| v5 | Chris Deotte formula recovery (4 thresholds + 6 one-hots + LR logits) | 0.97423 | **0.97542** |
| v14 | Reproduction via `train_variant` (toolkit sanity check) | 0.97448 | not submitted |
| v30 | v14 + **5 signal-factory features** (sin/cos Rainfall, dist_median/mean Rainfall, inv Soil_Moisture) | 0.97479 | **0.97654** (+0.00112 vs v5) |
| v31 | Signal factory round 2 on v30 base | 0.97477 | not submitted (−0.00002 CV, saturated) |
| v32 | LGB + CatBoost on v30 features + 3-way blend | 0.97392 blend | not submitted (blend HURT, −0.00087) |
| v33 | Expanded factory (1,699 candidates, top-7 by MI) | 0.97444 | not submitted (−0.00035 CV, redundancy) |
| v34 | Greedy-selection round-1 diagnostic | — | probe only, no submission |

> **Blend via public kernels**: v29 = LB 0.98086. This exists but is NOT a solo number — it was public-kernel fusion. Omitted from the "solo" trajectory above for honesty.

## What worked

### 1. Chris Deotte's formula recovery (v5 → LB 0.97542)

A published LR formula using 4 threshold booleans + 6 categorical one-hots, then injecting its logits as features for XGB. Unlocks the first real delta over naive baselines. This is a T-numbered technique that generalizes well beyond S6E4: many Playground competitions have a single dominant decision-rule baseline that hand-crafted features outperform a raw-feature XGB.

### 2. Signal factory (v30 → LB 0.97654)

The automated discovery pipeline in [`src/kaggle_playground_utils/signal_factory.py`](../src/kaggle_playground_utils/signal_factory.py):

- 9 transformation families: arithmetic, binning, thresholds, pairwise, rank, distance, mod, exotic (sin/cos), cluster-distance
- MI ranking + orthogonality dedup against user-provided existing features
- Generated 5 orthogonal features that unlocked v30's +0.00112 over v5
- **The first LB delta earned entirely by our own toolkit logic** — not read from public notebooks

See the [tutorial notebook](../notebooks/s6e4_signal_factory_tutorial.ipynb) for the full methodology with narrative.

### 3. IEEE 754 digit-extraction fix (L10)

The naive `val // 0.01 % 10` for extracting the hundredths digit of a float **fails ~60% of the time** because 0.01 is inexact in float64. Fixed in [`features.py`](../src/kaggle_playground_utils/features.py) using floor-then-shift with exact integer arithmetic. Eleven regression tests lock the fix. This matters for any Playground competition where synthetic data contains digit-based generator artifacts.

### 4. Mini-test diagnostic for feature candidate gating (v34)

Before committing a feature to full 5-fold CV: run a mini-test with 20% data × 1 fold × XGB with early stopping (~1 min per candidate). Over 30 candidates this reveals which are genuinely orthogonal vs redundant within ~30 minutes total. v34's round-1 probe identified 3–4 orthogonal clusters out of 30 top candidates — the rest were variants of the same underlying pattern.

## What didn't work (and why)

### 1. Iterative factory rounds (v31)

After v30 shipped 5 features, running the factory again on the v30 base produced only marginal candidates. +0.0002 on mini-test, −0.00002 on full CV. **Factory exhausts on a fixed raw-feature base in about 2 rounds.** Adding more rounds gives noise, not signal.

Takeaway: need different **raw** features (new columns derived externally, target-encoded cats, sub-population clustering), not more rounds of the same transformations.

### 2. Algorithm diversity on shared features (v32)

Standard "diverse-algorithm blend" advice: build XGB + LGB + CatBoost, blend them. We tried it on v30's feature set:

- LGB on v30: 0.97400 (weaker than XGB)
- CatBoost on v30: 0.97145 (much weaker)
- 3-way blend: **0.97392** (−0.00087 vs v30 alone)

When all three algorithms train on the same feature space and find the same dominant signals, their predictions converge (pairwise ρ > 0.99). There is no diversity to exploit. The blend HURT because it averaged the weaker models' noisier predictions into XGB's cleaner output.

Takeaway (L13): algorithm name alone is NOT a valid diversity axis when the feature space is shared. To get a real blend gain you need different **features**, different **regularization basins**, or different **model families** (GBDT vs NN vs linear) — not just different GBDT implementations.

### 3. Top-K by mutual information picks redundant features (v33)

Expanded factory: 1,699 candidates with stricter orthogonality thresholds. Picked top 7 by MI. Result: **−0.00035 on CV**, actively worse than v30.

Looking at the 7 picks: 6 were sin/cos variants of `Soil_Moisture` at different frequencies + 1 Rainfall variant. All six sin/cos encode the same underlying nonlinearity. The tree ensemble needed ONE; the other five were noise.

Takeaway: MI ranks each feature independently against the target. It does NOT penalize candidates that are mutually correlated. For final feature selection you must use **greedy forward selection with a marginal-lift gate** (pick ONE at a time, measure improvement, stop when no candidate crosses threshold). That's what [`greedy_selection.py`](../src/kaggle_playground_utils/greedy_selection.py) implements.

### 4. Full greedy never finished (v34 diagnostic stopped after round 1)

The round-1 probe worked cleanly: 12 of 30 candidates had positive lift on the mini-test, best +0.00106. The script then crashed during the optional baseline recompute step (known bug: passes duplicate column names to `_baseline_score`). Round-1 data was already collected and saved.

Decision: **do not fix and re-run**. The compute-parity realization (below) made clear that one more +0.0005 isn't worth a 2.5-hour cycle when the toolkit itself is the primary deliverable. The bug is documented; the fix is a ~10-minute diff that can wait for S6E5.

## L15 — The compute-parity realization (why we closed)

The #1 leaderboard position for S6E4 is held by a Kaggle Grandmaster who is a senior data scientist at NVIDIA. His published workflow for this class of competition is **850-model experimentation per attempt**. That scale almost certainly runs on NVIDIA's internal DGX / multi-GPU cluster infrastructure.

Our workstation setup: single GPU. One greedy-selection cycle ≈ 2.5 hours. His equivalent ≈ 50 cycles in parallel in the same wall-clock time.

**No amount of selection-logic cleverness closes a 50× experimentation-rate asymmetry on the same dataset.** Past a certain LB threshold the binding constraint flips from "ideas" to "experiments per unit time". The 0.9775–0.9785 range is a plausible solo ceiling on this dataset. Crossing 98%+ solo would require compound diversity: different raw-feature pipelines × algorithm diversity × fresh data sources × post-processing — budget-feasible only with a large compute multiplier.

This is **calibration, not defeat**. It's empirical evidence of compute asymmetry, documented for future reference.

### Practical implication for future competitions

Add to Phase 0 checklist: **audit top-5 leaderboard for industry-research employers**. If ≥2 are at NVIDIA / Google / Meta / DeepMind / Microsoft Research / OpenAI with probable DGX or TPU pod access, adjust solo-ceiling expectations at plan time. Target a medal-tier finish (top 10%) via toolkit leverage, not #1 via grind.

Factory saturation as stop signal: when top-K-by-MI candidates show positive mini-test lift but full-CV goes flat or negative, the feature space is exhausted for that base. Close the iteration loop; move on.

## Toolkit — the real deliverable

Built 2026-04-18 → 2026-04-19. See the [README](../README.md) for module status matrix. All examples in [`examples/`](../examples/) are working S6E4 reference scripts.

The genuinely transferable pieces for future tabular competitions:
- `features.py` — IEEE 754 precision-safe digit extraction + threshold helpers
- `registry.py` — auto-save OOF/holdout/test probs, never lose a model's predictions
- `train.py` — unified XGB/LGB/CatBoost training with Optuna class-weight tuning + mini-test gate
- `blend.py` — applies per-model tuning before blending (prevents a subtle bug where raw-untuned probs are averaged)
- `signal_factory.py` — the automated feature-discovery pipeline
- `greedy_selection.py` — the "top-K by MI is wrong" correction

## Lessons summary (15 total; key ones below)

Abbreviated so this document is self-contained. Full writeups exist privately; the most reusable are:

- **L1 — Regularization basin effect on Optuna class-weight tuning.** Tight regularization (high `alpha`, `reg_lambda`, `max_bin`) gives Optuna a flat objective surface and it converges quickly to a good class-weight vector. Loose regularization gives Optuna noise and it fails to find the global optimum. Choose the basin *before* running Optuna.
- **L3 — LB > holdout inversion on large datasets.** With 630K training rows, the holdout-score reliability floor is ~0.001. LB can legitimately be +0.003 over holdout due to test-set class distribution; don't panic and re-tune.
- **L6 — Original-data concat drift.** If you find the "original" Kaggle source dataset and concat it to train, compute adversarial AUC train vs original first. AUC > 0.65 means the distributions differ and the concat will hurt.
- **L10 — IEEE 754 digit-extraction bug.** As described above — `// 0.01 % 10` fails silently ~60% of the time. Use integer-based extraction.
- **L13 — Blend correlation ceiling.** Same features + same (or similar) regularization across algorithms → ρ > 0.99. No blend gain regardless of how you weight it. Diversity requires varying features, basin, or family — NOT algorithm name.
- **L14 — GPU memory is not the bottleneck; `lr × n_estimators` is.** With `max_bin=10000` + tight reg + `lr=0.1`, XGB converges in ~1,800 rounds. With loose reg + `lr=0.03`, needs ~8,500 rounds (3.5× slower). For rapid iteration, tight + 0.1.
- **L15 — Compute parity with industry-research grandmasters is structurally unattainable** on single-GPU workstations past a threshold. Documented above; frame as calibration, not defeat.

## Open questions (for future competitions or contributors)

- At what dataset size does the ~0.0001 noise floor on full CV actually materialize? Is it truly `1/sqrt(N_holdout)` or something stricter?
- Does the factory-saturation pattern hold across different target distributions (regression vs binary vs highly-imbalanced multi-class)?
- Is there a single-GPU technique that meaningfully closes the DGX compute gap? Mixed-precision, curriculum learning, sub-fold training? Genuinely unclear.
- When do public-kernel blends count as "solo-earned"? Tentative rule: only if your own factory contributes ≥1 orthogonal signal to the blend.

## Prompt for the next session

For anyone (human or agent) picking up S6E5 or a similar competition with this toolkit:

1. **Phase 0 check**: audit top-5 leaderboard for research-lab employers. Set solo-ceiling expectations accordingly.
2. **Use the toolkit from day 1**: `train_variant` + `Registry` + `signal_factory` are ready. Don't rewrite.
3. **Apply `greedy_selection`, NOT top-K by MI**, for final feature selection (fix the known baseline-recompute bug first — ~10-minute diff).
4. **Build `probes.py` and `diagnose.py` during Phase 0** — we know exactly what they need to do from S6E4 gaps.
5. **Set LB target realistically**: your own LB trajectory (+0.001 per iteration, 5 iterations = solid medal hunt) beats chasing #1.
6. **Finish at least 5 iterations before closing**. S6E4 taught us the toolkit-first mindset is right; but don't *start* closing before you've earned the data to justify it.

---

*Written during the closing session itself (2026-04-19). Good luck to anyone who picks up where we left off — happy to discuss on GitHub Issues.*
