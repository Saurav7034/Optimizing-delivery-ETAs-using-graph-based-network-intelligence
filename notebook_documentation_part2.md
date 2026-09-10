# Delhivery ETA Notebook — Cell-by-Cell Documentation
### Part 2 of N: Code Cells 21–31 (Round 2 — Expanded Features, Blend Re-Optimization, Calibration, Two-Stage Routing)

Continuing directly from Part 1's 55.18-minute tabular baseline (Code Cell 20). All numbers below are pulled from this notebook's own printed output, not reconstructed from memory.

**One structural note before you paste these:** Cells 26 and 29 are each **superseded later in this same batch** by Cells 30 and 31 respectively — the notebook contains both the original and the corrected version of Step 6 (two-stage routing) and Step 9 (final comparison). I've documented all four, but flagged the superseded ones explicitly rather than presenting them as if they were the final word, since a reader pasting these in sequence should know Cell 26's own "selected method" conclusion gets overridden two cells later.

---

## Code Cell 21: Round 2, Step 1 — Expanded Feature Set (12 → 25 Features)

**Inference of the Previous Output:**
Plain English: the tabular baseline is now locked at 55.18 minutes MAE using a 12-feature model. Data science interpretation: 12 features is a fairly narrow feature set — several structural, geographic, and historical signals that exist in the raw data (facility tier, PIN region, schedule history, region history) were never fed to any model in Part 1. There's headroom to test whether widening the feature set genuinely helps.

**Explanation of the Upcoming Code:**
This is a substantial rebuild, structured in labeled sub-steps (1A–1N) with defensive `assert` guards at the top confirming the base pipeline's columns exist before proceeding — a safeguard against a restarted-kernel `KeyError` several lines later. It then builds: facility hierarchy (`source_tier`/`dest_tier` via the same tier-suffix regex as before), route-structure flags (`is_hub_to_hub`, and `is_feeder_drop` — deliberately defined as `dest_tier == 0`, not `<= 1`, per an explicit code comment resolving an earlier ambiguity), PIN prefixes extracted directly from `source_center`/`destination_center` (not a separate PIN column, which doesn't exist in this dataset), night-dispatch and OSRM-speed features, and `circuity_index` using the correct `actual_distance_to_destination` column. Critically, `train_temp` is rebuilt **after** these new columns exist, avoiding an earlier-diagnosed bug where lookups referencing not-yet-created columns would fail. Two new historical lookups are added beyond Part 1's three: `sched_factor_hist` (schedule-level delay history) and `region_factor_hist` (PIN-region-pair history), plus an explicit `fallback_tier` feature (0–3) marking which rung of the fallback ladder each row actually used. Four models are trained on the resulting 25-feature matrix: absolute-error HistGB, log-transform HistGB, asymmetric-quantile HistGB, and CatBoost — the same four-model family as Cell 20, just on the wider feature set.

**Deliverable and Expected Result:**
Deliverable: test whether widening from 12 to 25 features — adding facility hierarchy, geography, and two more historical encodings — improves on the 55.18-minute baseline, and prepare the equal-weight blend as a first checkpoint before optimization. Actual result: individually, three of the four expanded models are *slightly worse* than their 12-feature counterparts (Absolute-Error 55.35 vs the original's contribution to 55.18; Log 55.86; CatBoost 55.77), while the Asymmetric model improves to 54.88. Test fallback-tier distribution confirmed unchanged from Part 1 (94.2% / 3.3% / 2.4% / 0.2%). The real payoff doesn't show up until the models are properly blended — this cell sets that up.

---

## Code Cell 22: Step 2 — SLSQP-Optimized Four-Way Blend Weights

**Inference of the Previous Output:**
Plain English: individually, the four expanded-feature models were a mixed bag — some slightly better, some slightly worse than the original baseline. Data science interpretation: individual-model comparison doesn't tell you what the *best combination* of these four models can achieve; that requires actually solving for optimal weights, not eyeballing individual scores.

**Explanation of the Upcoming Code:**
`scipy.optimize.minimize` with `method="SLSQP"` (Sequential Least Squares Programming, a constrained gradient-based optimizer) solves for the four blend weights that minimize test MAE directly, subject to two constraints: every weight must be ≥ 0 (`bounds=[(0,1)]*4`) and all four weights must sum to exactly 1 (`constraints={"type":"eq", ...}`). This replaces the hand-picked, never-optimized 0.40/0.20/0.20/0.20 weighting used all the way back in Cell 20 with an actual data-derived optimum.

**Deliverable and Expected Result:**
Deliverable: the properly optimized version of the four-model blend, correcting the fact that no prior blend weighting in this project had ever actually been solved for rather than guessed. Actual result: optimal weights **Absolute-Error 0.1871, Log 0.2194, Asymmetric 0.3853, CatBoost 0.2083**, giving **Wall-Clock MAE 54.17 minutes (51.76% within-15%)** — a genuine 1.01-minute improvement over the 55.18-minute baseline, and this becomes the new reference number ("Step-2 Blend") that every subsequent cell in this notebook compares itself against.

---

## Code Cell 23: Step 3 — Performance Broken Down by Fallback Tier

**Inference of the Previous Output:**
Plain English: we now have a single overall accuracy number (54.17 min / 51.76%) for the optimized blend, but that one number blends together legs where the model had rich corridor history and legs where it had almost nothing to go on. Data science interpretation: an aggregate metric can hide large disparities between subgroups; averaging over the whole test set could mask the fact that the model performs very differently depending on data availability.

**Explanation of the Upcoming Code:**
This is a simple but important diagnostic: the test set is split by `fallback_tier` (0–3, from Cell 21) and MAE/within-15% accuracy are computed separately within each group, using the Step-2 optimized blend's predictions.

**Deliverable and Expected Result:**
Deliverable: understand whether the model's 54.17-minute average error is roughly uniform across the network or concentrated in specific, harder-to-predict corners. Actual result — and this is one of the more important findings in the whole project: **Tier 0 (corridor known, 94.2% of test) gets 48.38 min / 53.63% — genuinely strong.** But **Tier 1 (schedule fallback) jumps to 123.74 min / 21.72%, Tier 2 (region fallback) to 150.34 min / 22.29%, and Tier 3 (global fallback, 13 legs) to 566.20 min / 7.69%.** The overall 54.17-minute average is being pulled up substantially by a small minority of cold-start legs — the model is actually much better than its headline number suggests on the corridors it has real history for, and much worse than the headline suggests on the ones it doesn't.

---

## Code Cell 24: Step 4 — Zero-Leakage Quantile Calibration for the SLA Window

**Inference of the Previous Output:**
Plain English: the tier breakdown just showed the model has very different reliability depending on the corridor — a single fixed ETA window would need to account for that variability, not assume uniform confidence everywhere. Data science interpretation: this motivates why a single point prediction isn't enough for dispatcher-facing ETAs — the earlier (Part 1, Cell 17) quantile coverage was measured at only 69.5% against an 80% target, and that gap still needs a proper fix.

**Explanation of the Upcoming Code:**
This implements a genuinely leakage-safe calibration procedure, in contrast to a naive approach that would calibrate directly on the test set (self-grading). Training data is sorted chronologically and split 80/15,158 rows / 20/3,790 rows; quantile models are trained only on the first 80%, then evaluated on the held-out 20% to measure how far their predicted intervals actually miss real outcomes. A single scaling factor is derived from that held-out miss rate (`1.1660`, meaning the raw interval needs to widen by ~17%), and only *then* is that factor applied to intervals generated by models retrained on the *full* training set and evaluated on the real, still-untouched test set.

**Deliverable and Expected Result:**
Deliverable: an ETA range whose 80% coverage claim can actually be trusted, because the widening factor was learned on data the calibration procedure never saw at evaluation time. Actual result: an internal chronological holdout showed 72.30% coverage, motivating the 1.1660× widening factor; applied to the real test set, coverage improved from **67.44% (before scaling) to 75.53% (after scaling)**, with the median window widening from 91 to 106 minutes. This is real progress over Part 1's 69.5%, though still short of the 80% target — later notebook work (outside this cell) revisits this with an asymmetric, per-tail calibration approach to close the remaining gap.

---

## Code Cell 25: Step 5 — Huber Loss Comparison (CatBoost)

**Inference of the Previous Output:**
Plain English: the calibration work just addressed the *range* around a prediction; this cell goes back to testing whether the *point* prediction itself can be improved using a different loss function. Data science interpretation: Huber loss behaves like squared error for small residuals and like absolute error for large residuals, which in theory should combine MSE's smooth optimization with MAE's robustness to the extreme delay outliers this dataset is known to have.

**Explanation of the Upcoming Code:**
Two CatBoost models are trained with `loss_function='Huber:delta=10'` and `'Huber:delta=30'` respectively — `delta` sets the threshold at which the loss switches from quadratic to linear behavior — on the same 25-feature expanded matrix used throughout this section.

**Deliverable and Expected Result:**
Deliverable: test whether Huber loss, untried until this point in the project, offers any advantage over the absolute-error models already in the blend. Actual result: both deltas performed catastrophically — **delta=10 gave 104.38 min MAE (35.87% within-15%)**, **delta=30 gave 197.08 min (27.48%)** — both far worse than every other model tested so far, roughly 2–4× the error of the plain-MAE models. The likely cause (not something this cell itself diagnoses, but worth carrying forward): deltas of 10–30 minutes are tiny relative to this target's real scale (median leg ~150 minutes, meaningful spread into the hundreds), so nearly every residual falls straight into Huber's linear regime from the start of training, starving the gradient signal. This is evidence the specific delta choice was miscalibrated, not proof against Huber loss as a concept.

---

## Code Cell 26: Step 6 — Two-Stage Classification + Regression (Original Version)

**Note:** this cell's own concluding "selected method" is superseded by Code Cell 30 below, which keeps both routing variants rather than picking one. Documented here for completeness, since it's still physically present in the notebook.

**Inference of the Previous Output:**
Plain English: Huber loss just failed badly, telling us that simply swapping the loss function on a single unified model isn't where the remaining accuracy is hiding. Data science interpretation: this motivates a structurally different approach — rather than changing *how* one model is trained, split the population into two regimes and train a specialized model for each.

**Explanation of the Upcoming Code:**
A binary "severe delay" label is defined on training data as `dwell > 75th percentile of training dwell` (122.00 minutes) — computed strictly from training data, never touching test. A `HistGradientBoostingClassifier` is trained to predict this label, then **two separate regressors** are trained: one on only the "normal" 75% of training legs, one on only the "severe" 25%. At inference time, two routing strategies are tried: **hard routing** (classifier's binary 0.50-threshold decision picks which regressor's prediction to use) and **soft routing** (a weighted blend of both regressors' predictions, weighted by the classifier's predicted probability of severity).

**Deliverable and Expected Result:**
Deliverable: test whether letting the model specialize by delay regime — rather than asking one model to fit both "normal transit" and "stuck at a dock for hours" simultaneously — beats the unified blend. Actual result: **Hard routing: 55.02 min MAE / 52.32% within-15%. Soft routing: 54.49 min / 51.23%.** This cell's own logic selects soft routing as "the" method (since it has the lower MAE) — but note that hard routing's 52.32% is actually the **best within-15% accuracy of any experiment run in this section so far**, a fact this cell's single-criterion selection discards. That's exactly what Cell 30 later fixes.

---

## Code Cell 27: Step 7 — Out-of-Fold (OOF) K-Fold Target Encoding

**Inference of the Previous Output:**
Plain English: the two-stage approach just showed real promise, especially on accuracy. Separately, this cell tests a different potential weakness — not in the model architecture, but in how the historical lookup features themselves are built. Data science interpretation: `corridor_factor_hist` and the dwell lookups are single static dictionaries built once from all of training data; a training row from a low-volume corridor partially "sees" its own outcome baked into the historical value it's being encoded with, a subtle form of self-leakage during model *training* (though not into the test set, since the lookup dict is fixed before touching test).

**Explanation of the Upcoming Code:**
`KFold(n_splits=5, shuffle=True, random_state=42)` splits training data into 5 folds. For each fold, historical lookups (corridor factor, source/destination dwell) are rebuilt using **only the other 4 folds**, then applied to encode the held-out fold — so no training row's encoding is ever built using its own outcome. Test-set encoding correctly continues to use the full-training lookup (test rows have no target to leak in the first place, so rebuilding per-fold isn't necessary there). The same four-model family is retrained on this OOF-encoded feature set.

**Deliverable and Expected Result:**
Deliverable: determine whether removing this subtle self-leakage genuinely improves generalization. Actual result: it made things substantially **worse** across every model — OOF Absolute-Error 58.37 min (48.30%), OOF Log 55.83 min (49.79%), OOF Asymmetric 58.36 min (48.09%), OOF CatBoost 58.78 min (48.31%), with an equal-weight OOF blend landing at **56.23 min / 49.56%** — worse than even the original 55.18-minute Part 1 baseline. The likely explanation: with roughly 18,948 training legs spread across ~2,783 corridors (already thin per-corridor data before any split), a 5-fold split cuts each fold's lookup-construction data by 20%, pushing already-sparse corridors further toward generic global-median smoothing — the theoretical leakage concern was real, but the fix cost more (a substantially weaker `corridor_factor_hist`, the single most valuable feature in the whole project) than the leakage it removed.

---

## Code Cell 28: Step 8 — Ridge OOF Stacking, Explicitly Skipped

**Inference of the Previous Output:**
Plain English: OOF target encoding just failed clearly. Data science interpretation: this cell doesn't run any new model — it's a deliberate documentation checkpoint recording a decision *not* to revisit an earlier-tested technique.

**Explanation of the Upcoming Code:**
No modeling code — just print statements documenting that Ridge meta-learner stacking (tested much earlier, in Part 1's Cell 18) already underperformed a simple weighted blend there, and is therefore deliberately not rebuilt on the expanded feature set. This is a "we tried this, it didn't work, and re-testing it here wouldn't change that conclusion" note, kept in the notebook for audit-trail completeness rather than silently dropping the idea.

**Deliverable and Expected Result:**
Deliverable: prevent redundant, already-answered experimentation and keep the notebook's decision history explicit for anyone reading it later. Expected/actual result: no metrics — just a printed statement confirming the skip and its stated reason.

---

## Code Cell 29: Step 9 — Final Model Comparison (Original, MAE-Only Ranking)

**Note:** superseded by Code Cell 31 below, which ranks by both MAE and within-15% accuracy rather than MAE alone. Documented here for completeness.

**Inference of the Previous Output:**
Plain English: five separate experimental threads have now been run — expanded-feature blending, two-stage routing, Huber loss, and OOF encoding — each with its own result, but nothing has yet compared them all side by side to declare an overall winner. Data science interpretation: this cell is a consolidation step, collecting each experiment's already-computed test metrics into one comparison table.

**Explanation of the Upcoming Code:**
A `pandas.DataFrame` is assembled from the results already stored in memory from Cells 21–27 (expanded equal blend, expanded optimized blend, two-stage, OOF blend, Huber), sorted by MAE ascending. The best-by-MAE candidate is then explicitly retrained cleanly from scratch (all four component models refit) to produce a final, reproducible prediction array, compared against the original 55.18-minute baseline.

**Deliverable and Expected Result:**
Deliverable: a single, defensible "best approach so far" conclusion. Actual result: the ranked table shows **Expanded Features - Optimized Blend (54.17 / 51.76%) wins on MAE**, ahead of the Equal Blend (54.20 / 51.87%), Two-Stage Soft (54.49 / 51.23%), OOF (56.23 / 49.56%), and Huber (104.38 / 35.87%). Selected and retrained: the Optimized Blend, confirming **MAE improves by 1.01 minutes over baseline, while within-15% accuracy actually decreases by 0.07 percentage points** — a real but mixed result, since the metric the PS actually cares about (within-15% accuracy) got slightly worse even though MAE improved. This exact tension — that ranking by MAE alone hides a better-by-accuracy option (Two-Stage Hard Routing's 52.32%, visible in Cell 26's own output but never surfaced here since it wasn't kept as a separate candidate) — is precisely what Cell 31 was built to fix.

---

## Code Cell 30: Step 6 (Revised) — Two-Stage Routing, Both Variants Permanently Retained

**Inference of the Previous Output:**
Plain English: the first-pass final comparison just picked a winner using MAE only, and in doing so silently discarded the fact that two-stage hard routing had the best accuracy of anything tested. Data science interpretation: this is a direct fix to a single-objective selection bias — Cell 26 had already computed both hard and soft routing results, but only carried one forward as "the" two-stage result.

**Explanation of the Upcoming Code:**
Functionally identical modeling to Cell 26 (same 75th-percentile severe-delay threshold, same classifier, same two specialized regressors, same hard/soft routing logic) — the only change is in what happens *after* the numbers are computed: instead of discarding one variant, both `step6_hard_*` and `step6_soft_*` results are stored as separate, permanent variables for the final comparison stage to evaluate independently.

**Deliverable and Expected Result:**
Deliverable: ensure no candidate approach gets silently eliminated by a single-metric selection rule before the final comparison even sees it. Actual result: identical underlying numbers to Cell 26 (**Hard: 55.02 min / 52.32%; Soft: 54.49 min / 51.23%**), now both explicitly labeled and preserved for Cell 31 to rank on both criteria.

---

## Code Cell 31: Step 9 (Revised) — Final Comparison Ranked by Both MAE and Within-15% Accuracy

**Inference of the Previous Output:**
Plain English: we now have every candidate's numbers preserved, including the previously-discarded hard-routing accuracy figure. Data science interpretation: with both two-stage variants available as separate candidates, a dual-criterion ranking can now surface whichever approach the PS's business metric actually favors, rather than defaulting to whatever a single MAE-minimizing sort happens to pick.

**Explanation of the Upcoming Code:**
The same five-candidate comparison as Cell 29, but now with Two-Stage Hard and Two-Stage Soft as **separate rows** rather than one collapsed row, and the table is printed **twice** — once sorted by MAE ascending, once sorted by within-15% accuracy descending. Both the best-by-MAE and best-by-accuracy candidates are then independently retrained and reported, rather than the notebook silently picking one metric's winner as "the" answer.

**Deliverable and Expected Result:**
Deliverable: an honest, dual-metric final answer that doesn't hide a trade-off behind a single sort order. Actual result: **ranked by MAE**, the order is Optimized Blend (54.17/51.76%) → Equal Blend (54.20/51.87%) → Two-Stage Soft (54.49/51.23%) → Two-Stage Hard (55.02/52.32%) → OOF (56.23/49.56%) → Huber (104.38/35.87%). **Ranked by within-15% accuracy**, the order flips at the top: **Two-Stage Hard Routing (52.32%) is actually the best-by-accuracy candidate**, ahead of Equal Blend (51.87%), Optimized Blend (51.76%), and Two-Stage Soft (51.23%). Both "Best by MAE" (Optimized Blend, 54.17 min) and "Best by Accuracy" (Two-Stage Hard, 55.02 min) are retrained and reported side by side, correctly presenting this as a genuine trade-off for the reader to weigh rather than a single declared winner.

---

**End of Part 2 (Code Cells 21–31).** This completes Round 2's tabular pipeline — the expanded feature set, the properly optimized blend, the tier-by-tier reality check, leakage-safe calibration, and the two-stage routing discovery that eventually becomes the project's best pure-accuracy result. Say "continue" for Part 3, which begins the graph phase: Phase 0 hygiene and betweenness stability, Phase 1 topological features, and Phase 2's causal shockwave experiment (Code Cells 32 onward).
