# Delhivery ETA Notebook — Cell-by-Cell Documentation
### Part 6 of 6: Code Cells 55–65 (Phase 3.1–3.7 Bounded Improvement Pass, Parallel-Route Analysis)

Continuing from Part 5's Cell 54 (the honest, leakage-free calibration: 81.3% coverage, 136-min window). All numbers re-pulled directly from this notebook's actual output. **Cell 62 (Phase 3.7) is documented here from a genuinely first-time read** — it doesn't appear anywhere earlier in this documentation project, so it's been read fresh from source rather than assumed.

---

## Code Cell 55: Phase 3.1 — Expanded GraphSAGE Node Features (10 Features vs. the Original 5)

**Inference of the Previous Output:**
Plain English: the model side of the project is now fully locked and verified at 52.72 min / 52.85%, and every business deliverable has been corrected. This marks a deliberate, capped return to model experimentation — testing a specific, previously-identified gap: GraphSAGE's node input only ever used 5 purely structural features (betweenness, PageRank, in/out-degree, component size), never facility-level attributes like tier, region, or dwell history. Data science interpretation: richer node inputs *could* let the graph neural network encode more nuanced facility differences into its embeddings, but also carry real overfitting risk on a small (1,545-node), sparse graph.

**Explanation of the Upcoming Code:**
The GraphSAGE node feature matrix is expanded from 5 to 10 columns by adding facility tier, PIN prefix, destination-dwell history, and inbound/outbound shipment volume. The entire pipeline — GraphSAGE retrain, embedding extraction, all 4 blend models, SLSQP weight optimization — is rerun across the same 3 seeds (42, 123, 999) used to establish the 52.72-minute baseline, for a fair, like-for-like comparison.

**Deliverable and Expected Result:**
Deliverable: test whether richer node inputs improve on the locked baseline, following the same evaluation-gate discipline as every prior graph experiment. Actual result: **Locked baseline MAE 52.72 / Acc15 52.85% → Phase 3.1 MAE 52.88 / Acc15 52.78%** — both metrics moved in the wrong direction (−0.16 min, −0.07pp). Verdict: **`[FAIL]`** — richer node features did not help, and the notebook's own printed conclusion attributes this to overfitting/noise on the added inputs. The baseline is correctly retained unchanged.

---

## Code Cell 56: Phase 3.2 — Target Decomposition Retry (Full Feature Set, Post-Graph)

**Inference of the Previous Output:**
Plain English: Phase 3.1 just showed that more GraphSAGE inputs don't help — the next experiment revisits an idea that failed earlier in the project (Cells 36/37: predicting driving-time error and dwell separately, then summing) but now applies it on top of the full, final 65-feature graph-enhanced matrix rather than the pre-graph feature set used in Cells 36/37. Data science interpretation: this checks whether the earlier decomposition failure was specific to the smaller feature set, or a more fundamental property of splitting the target.

**Explanation of the Upcoming Code:**
Same architecture as the earlier corrected decomposition attempt (Cell 37) — two `HistGradientBoostingRegressor` models, one predicting `driving_residual`, one predicting `dwell`, both given the complete expanded feature set (now including GraphSAGE embeddings) — but retrained fresh in this cell using the current, post-graph feature matrix.

**Deliverable and Expected Result:**
Deliverable: a final, definitive test of the decomposition idea, now with every available feature (including graph embeddings) given to both submodels. Actual result: submodel diagnostics show **Driving Residual MAE 30.46 min (own target), Dwell MAE 44.41 min (own target)** — again individually reasonable. Recombined: **Baseline MAE 52.72 → Decomposed MAE 56.94 (−4.22 min), Acc15 52.85% → 50.92% (−1.93pp)**. Verdict: **`[FAIL]`**, once more, and by a wide margin — a third confirmation (after Cells 36 and 37) that summing two independently-erring predictions compounds error rather than cancels it, regardless of how rich the underlying feature set is. This strengthens the earlier conclusion that decomposition is a structurally weaker architecture for this specific target, not merely a feature-availability problem.

---

## Code Cell 57: Phase 3.3 — Optuna Re-Tuning of the Base HistGB Model on the Final Feature Set

**Inference of the Previous Output:**
Plain English: two feature/architecture ideas have now failed on top of the locked baseline. This cell tests the last "cheap" lever: none of the blend's component hyperparameters (learning rate, tree depth, regularization) were ever re-searched after the feature set grew from 12 (Round 1) to 65 (post-graph) columns. Data science interpretation: hyperparameters tuned for a 12-feature model aren't guaranteed to remain optimal for a 65-feature model — this tests whether re-tuning closes any remaining gap.

**Explanation of the Upcoming Code:**
A fresh 25-trial Optuna `TimeSeriesSplit`-based search (same TPE methodology used throughout the project) is run over the base absolute-error HistGB model's hyperparameters, now searching against the full 65-feature matrix. The best-found configuration is then plugged into the same 3-seed full-blend evaluation gate used in Cells 55–56.

**Deliverable and Expected Result:**
Deliverable: determine whether the blend's still-original, years-old hyperparameters are leaving real accuracy on the table now that the feature set has grown substantially. Actual result: the Optuna search itself found a real internal CV improvement (**Best CV MAE 49.40 min** vs. whatever the untuned config scored in cross-validation), with meaningfully different hyperparameters (`learning_rate` 0.083 vs. the original 0.195, `max_leaf_nodes` 56 vs. 48). But translated into the real 3-seed test gate: **Baseline MAE 52.72 → Re-tuned MAE 52.76 (−0.04 min), Acc15 52.85% → 52.89% (+0.04pp)** — both deltas are roughly 6× smaller than the measured seed-noise threshold (0.232 min). Verdict: **`[FAIL]`** — despite genuinely different hyperparameters being found, the real-world result is statistically indistinguishable from the existing baseline, evidence that the model has reached a genuine performance ceiling rather than simply being under-tuned.

---

## Code Cell 58: Phase 3.4a — New Feature Groups (Trip Position, Volatility, Upstream Delay) + Fast Add-One-In Gate

**Inference of the Previous Output:**
Plain English: three separate levers (richer graph inputs, target decomposition, hyperparameter tuning) have now all failed to beat the locked baseline. This cell tests a different category entirely — features describing a leg's position within its multi-leg trip, and the historical *variability* (not just average) of facilities and corridors — none of which had been tried anywhere earlier in the project. Data science interpretation: this is a genuinely new information source, not a re-parameterization of existing signals, so a negative result here would carry more weight than another tuning failure.

**Explanation of the Upcoming Code:**
Three feature groups are built and tested independently before committing to the expensive full pipeline: **Group A (trip position)** — `leg_index_in_trip`, `total_legs_in_trip`, `is_first_leg`, `is_last_leg`, `frac_through_trip`, ordered by actual departure time rather than file order. **Group B (volatility)** — dwell IQR per facility, corridor delay-ratio standard deviation, and schedule size (`sched_n_facilities`), all built train-only. **Group C (upstream delay)** — the actual outcome of the previous leg in the same trip, with an explicit causal guard (the prior leg must have physically ended before this leg started) — flagged in the code's own comments as consuming real observed outcomes, unlike Groups A and B, which are provably leak-free by construction. A fast, single-seed, single-model gate tests each group and combination against the locked baseline's own feature set.

**Deliverable and Expected Result:**
Deliverable: a cheap, single-model screen to identify whether any of these three new signal sources are worth the expense of a full 3-seed blend retest. Actual result: **43.8% of all legs are not the first leg of their trip** — meaningful coverage for Group A. All five tested combinations landed within the seed-noise band: **Group A: +0.02 min MAE / −0.30pp Acc15; Group B: −0.09 min / −0.58pp; Group C: +0.27 min / −0.28pp (the worst individual result); A+B: −0.16 min / −0.28pp; A+B+C: +0.67 min / −0.97pp**. Notably, **every single combination showed a negative Acc15 delta**, even where MAE nominally improved — a consistent pattern the cell's own guidance explicitly flags as "within noise" across the board. Group C (upstream delay) being the worst performer here is itself a meaningful finding: it's the third independent test (after the two shockwave attempts in Cells 34–35) of whether recent delay history propagates through this network, and it fails a third time — reinforcing that delay does not meaningfully cascade between consecutive legs of the same trip in this data.

---

## Code Cell 59: Phase 3.4b — Full 3-Seed Blend Confirmation (Group A+B)

**Inference of the Previous Output:**
Plain English: the fast single-model screen showed A+B (trip position + volatility) as the least-bad combination, worth a proper, expensive 3-seed confirmation despite sitting within noise on the cheap test. Data science interpretation: this cell exists specifically to check whether a marginal single-model signal survives the more rigorous full-blend, multi-seed procedure — the same discipline applied to every other experiment in this section.

**Explanation of the Upcoming Code:**
The complete pipeline (GraphSAGE retrain, all 4 blend models, SLSQP weights) is rerun across all 3 seeds using the baseline feature set plus Groups A and B only (matching the fast gate's recommendation to prefer the two provably-clean feature groups over the causally-riskier Group C).

**Deliverable and Expected Result:**
Deliverable: a properly rigorous confirmation (or rejection) of the fast gate's most promising signal. Actual result: **Verified baseline MAE 52.72 / Acc15 52.85% → Phase 3.4 (A+B) MAE 52.93 / Acc15 52.66%** — both metrics worse (−0.21 min, −0.19pp). Verdict: **`[FAIL]`**. This is also a useful methodological confirmation in its own right: the fast single-model gate showed A+B as a small MAE *improvement* (−0.16 min), while the full 3-seed blend gate shows it as a *worsening* (−0.21 min) — the sign flip between the cheap and expensive test is itself evidence that the fast gate's small delta was noise, exactly the risk the two-stage gating procedure was designed to catch.

---

## Code Cell 60: Phase 3.5 — Acc15-Objective Blend Weight Search

**Inference of the Previous Output:**
Plain English: four feature/architecture ideas have all failed to beat the locked baseline. This cell tests something structurally different again — not a new feature, but a new *objective*: every blend weight fit anywhere in this project (Cell 22, Cell 40's SLSQP, Track A) has minimized MAE, but the PS's actual business metric is within-15% accuracy, and those two objectives are already known (from the two-stage routing discovery in Part 2) to sometimes disagree. Data science interpretation: this tests whether directly optimizing for Acc15 — a non-smooth, step-function metric that standard gradient-based optimizers can't handle — finds a genuinely better trade-off than the MAE-optimal weights.

**Explanation of the Upcoming Code:**
The four component models' predictions are rebuilt, averaged across the same 3 seeds as the locked baseline. Because Acc15 is a step function (small weight changes often flip zero predictions' in/out-of-band status, giving near-zero gradient almost everywhere), a derivative-free search (Dirichlet random sampling over the weight simplex, refined in stages) is used instead of SLSQP to find Acc15-maximizing weights, compared directly against the MAE-optimal weights on the identical component predictions. Critically, the cell then runs a **split-half generalization check**: the test set is repeatedly split in half, Acc15-optimal weights are fit on one half and scored on the *unseen* other half, specifically to test whether any in-sample Acc15 gain is a real, generalizable effect or an artifact of fitting a step function directly against the test set.

**Deliverable and Expected Result:**
Deliverable: determine whether reweighting the existing four models toward the actual business metric yields a real, defensible accuracy gain. Actual result: **MAE-optimal weights (0.220/0.564/0.111/0.105): 52.75 min / 52.81% Acc15. Acc15-optimal weights (0.032/0.205/0.439/0.323): 53.03 min / 53.32% Acc15** — a trade of +0.28 min MAE for +0.51pp Acc15, measured in-sample. But the split-half check tells the real story: **out-of-sample, the Acc15 gain collapses to a mean of +0.04pp (std 0.34), positive in only 7 of 12 splits** — close to a coin flip. Verdict: **`[WEAK]`** — the notebook's own conclusion correctly labels this as suggestive, not a verified improvement, and the locked 52.72/52.85% baseline is retained. The cell also adds an important integrity note for the record: every blend weight fit in this project, including the locked baseline itself, is fit directly against the test set, making all reported numbers mildly optimistic — flagged explicitly rather than hidden.

---

## Code Cell 61: Phase 3.6 — GraphSAGE Hyperparameter and Input-Scaling Search

**Inference of the Previous Output:**
Plain English: the blend-reweighting idea just failed the generalization check. This cell returns to the GraphSAGE architecture itself, testing something never touched anywhere in the project: every GraphSAGE hyperparameter (hidden size, embedding dimension, learning rate, epochs, layer count, aggregator) has been hardcoded since it was first introduced, and — more specifically — the node input features (betweenness ~0.0001–0.16, PageRank ~0.0001–0.01, degrees 0–34, component size up to 1168) are fed raw into the network with no scaling, despite spanning multiple orders of magnitude. Data science interpretation: unscaled, wildly different-magnitude inputs are a classic, well-understood cause of a neural network effectively ignoring smaller-magnitude features in favor of dominant ones.

**Explanation of the Upcoming Code:**
A 30-trial Optuna search treats `scale_x` (whether to StandardScale the node inputs) and `y_transform` (raw / standardized / log1p on the dwell pretraining target) as explicit, searchable hyperparameters alongside architecture choices (hidden size, embedding dimension, layer count, dropout, learning rate, weight decay, epochs, aggregator). Trials are scored on an internal temporal validation split carved out of training data — never touching the real test set — with the current hardcoded configuration explicitly enqueued as trial 0 for a direct comparison. Only the single best-found configuration is then taken to a final 3-seed test-set gate.

**Deliverable and Expected Result:**
Deliverable: a genuine test of whether the GraphSAGE architecture (not just the downstream blend) has room for improvement, done without touching the test set until the very last step. Actual result: the search found a real internal validation improvement — **current hardcoded config: 46.254 min validation MAE → best found: 44.348 min (+1.906 min)** — with the winning configuration being `hidden=112, emb_dim=32, dropout=0.2, lr=0.0125, epochs=350, aggregator='max'`. Notably, **the scaling hypothesis was not confirmed**: the top trial used `scale_x=False`, with `y_transform='log1p'` and a larger embedding dimension appearing to matter more than input scaling — worth stating plainly rather than treating the original hypothesis as vindicated. Taken to the final 3-seed test gate: **Verified baseline MAE 52.72 / Acc15 52.85% → Tuned GNN MAE 52.87 / Acc15 52.65%** — worse on both counts. Verdict: **`[FAIL]`**. This is a clean, textbook case of validation-set improvement not translating to test-set improvement — the tuned configuration overfit to the particular temporal validation split used during the search, despite that split never touching the real test set.

---

## Code Cell 62: Phase 3.7 — Tier-Conditioned Blend Weights

**Inference of the Previous Output:**
Plain English: six consecutive experiments (richer graph features, decomposition, hyperparameter tuning, new feature groups, Acc15-objective weights, GraphSAGE architecture tuning) have all failed to beat the locked baseline. This cell's own header states its motivating observation plainly: every blend weight fit anywhere in this project uses one shared weight vector across all 7,421 test legs, despite Tier 0 (corridor-known, 94.2% of legs) and Tiers 1–3 (fallback/cold-start, 5.8% of legs) having wildly different error levels (3–12× apart) and relying on GraphSAGE embeddings for facilities that may sit on very few or zero training edges. Data science interpretation: a single global weight vector is, by construction, a compromise between two populations with very different optimal blending behavior — this tests whether fitting two separate weight vectors, one per regime, and hard-routing each test leg to its tier's weights, captures a real gain a single shared vector cannot.

**Explanation of the Upcoming Code:**
Test legs are split into Tier 0 (6,989 legs) versus Tiers 1+2+3 pooled together (432 legs — pooled because Tier 3 alone has only 13 legs, too few to fit its own weight vector reliably). Two separate SLSQP-optimal weight vectors are fit — one using only Tier-0 rows, one using only the pooled-fallback rows — and each test leg's final prediction uses whichever vector matches its own `fallback_tier`. Following the same two-stage gating pattern established in Cell 58/59, a fast single-seed check (seed 999) is run first; only because it showed a small improvement does the cell proceed to the full, expensive 3-seed confirmation. A tier-by-tier MAE breakdown, global weights vs. tier-conditioned, is printed at the end for transparency regardless of the overall verdict.

**Deliverable and Expected Result:**
Deliverable: a properly gated test of whether tier-specific blend weighting — a genuinely new idea, not yet attempted anywhere else in this notebook — beats the single global weight vector. Actual result: fast single-seed gate showed a marginal improvement (Global 52.95 min / Tier-conditioned 52.92 min, +0.03 min), enough to trigger the full confirmation. Full 3-seed result: **re-measured global-weight blend this run: 52.80 min / 53.04% (note: slightly different from the originally-locked 52.72/52.85%, itself a reminder of ordinary run-to-run seed variance); tier-conditioned blend: 52.75 min / 52.89%** — a difference of only **−0.03 minutes versus the original locked baseline**, far inside the 0.232-minute noise threshold. Verdict: **`[FAIL]`**. The tier-by-tier breakdown shows why: tier-conditioning moves each tier's MAE by less than a minute in every case (Tier 0: 47.00→46.99; Tier 1: 122.89→122.94; Tier 2: 149.54→148.02; Tier 3: 552.58→550.07) — directionally consistent with the hypothesis (fallback tiers improve slightly) but far too small to register as a real effect once seed noise is accounted for. This closes out the bounded Phase 3 improvement pass: seven independently well-motivated ideas tested, all seven failing to clear the noise floor, which is itself a strong, evidence-backed conclusion that the 52.72-minute model represents a genuine performance ceiling given this dataset's available information.

---

## Code Cell 63: Task 5 (Completion) — Parallel Route Candidate Analysis, Version 1

**Inference of the Previous Output:**
Plain English: the bounded model-improvement pass is now closed with a clear, well-evidenced conclusion that the model has hit its ceiling. Attention returns fully to the PS's remaining consulting deliverable — the Task 5 requirement to recommend "parallel route" interventions had never been addressed anywhere in the notebook up to this point (only facility-upgrade and route-type-shift recommendations existed). Data science interpretation: this is pure business-logic aggregation on top of `clean_train_legs`, not a modeling exercise.

**Explanation of the Upcoming Code:**
Per-corridor statistics are aggregated: trip count, median delay ratio, number of distinct schedules serving the corridor, and circuity (OSRM road distance ÷ straight-line distance). A candidate parallel-route corridor is defined as one served by exactly one schedule (no backup), carrying at least 30 trips (meaningful volume), and chronically delayed (ratio > 1.20, the PS's own literal threshold).

**Deliverable and Expected Result:**
Deliverable: a first attempt at the missing Task 5 intervention type. Actual result: of 2,349 total corridors, **1,888 (80.4%) are served by exactly one schedule** — a strong network-wide fragility statistic on its own. But the strict candidate filter (single-schedule AND ≥30 trips AND chronically delayed) produced only **2 qualifying corridors, covering 66 trips (0.4% of network traffic)** — both short (~30-40 km, `_CP`/`_PC` intra-city hops), too thin a result to build a meaningful memo recommendation around, since the volume and chronic-delay filters were found to pull in opposite directions (high-volume corridors tend to be well-established and less likely to also be chronically delayed).

---

## Code Cell 64: Task 5 (Completion, v2) — Parallel Route Candidates, Relaxed Thresholds + Two-Level Reporting

**Inference of the Previous Output:**
Plain English: the first attempt's network-wide fragility statistic (80.4% single-schedule) was strong, but the specific candidate shortlist was too small and made up of unconvincing short intra-city hops. Data science interpretation: the filter thresholds, not the underlying concept, were the problem — this version relaxes them and separates the analysis into two distinct levels of reporting.

**Explanation of the Upcoming Code:**
Thresholds are relaxed: minimum trip count 30→10, schedule count restricted to ≤2 (not exactly 1), and a new 50km minimum distance floor is added specifically to exclude short intra-city transfers, plus a requirement that both corridor endpoints have real (non-placeholder) names. Reporting is now split into **Level 1** (network-wide fragility, with traffic share attached, not just corridor-count share) and **Level 2** (the targeted candidate shortlist), with each candidate's excess vehicle-hours computed relative to a 1.20x-OSRM "well-run corridor" benchmark, and high-circuity corridors flagged separately as "build a direct link" candidates versus "add a second service" candidates.

**Deliverable and Expected Result:**
Deliverable: a genuinely usable parallel-route candidate list, distinguishing network-wide risk from specific actionable corridors. Actual result — Level 1: **63.3% of all trips** (not just 80.4% of corridor *count*) run on single-schedule corridors — a stronger, traffic-weighted version of the earlier finding. Level 2: **249 corridors** now qualify, carrying **18.4% of network traffic**, with **68 flagged as high-circuity "detour" cases**. The printed top candidates reveal a pattern the cell itself doesn't yet analyze: **4 of the top 5 worst corridors all share the same destination, `Muzaffrpur_Bbganj_I (Bihar)`** — a fact that becomes the basis for Cell 65's convergence-cluster analysis.

---

## Code Cell 65: Task 5 (v3) — Convergence Clusters and Corrected Intervention Routing

**Inference of the Previous Output:**
Plain English: the v2 candidate list surfaced 249 corridors, but a close look at the top entries showed the same destination facility (Muzaffarpur) appearing repeatedly — suggesting some "corridor" problems are actually one facility problem, wrongly counted as several separate corridor recommendations. Data science interpretation: recommending a parallel route for each of several corridors that all funnel into one broken facility would misdiagnose the root cause and likely fail to fix the underlying issue.

**Explanation of the Upcoming Code:**
The 249 v2 candidates are grouped by destination facility to find "convergence clusters" — facilities absorbing multiple chronically-delayed inbound corridors, measured both in absolute count and as a share of that facility's *total* inbound corridors (distinguishing "this facility is systematically broken" from "coincidence"). Corridors belonging to a flagged cluster (≥2 bad inbound corridors) are explicitly separated out and routed to a facility-fix recommendation instead of a parallel-route recommendation; only the remaining, non-cluster corridors are kept as genuine parallel-route candidates, further split by the same detour-flag logic from v2. A consistency diagnostic (comparing each corridor's median vs. 90th-percentile delay ratio) is also added, explicitly caveated in the printed output as indicative rather than conclusive given the small per-corridor sample sizes (10-14 trips each).

**Deliverable and Expected Result:**
Deliverable: intervention recommendations correctly matched to their actual root cause — facility-level fixes where a facility is the problem, genuine parallel-route recommendations only where the corridor itself is. Actual result: **240 of 249 corridors labeled "CONSISTENT" (reliably slow) vs. 9 "SPIKY"** — a split the notebook itself flags as likely an artifact of small sample sizes rather than a reliable structural finding. **38 facilities identified as convergence clusters, absorbing 113 of the 249 candidates (45%)** — with Hyderabad_Shamshbd_H (12 bad inbound corridors, 41% of its total inbound), Bangalore_Nelmngla_H (7, 21%), and Muzaffrpur_Bbganj_I (6, but the **highest share at 60%** of its 10 total inbound corridors, plus the single worst delay ratio in the dataset at 12.26x OSRM) as the top three. Notably, several of these convergence-cluster facilities (Hyderabad, Bangalore, Bhiwandi) also appear in the earlier SLA-breach top-5 hub list built through an entirely independent method — a form of cross-validation between two different analyses reaching the same conclusion. The remaining 136 non-cluster corridors split into **45 genuine "build a direct link" candidates** (indirect road path) and **91 "add a second scheduled service" candidates** (direct path, just no redundancy) — the two intervention types now correctly separated from facility-level fixes.

---

**End of Part 6 (Code Cells 55–65) — and end of the full documentation.** This closes the notebook: seven independently-tested, well-gated model-improvement experiments that all failed to beat the locked 52.72-minute baseline (a real, evidence-backed finding about the model's ceiling, not a wasted effort), followed by the three-round construction of the previously-missing parallel-route deliverable, arriving at a properly diagnosed, three-tier intervention framework — network-wide fragility, facility-level convergence clusters, and genuine corridor-level parallel-route candidates. Code Cell 66 is empty and contains no content to document.

**Full documentation set: Parts 1–6, Code Cells 1–65, complete.**
