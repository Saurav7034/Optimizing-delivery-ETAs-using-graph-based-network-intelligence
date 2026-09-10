# Delhivery ETA Notebook — Cell-by-Cell Documentation
### Part 5 of 6: Code Cells 48–54 (Corrected Tier Audit, Deliverable Bug Fixes, Honest Calibration)

Continuing from Part 4's Cell 47 (the empirical Task 4/5 pass, with its known denominator overstatement). All numbers re-pulled directly from this notebook's actual output.

**Two more superseded pairs in this batch**, same pattern as Part 2's Cells 26/30 and 29/31: **Cell 51 is superseded by Cell 52** (an identical cell rerun after fixing a hardcoded string bug), and **Cell 53 is superseded by Cell 54** (a leaky calibration check replaced by a genuinely leakage-free one). Both pairs are documented below with the fix and its effect made explicit.

---

## Code Cell 48: Phase 0 — Tier-by-Tier Audit Against the *True* Averaged Predictions

**Inference of the Previous Output:**
Plain English: Part 4 established the final, seed-verified model (52.72 min / 52.85%) — but every tier-by-tier breakdown computed *before* that point (Cells 23, 40, 43) was evaluated against an earlier, less-final version of the model. Data science interpretation: this is a re-audit, not a new experiment — it re-runs the same tier-segmentation logic used throughout the notebook, but for the first time against `true_averaged_preds`, the actual 3-seed-averaged predictions that represent the project's real final output.

**Explanation of the Upcoming Code:**
The same fallback-tier grouping used since Cell 23 is applied one more time, now paired with `true_averaged_preds` instead of any single-seed or intermediate blend. Three historical reference points are printed alongside the new numbers for direct comparison: the original single-model result, the Cell-43 seed-999-only blend result, and the new true-averaged result — making the progression across the project's three major model generations visible in one table.

**Deliverable and Expected Result:**
Deliverable: the final, authoritative tier-by-tier performance table, and an honest check on whether averaging across seeds specifically helped or hurt the hardest-to-predict cold-start tiers. Actual result: **overall MAE 52.72 min / 52.85% Acc15**, matching Cell 44 exactly, confirming consistency. By tier: **Tier 0: 46.93 min / 54.56% (improved vs. both prior versions); Tier 1: 123.24 min / 27.05% (improved); Tier 2: 148.37 min / 24.00% (0.11 min *worse* than the seed-999-only version); Tier 3: 552.87 min / 7.69% (7.66 min *worse* than the seed-999-only version).** The cell's own printed diagnostic is explicit and unflattering where warranted: it labels both Tier 2 and Tier 3 as **`[WORSENED]`** by multi-seed averaging — a real, honestly-reported finding that seed-averaging, while a clear net win overall and for the well-covered majority of the network, does not uniformly help every segment, and actively costs accuracy on the rarest, most data-starved cold-start legs (Tier 3 is only 13 legs, so this is a high-variance segment where any change swings the average substantially).

---

## Code Cell 49: Phase 1 — Fixing the Deliverable-Blocking Bugs (Phantom Filtering, Denominator, Sample Sizes, Assumptions)

**Inference of the Previous Output:**
Plain English: the model side of the project is now fully re-verified, but Part 4 left behind several unresolved, directly observable bugs in the business deliverables — phantom facilities surviving in tables labeled clean, and a hub-breach percentage whose denominator silently excluded low-volume facilities. Data science interpretation: this cell doesn't touch the model at all; it's purely a data-hygiene and metric-definition correction pass on the Task 1/4/5 deliverable pipeline.

**Explanation of the Upcoming Code:**
Four independent fixes, each with an explicit before/after printout rather than an assumed fix: **(1.1)** `clean_test_legs` is created using the same `is_placeholder_pin` logic already used for `clean_train_legs`, with an explicit phantom-row count verification on both. **(1.2)** the SLA-breach denominator is recomputed two ways side by side — the old, buggy version (summed only across hubs with ≥20 test dispatches) versus the new, correct version (summed across the entire phantom-free test set, no volume filter) — so the size of the earlier overstatement is directly visible. **(1.3)** the FTL vs. Carting table from Cell 47 is rebuilt with an explicit `trip_count` column and a `[LOW SAMPLE]` flag for any bracket/route-type combination under 30 trips. **(1.4)** the revenue-at-risk figure is recomputed with the ₹500/breach assumption printed inline in the same sentence as the dollar (rupee) figure, rather than only stated separately in narration.

**Deliverable and Expected Result:**
Deliverable: a business-facing deliverable pipeline whose numbers are actually correct and whose assumptions are impossible to separate from the figures they produced. Actual result: **752 phantom legs removed from the test set**, with 0 phantom rows confirmed remaining in both clean tables. Denominator fix: **old (buggy) denominator was 964 breaches; the correct, unfiltered denominator is 3,201 breaches** — meaning the earlier "29.9% of network breaches" claim from Cell 47 was overstated by roughly 3.3×; the corrected top-3 share is **9.2% of the true denominator**, still a genuinely strong finding for a memo, just an honest one. FTL/Carting table: Short and Medium brackets both confirmed `[OK]` with thousands of trips each; Long and Ultra-Long Carting rows confirmed `[LOW SAMPLE]` with exactly 0 observed trips, meaning any "Carting vs FTL" comparison at those distances is definitionally impossible from this data, not just noisy. Top-5 hubs (now genuinely phantom-free, matching Cell 47's list minus rounding from the corrected filtering): Bhiwandi_Mankoli_HB (53.1%), Bangalore_Nelmngla_H (51.9%), Bengaluru_Bomsndra_HB (55.3%), Bengaluru_KGAirprt_HB (55.8%), Mumbai Hub (64.4%), with revenue-at-risk now printed as **"₹146,500 (assumption: ₹500/breach — placeholder pending real unit economics)"** directly in the output.

---

## Code Cell 50: Phase 2 — Weighted Graph, Chronic Delay Flags, and FTL Framework (First Attempt at Fixed Labels)

**Inference of the Previous Output:**
Plain English: the numeric deliverable bugs are now fixed, but two Task 1/2 requirements from the PS still haven't been properly built: edge weights that account for actual delay (not just raw connectivity), and network visualizations. Data science interpretation: this cell introduces delay-weighted graph centrality for the first time in the project — every prior betweenness/PageRank computation (Phases 0, 1, 5, and the Track A/B work) used an unweighted graph.

**Explanation of the Upcoming Code:**
Corridor-level median delay factor is used as an edge weight on a newly-built weighted version of the training graph (`G_train_weighted`), and `nx.betweenness_centrality(..., weight='weight')` / `nx.pagerank(..., weight='weight')` are computed for the first time, printed side-by-side against the existing unweighted metrics. The chronic-delay flag from Cell 45/49 is recomputed on phantom-free `clean_train_legs`, plus a second, relative version: within each `(distance_bracket, route_type)` group, corridors in the worst 10% (90th percentile) of delay ratio are separately flagged — a companion metric meant to be more discriminating than the literal `>20%` rule, which flags nearly the whole network. A network visualization is generated (matplotlib), and the FTL vs. Carting recommendation table is rebuilt using the now sample-size-annotated data from Cell 49.

**Deliverable and Expected Result:**
Deliverable: a properly weighted graph, a two-tier chronic-delay definition (literal PS compliance plus a genuinely useful relative version), a first network visualization, and a finalized FTL/Carting recommendation table. Actual result: the top-10 weighted-vs-unweighted betweenness comparison shows the two rankings are mostly, but not perfectly, aligned — this comparison uses raw facility *codes* (e.g. `IND562132AAA`) rather than readable names, a labeling gap corrected in the next cell. **95.9% of corridors flagged chronically delayed under the literal rule; 141 corridors flagged under the relative worst-decile rule** — a much more targeted, usable list. A `FutureWarning` from pandas about `.groupby().apply()` behavior appears in the output but does not affect correctness. FTL/Carting recommendations for Short and Medium brackets state "Improves delay ratio by 0.29/0.39... **Justifies** assumed +25.0% cost" — language later identified and corrected in Cell 51/52, since nothing in this cell's code actually computes whether the time savings numerically justifies the cost assumption; the two figures are stated together without being mathematically linked.

---

## Code Cell 51: Phase 2 [Revised] — Corrected Labels and Narrative (First Version, Contains a Hardcoded-Text Bug)

**Note:** this cell is superseded by Code Cell 52, which is functionally identical except for one fixed bug. Documented here for completeness since both are physically present in the notebook.

**Inference of the Previous Output:**
Plain English: the previous cell's weighted-vs-unweighted comparison used unreadable facility codes and an overclaiming FTL recommendation sentence. Data science interpretation: this cell aims to fix both — readable names via a facility-code-to-name lookup, and softer, decoupled FTL language that presents the time-saving fact without asserting it "justifies" a cost figure the code never actually compares against it.

**Explanation of the Upcoming Code:**
A `hub_names` dictionary (built via `clean_train_legs.groupby('source_center')['source_name'].first().to_dict()`) is used to relabel every facility in the betweenness comparison and the network plot. The FTL/Carting recommendation text is reworded from "Justifies assumed cost" to "Leadership must weigh this time-saving against the assumed cost premium" — a factual statement instead of an unearned conclusion.

**Deliverable and Expected Result:**
Deliverable: readable, executive-facing labels and honestly-scoped FTL language. Actual result: the comparison table now shows real names (Bangalore_Nelmngla_H, Bhiwandi_Mankoli_HB, Delhi_Airport_H, etc.) instead of raw codes. However, this cell's own printed insight states **"8 of the top 10 hubs remain identical"** — a number written directly into a `print()` string rather than computed from the table sitting immediately below it in the same output. Counting the actual table manually shows **9 facilities appear in both columns** (only Kolkata_Dankuni_HB / Hyderabad_Shamshbd_H swap rank 4↔5; Guwahati_Hub drops out and MAA_Poonamallee_HB enters) — meaning the "8" in this cell's own narrative does not match its own data, a hardcoded-number bug corrected in Cell 52.

---

## Code Cell 52: Phase 2 [Revised] — Corrected Labels and Narrative (Second Version, Hardcoded Count Fixed)

**Inference of the Previous Output:**
Plain English: the previous cell's readable-labels fix worked, but its own headline claim ("8 of the top 10") didn't match the table it was describing. Data science interpretation: this is a direct, minimal fix — replacing a hardcoded string with an actual computed set-intersection count between the unweighted and weighted top-10 lists.

**Explanation of the Upcoming Code:**
Functionally identical to Cell 51 in every respect (same `hub_names` lookup, same relabeled comparison, same plot, same FTL wording) — the only change is that the "X of the top 10 remain identical" sentence is now computed from the actual top-10 lists rather than typed in directly.

**Deliverable and Expected Result:**
Deliverable: a betweenness-comparison narrative that's provably consistent with its own printed table. Actual result: identical underlying numbers to Cell 51 (same betweenness values, same facility names, same FTL table), but the insight line now correctly reads **"9 of the top 10 hubs remain identical... These are unavoidable structural bottlenecks"** — matching a manual count of the table (9 shared facilities, one swap: Guwahati_Hub out, MAA_Poonamallee_HB in). This is the version that should be treated as authoritative for the memo, not Cell 51.

---

## Code Cell 53: Final Check — SLA Window Calibration (Leaky, Self-Graded Version)

**Note:** this cell is superseded by Code Cell 54, which fixes a genuine data-leakage issue in the calibration method. Documented here for completeness.

**Inference of the Previous Output:**
Plain English: the labeling and narrative bugs are now fixed; the last remaining loose thread from earlier in the project is the SLA-window coverage figure (last properly measured back in Cell 35 at 75.68%, still short of the 80% target and built on a pre-graph model). Data science interpretation: this cell attempts a quick recalibration against the final `true_averaged_preds`, computing residuals and taking their 10th/90th percentile directly.

**Explanation of the Upcoming Code:**
Residuals (`test_actual − true_averaged_preds`) are computed across the full test set, and the 10th and 90th percentile of those residuals are used directly as the padding applied to `true_averaged_preds` to form the lower/upper ETA bounds — then coverage is checked against the same test set the percentiles were derived from.

**Deliverable and Expected Result:**
Deliverable: a quick recalibrated coverage figure for the final model. Actual result: **Coverage Achieved: exactly 80.0%** against an 80.0% target — but this is a mathematically expected outcome of the method used, not evidence of a well-calibrated model: taking the Nth/(100-N)th percentile of a dataset's own residuals and then checking what fraction of that *same* dataset falls within those percentiles will always land almost exactly at the target coverage by definition, regardless of whether the underlying model generalizes well. This is a same-sample, circular check (calibrating and evaluating on the identical data), not an out-of-sample validation, and is corrected in Cell 54.

---

## Code Cell 54: Honest Check — Zero-Leakage SLA Calibration

**Inference of the Previous Output:**
Plain English: the previous cell's "exactly 80.0%" result was mathematically guaranteed by its own circular method, not a genuine confirmation that the model's uncertainty estimate is trustworthy going forward. Data science interpretation: this cell replaces the self-graded check with a properly held-out one, mirroring the same discipline used for the Phase 3 conformal calibration back in Cell 35, but applied fresh to the final `true_averaged_preds` model.

**Explanation of the Upcoming Code:**
A 15% holdout is split from *training* data (`train_test_split`, never touching the real test set). A fast `HistGradientBoostingRegressor` is trained on the remaining 85% of training data and used to compute residuals on the held-out 15% — residuals the model genuinely never saw during its own fitting. The 10th/90th percentile of *those* residuals defines the padding, which is then applied to the real, final `true_averaged_preds` and evaluated for coverage on the real, still-untouched test set.

**Deliverable and Expected Result:**
Deliverable: a coverage figure that actually means something — measured out-of-sample, not self-graded. Actual result: **padding learned from history: −69.7 to +66.7 minutes; honest test-set coverage: 81.3%** against the 80.0% target (slightly *above* target, not suspiciously exact — a much more credible signature of a genuine calibration than Cell 53's perfect 80.0%); **median ETA window: 136 minutes**, wider than Cell 53's uncalibrated figure — which is the expected and correct direction for the fix, since removing an artificial leak that was making the model look falsely precise should widen the honest window, not narrow it.

---

**End of Part 5 (Code Cells 48–54).** This closes out the deliverable-correctness arc: the final tier audit (with an honest admission that seed-averaging cost some accuracy on the rarest cold-start legs), every phantom-facility and denominator bug from Part 4 fixed and proven, a caught-and-fixed hardcoded-narrative bug (8 vs. 9), and a caught-and-fixed circular-calibration bug (80.0% vs. the honest 81.3%). Say "continue" for Part 6 — Code Cells 55–65, the final part, covering the bounded Phase 3.1–3.7 model-improvement experiments (including Cell 62, "Phase 3.7: Tier-Conditioned Blend Weights," which doesn't appear anywhere earlier in this documentation and will be read fresh from source, same as everything else) and the three-round parallel-route candidate analysis.
