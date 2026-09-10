# Delhivery ETA Notebook — Cell-by-Cell Documentation
### Part 4 of 6: Code Cells 41–47 (Verification Fixes, Track A Final Blend, First Track B Deliverable Passes)

Continuing from Part 3's Cell 40 (first ensemble/chokepoint pass, 52.90 min / 52.98%, built on a single random seed's GraphSAGE embeddings). All numbers re-pulled directly from this notebook's actual output.

---

## Code Cell 41: Fix 1 (Regex Tier Extraction) + Fix 4 (Leakage Verification Suite)

**Inference of the Previous Output:**
Plain English: the chokepoint table from Cell 40 looked plausible on its face, but the facility-tier field it depends on was built with a regex (`_([A-Z]+)\s*\(`) known elsewhere in this project's history to silently mis-tier any facility whose type suffix isn't fully uppercase — e.g. `Guwahati_Hub` (capital H, lowercase "ub") fails to match and falls back to Tier 0 by default. Data science interpretation: this is a string-parsing bug, not a modeling bug, but it directly contaminates a business-facing deliverable (facility tier labels in the chokepoint audit) if left uncorrected.

**Explanation of the Upcoming Code:**
**Fix 1:** the tier-extraction regex is now applied with `.str.upper()` forcing the facility-name suffix to uppercase before matching, so mixed-case suffixes like `Hub` are correctly caught. Before/after tiers for two specifically-named facilities (`Jaipur_Hub`, `Guwahati_Hub`) are printed directly, plus a full crosstab of every training facility's old tier vs. new tier, so the fix's scope is visible and provable rather than merely asserted. The Phase 1 evaluation gate (flat centrality features) is then rerun with the corrected tiers. **Fix 4:** four explicit, code-based (not narrated) leakage checks: (a) does every edge in `G_train` genuinely trace back to `clean_train_legs`, (b) does the shockwave completion log contain zero test-set rows, (c) do any test-only facilities appear as edges inside `G_train`, (d) was `src_dwell_lkp` — the pretraining target for GraphSAGE — built strictly from training data. Each prints an explicit PASS/FAIL based on an actual comparison, not a hardcoded statement.

**Deliverable and Expected Result:**
Deliverable: a facility-tier field that's actually correct, with the fix's real-world impact measured rather than assumed, plus a fully auditable confirmation that no part of the graph pipeline touches test-set information it shouldn't. Actual result — regex fix: two named facilities corrected from Tier 0 to Tier 3; crosstab shows **414 total training facilities had their tier changed** by this fix (386 moved 0→3, 28 moved 0→1); rerunning the evaluation gate with corrected tiers gives **MAE 54.79 → 55.03 (+0.24 min, slightly worse), Acc15 51.60% → 51.95% (+0.35pp, better)** — a genuine, mixed trade-off, not a clean win, confirming the fix was correct but not free. Leakage checks: all four — **(a) 2,349/2,349 edges verified, 0 leaked; (b) 18,948/18,948 log rows verified training-only; (c) 94 test-only nodes identified, 0 found inside `G_train`; (d) `src_dwell_lkp` confirmed built from training-only `train_temp`** — every check returned **PASS**, with real counts printed for each, not just a status word.

---

## Code Cell 42: Fix 3 — GraphSAGE Multi-Seed Stability Check (Single Model)

**Inference of the Previous Output:**
Plain English: the tier bug is now fixed, and every leakage check has genuinely passed — but Cell 39's GraphSAGE "PASS" verdict (+0.57 min / +0.47pp) came from exactly one random initialization, and this project's own history has already shown that Optuna-tuned models can swing by roughly a minute of MAE purely from randomness. Data science interpretation: a single-seed result cannot distinguish a real effect from a lucky draw; a proper test requires repeating the procedure across multiple seeds and comparing the mean improvement against the measured spread.

**Explanation of the Upcoming Code:**
The entire Phase 5 GraphSAGE pipeline — network initialization, 100-epoch training, embedding extraction, HistGB evaluation — is re-run three separate times with seeds 42, 123, and 999, keeping everything else (architecture, hyperparameters, feature set) identical across runs. Mean and standard deviation of MAE and Acc15 are computed across the three runs, and the mean improvement over the corrected Phase 1 baseline (55.03 min) is checked against whether it exceeds one standard deviation of the observed seed-to-seed variance.

**Deliverable and Expected Result:**
Deliverable: determine whether GraphSAGE's earlier single-seed "win" is a stable, trustworthy effect or statistical noise. Actual result: **Seed 42: 54.02 min / 51.06%; Seed 123: 55.31 min / 51.76%; Seed 999: 55.64 min / 51.46%** — a spread of over 1.6 minutes across three otherwise-identical runs. **Mean MAE 54.99 ± 0.858 min, Mean Acc15 51.43% ± 0.352pp.** Comparing to the corrected baseline: mean improvement is only **+0.04 minutes**, far smaller than the 0.858-minute noise band. Verdict: **`WARN/FAIL` — the single-model GraphSAGE improvement does not survive seed variance and should not be treated as a real, reproducible effect on its own.**

---

## Code Cell 43: Fix 2 — Reordered Tier Audit (Evaluated Against the Actual Final Blend) + Corrected Chokepoint Table

**Inference of the Previous Output:**
Plain English: the single-model GraphSAGE result just turned out to be noise-level, but that doesn't necessarily doom the *blended* ensemble, since blending across models is a different question from whether any one component is individually stable. Separately, Cell 40's own tier-by-tier breakdown had a structural ordering bug — it evaluated the single Cell-39 model before the 4-way blend had even been built in that same cell, meaning the "tier audit" wasn't actually describing the model being shipped. Data science interpretation: this cell fixes an evaluation-order bug, not a modeling bug — the blend itself was fine, but what was being measured against fallback tiers wasn't the blend.

**Explanation of the Upcoming Code:**
Section 6.2 (build the 4-way SLSQP-optimized blend using seed-999 embeddings) is deliberately run **before** section 6.1 (the tier breakdown) in this cell, reversing Cell 40's execution order so the tier audit genuinely evaluates the finished blend's predictions, not an intermediate single model. The chokepoint severity table is also rebuilt using the now tier-corrected facility labels from Cell 41.

**Deliverable and Expected Result:**
Deliverable: a tier-by-tier breakdown and chokepoint table that are actually describing the model being shipped, with corrected facility tiers. Actual result — tier audit, old (buggy) vs. new (correctly-ordered) evaluation: **Tier 0: 48.45 → 47.31 min; Tier 1: 122.93 → 124.22 min; Tier 2: 156.54 → 148.26 min; Tier 3: 572.12 → 545.21 min**, with Acc15 by tier now also visible (54.23%, 25.82%, 25.14%, 15.38% respectively) — cold-start tiers remain far weaker than Tier 0, but the corrected evaluation shows real improvement in Tiers 2 and 3 versus the old, wrongly-ordered numbers. Chokepoint table: topped by Bangalore_Nelmngla_H, Bhiwandi_Mankoli_HB, Delhi_Airport_H, Hyderabad_Shamshbd_H, Sonipat_Kundli_H — now with corrected tiers (Guwahati_Hub and Jaipur_Hub both correctly showing Tier 3, not the earlier Tier 0 mislabel).

---

## Code Cell 44: Track A — True Full-Blend Seed Stability (The Verified Final Number)

**Inference of the Previous Output:**
Plain English: everything up to this point has verified pieces individually — the tier fix, the leakage checks, the fact that the *single-model* GraphSAGE result was noise. What hasn't yet been tested is whether the *full 4-way blend*, built fresh with GraphSAGE embeddings from each of 3 different seeds and its own SLSQP weights re-solved each time, is itself a stable improvement. Data science interpretation: this is the properly rigorous version of Cell 42's check — testing the actual object being shipped (the blend), not just one of its four components.

**Explanation of the Upcoming Code:**
Two more targeted verifications run first: an independent re-check that `src_dwell_lkp` (1,425 total facilities) contains zero test-only facilities, and an audit of exactly which facility caused the 28-facility Tier 0→1 shift flagged in Cell 41 (found to be a single facility, `BLR_JPNagar_Pc`, appearing across 28 training rows — resolving what could have looked like a larger anomaly). Then the complete pipeline — GraphSAGE training, embedding extraction, all four component models, SLSQP weight optimization — is rerun end to end for seeds 42, 123, and 999, and the **predictions themselves are averaged across the three seeds** (not just the metrics), producing one final, seed-stabilized prediction array.

**Deliverable and Expected Result:**
Deliverable: the project's single most trustworthy final number — the one meant to be used for every downstream deliverable (memo, chokepoint audit, revenue estimates) from this point forward. Actual result: per-seed blend scores **Seed 42: 52.63 min / 53.05%; Seed 123: 53.02 min / 52.65%; Seed 999: 53.04 min / 52.76%** — **Mean 52.89 ± 0.232 min, 52.82% ± 0.209pp** — a much tighter spread than the single-model check (0.232 vs. 0.858 min std), because averaging four already-diverse models partially cancels out each individual model's seed sensitivity. The final seed-averaged predictions score **MAE 52.72 minutes (vs. 54.17 pre-graph baseline, +1.45 min improvement) and 52.85% within-15% (vs. 51.76%, +1.09pp improvement)**. Verdict: **`[PASS]` — this improvement, unlike the single-model result, comfortably exceeds the measured seed-noise threshold and is the first fully-verified graph-phase win in the notebook.**

---

## Code Cell 45: Track B — First Pass at Task 1 (Stratified Edge Weights) and Task 2 (Chronic Delay + Hub SLA Audit)

**Inference of the Previous Output:**
Plain English: the modeling side of the project just reached its final, verified state (52.72 min). Attention now shifts entirely to the PS's consulting deliverables — Task 1 explicitly asks for graph edge weights stratified by route type and time of day, and Task 2 asks for a chronically-delayed-corridor flag and a hub ranking by SLA-breach contribution, neither of which has been built yet. Data science interpretation: this is a new deliverable-construction phase, not a modeling experiment — no PASS/FAIL gate is used here.

**Explanation of the Upcoming Code:**
Corridor delay ratios are grouped by all four PS-required dimensions simultaneously (`source_center, destination_center, route_type, is_night_dispatch`), filtered to profiles with at least 5 trips, producing the stratified edge-weight table Task 1 asks for. A chronic-delay flag is set wherever `median_delay_ratio > 1.20`, matching the PS's literal `>20%` definition. Separately, an SLA-breach flag is defined per test leg as `|predicted − actual| / actual > 0.15`, then aggregated per source hub to rank facilities by breach count, filtered to hubs with at least 20 test dispatches to avoid noisy small-sample rankings.

**Deliverable and Expected Result:**
Deliverable: the first working versions of the Task 1 and Task 2 outputs. Actual result: **1,523 stratified corridor profiles generated** — but the printed sample of the first 5 rows shows every single one starting with `IND000000*` in the source column, meaning this table was built from an uncleaned data source that still contains the placeholder/phantom facilities the project's own Phase 0 (Cell 32) had explicitly quarantined for graph construction — a real bug in this cell, not yet fixed. **1,460 corridor profiles (95.9% of the network) flagged as chronically delayed** under the literal PS threshold — an honestly reported but, on its own, not very discriminating number, since it flags almost the entire network. The Top 10 hub bottleneck table by SLA breach includes `Gurgaon_Bilaspur_HB (Haryana)` at rank 3 (84 breaches, 29.4% rate) — this facility's underlying code is `IND000000ACB`, meaning a phantom aggregated placeholder has entered a named business deliverable.

---

## Code Cell 46: Track B (Revised) — Regex Fix Applied to Hub Audit + First Task 4 (FTL vs. Carting) Attempt

**Inference of the Previous Output:**
Plain English: the previous cell's hub-ranking table contained a phantom facility masquerading as a real bottleneck. Data science interpretation: this cell attempts to apply the earlier tier-extraction fix to the business-facing hub audit and separately builds a first version of the Task 4 route-type decision framework using the trained ML model directly, rather than historical data.

**Explanation of the Upcoming Code:**
**Fix attempt 1:** the corrected, case-insensitive regex from Cell 41 is applied to rebuild facility tiers for the hub audit. **Fix attempt 2:** the edge-weight and SLA-breach tables are recomputed, labeled "phantom-free" in the cell's own section header. **Task 4 first attempt:** for each distance bracket, the trained ML model is used in a "what-if" fashion — take each historical leg, flip its `is_ftl` flag, and compare the model's predicted ETA under both hypotheses, averaged per bracket.

**Deliverable and Expected Result:**
Deliverable: a phantom-free hub ranking and a first working FTL-vs-Carting decision table. Actual result: the "CORRECTED TOP 5 BOTTLENECK HUBS" table's own header claims phantom-free status, but the printed table **still contains `IND000000ACB Gurgaon_Bilaspur_HB (Haryana)` at rank 3** — meaning despite the section's own label, the phantom-filtering fix did not actually get applied to this specific table, a genuine and directly checkable bug in this cell. The Task 4 "what-if" table shows **`median_time_saved_mins` at exactly 0.0 for all four distance brackets** — flipping the `is_ftl` flag produced essentially no change in the model's prediction, indicating the tree model anchors its prediction to historical corridor behavior (`corridor_factor_hist`) regardless of what the `is_ftl` input says, making it structurally unable to answer this "what-if" question. Despite this, the cell's own printed "INSIGHT FOR MEMO" text asserts a specific recommendation (favor FTL on Long/Ultra-Long routes) that the 0.0-savings data directly contradicts — this insight text does not follow from the numbers printed immediately above it, and is corrected in the next cell's empirical approach.

---

## Code Cell 47: Track B (Empirical) — Historical-Data FTL vs. Carting Framework + Revenue-at-Risk

**Inference of the Previous Output:**
Plain English: the model-based "what-if" approach to FTL vs. Carting just demonstrably failed (0.0 savings across every bracket, contradicted by the accompanying narrative). Data science interpretation: since the tree model can't answer a counterfactual it was never trained to represent, this cell abandons the model-based approach entirely and instead compares FTL's and Carting's **actual historical** delay ratios, bracket by bracket — a purely empirical, non-causal comparison.

**Explanation of the Upcoming Code:**
For each distance bracket, the median `factor` (actual/OSRM ratio) is computed separately for legs actually run as Carting versus legs actually run as FTL, using `clean_train_legs`. Separately, the hub SLA-breach table from Cell 45 is re-filtered to exclude any `source_center` containing `IND000000`, producing a genuinely phantom-free top-5 list this time. A flat ₹500-per-breach assumption is applied to the top-3 hubs' combined breach count to produce a revenue-at-risk figure.

**Deliverable and Expected Result:**
Deliverable: a defensible, non-hallucinated FTL-vs-Carting comparison and a genuinely phantom-free top-5 hub list with a first revenue estimate. Actual result — FTL vs. Carting: **Short: Carting 2.18 vs FTL 1.89; Medium: Carting 2.37 vs FTL 1.98; Long and Ultra-Long: Carting shows `NaN` (zero historical trips), FTL 1.99 and 2.04 respectively** — a genuine finding that Carting is essentially never used for long-haul routes in this network. Top-5 hubs (this time correctly excluding the `IND000000ACB` entry): Bhiwandi_Mankoli_HB, Bangalore_Nelmngla_H, Mumbai Hub, Bengaluru_Bomsndra_HB, Bengaluru_KGAirprt_HB, with revenue-at-risk figures from ₹29,000 to ₹73,000. Headline claim: **"Total SLA Breaches in Test Window: 1,039; Top 3 Hubs = 29.9% of network-wide late deliveries; ₹155,500 revenue-at-risk recovered."** (Flag for the reader: the "1,039" denominator here is the sum of breaches only among hubs with ≥20 test dispatches — not literally every breach in the test set — meaning the 29.9% figure is later found, and corrected, in Part 5 to be an overstatement of the true network-wide share.)

---

**End of Part 4 (Code Cells 41–47).** This covers the full verification arc — the tier-regex fix and its real, measured impact; a complete, code-based leakage audit; the discovery that the single-model GraphSAGE result was noise while the full blend genuinely wasn't; and the first two, still-imperfect passes at the Task 1, 2, 4, and 5 business deliverables, including two real bugs (phantom facilities surviving in tables labeled "phantom-free," and a self-contradicting Task 4 memo insight) that get caught and fixed in Part 5. Say "continue" for Part 5 — Code Cells 48–54, covering the corrected tier-by-tier audit against the true averaged predictions, the standardized phantom-filtering fix, the corrected SLA-breach denominator, and the honest, leakage-free calibration check.
