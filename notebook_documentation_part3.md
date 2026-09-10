# Delhivery ETA Notebook — Cell-by-Cell Documentation
### Part 3 of 6: Code Cells 32–40 (Graph Phases 0–5, First Chokepoint Audit)

Continuing from Part 2's Step-2 tabular blend (54.17 min / 51.76%) and Two-Stage discovery. All numbers below are re-pulled directly from this specific notebook's printed output — several differ from numbers seen in earlier passes of this project, since the GraphSAGE/Optuna cells in this section are unseeded and vary run to run. Nothing here is carried over from memory.

---

## Code Cell 32: Phase 0 — Graph Hygiene and Structural Audit

**Inference of the Previous Output:**
Plain English: the tabular pipeline is now fully built out, with a best-by-MAE and a best-by-accuracy candidate both identified. Data science interpretation: before any graph can be built from this data, the known `IND000000*` placeholder facilities (flagged as `is_placeholder_pin` all the way back in Cell 3) need to be explicitly excluded — otherwise they'd merge into artificial "mega-hubs" and distort every centrality calculation that follows.

**Explanation of the Upcoming Code:**
Four checks/steps, in order: **(0.1)** quarantine placeholder legs from the training set before graph construction. **(0.2)** build `G_train`, a `networkx.DiGraph`, using only the phantom-free legs — nodes are facilities, edges are observed corridors. **(0.3)** a connected-component audit using `nx.connected_components` (on the undirected version) to check whether the network is one cohesive graph or fragmented into isolated clusters — this matters because centrality metrics behave very differently for a node in a large connected component versus a small isolated island. **(0.4)** a betweenness stability check: the training period is split into two temporal halves, betweenness centrality is computed independently on each, and a Spearman rank correlation measures whether the same facilities consistently rank as important across both halves — a low correlation would mean centrality rankings are noisy and shouldn't be trusted for a bottleneck-audit deliverable.

**Deliverable and Expected Result:**
Deliverable: a graph built on genuinely clean data, with explicit evidence about its structural reliability before anything is built on top of it. Actual result: **18,948 training legs → 2,102 quarantined phantom legs → 16,846 valid legs**, producing a graph of **1,545 facilities (nodes) and 2,349 corridors (edges)**. Component audit: **74 distinct components, a giant component of 1,168 facilities, and 73 separate island clusters totaling 377 facilities** — meaning roughly a quarter of the network's facilities sit outside the main connected structure. Betweenness stability: **1,301 nodes common to both temporal halves, Spearman correlation 0.7065**, flagged by the cell itself as `[WARN] moderately stable — proceed with caution` — an honest, self-reported caveat that any later "top bottleneck" ranking carries some real rank instability, not a perfectly settled ordering.

---

## Code Cell 33: Phase 1 — Topological Centrality Features and Evaluation Gate

**Inference of the Previous Output:**
Plain English: the graph now exists and has been checked for structural soundness, with an honest warning that its rankings are moderately, not perfectly, stable. Data science interpretation: the graph itself is not yet a feature — this cell is where its structure first gets converted into numbers a regression model can actually use.

**Explanation of the Upcoming Code:**
Standard graph centrality metrics are computed on `G_train`: betweenness centrality (how often a node sits on the shortest path between other node pairs — identifies structural bridges), PageRank, in-degree, out-degree, and clustering coefficient. Eight resulting columns (source and destination versions of a subset of these) are attached to both `train_graph_df` and `test_graph_df`, expanding the tabular feature count from 25 to 33. A `HistGradientBoostingRegressor` is trained on this expanded set and compared against the same model trained on the pre-graph 25-feature set, using an explicit gate: does adding graph features measurably improve MAE and accuracy, or is the improvement just noise?

**Deliverable and Expected Result:**
Deliverable: the first direct test of whether graph-derived structural information adds real predictive value beyond what the tabular features already capture — the core question the whole graph phase exists to answer. Actual result: **Base (tabular-only) MAE 55.35 min / 50.87% → Graph-Enhanced MAE 54.79 min / 51.60%**, an improvement of **+0.55 min MAE and +0.73pp accuracy**. Verdict printed: **`[PASS]`** — flat topological centrality alone, at essentially zero additional cost beyond computing standard graph metrics, produces the single largest graph-phase improvement found in this notebook, before any neural embedding work is even attempted.

---

## Code Cell 34: Phase 2 — Causal Shockwave Features (Original, Source-Grouped Version)

**Inference of the Previous Output:**
Plain English: flat network position just proved genuinely useful — the next question is whether *dynamic* network conditions (is this facility currently backed up right now?) add anything on top of static structure. Data science interpretation: this tests a time-windowed, causally-guarded feature rather than a static graph property.

**Explanation of the Upcoming Code:**
For each leg, a "shockwave" feature is computed: the smoothed average delay factor of other legs that arrived at the same source facility within the 6 hours immediately preceding this leg's departure — with an explicit guard that only arrivals strictly before this leg's `od_start_time` are counted (avoiding lookahead leakage), and Bayesian smoothing (`m=20`, same style as the corridor lookups) to avoid overreacting to a single truck's arrival at a low-volume facility. The feature is attached and the same PASS/FAIL evaluation gate from Cell 33 is re-run.

**Deliverable and Expected Result:**
Deliverable: test whether recent local congestion propagates in a way that's predictive of a departing leg's own delay. Actual result: **Phase 1 MAE 54.79 → Shockwave MAE 54.98 (−0.18 min), Acc15 51.60% → 51.42% (−0.18pp)** — both metrics moved the wrong direction. Verdict: **`[FAIL]`**. (Note carried forward for context: this specific version groups history by `source_center`, which — as later diagnosed elsewhere in this project's history — measures "other trips that also departed this hub" rather than "trips that recently arrived here," a subtle mismatch from the intended causal question. Cell 35 tests the corrected version.)

---

## Code Cell 35: Phase 2 [Corrected] — Shockwaves (Destination-Grouped) + Phase 3 — Conformal SLA Calibration

**Inference of the Previous Output:**
Plain English: the first shockwave attempt failed, but with a known conceptual bug in how "recent arrivals" was measured — worth testing the corrected version before concluding dynamic congestion doesn't matter here. Data science interpretation: this cell contains two unrelated experiments back to back — a corrected re-test of Cell 34's hypothesis, followed by a completely separate deliverable (SLA window calibration).

**Explanation of the Upcoming Code:**
**Shockwave correction:** the history log is now grouped by `destination_center` rather than `source_center` — correctly measuring "trips that recently *arrived* at this hub," which is the causally correct proxy for "is this facility currently backed up." Same 6-hour window, same causal guard, same evaluation gate as before. **Conformal calibration:** a `HistGradientBoostingRegressor` with `loss='quantile'` at both 0.10 and 0.90 is trained, and asymmetric non-conformity scores are computed separately for the lower and upper tails on a chronological holdout — rather than one shared symmetric widening factor, the lower and upper bounds each get their own, independently-derived padding, reflecting the fact that delivery delays are right-skewed (a truck is rarely dramatically early, but can easily be dramatically late).

**Deliverable and Expected Result:**
Deliverable: a genuine, corrected re-test of the shockwave hypothesis, and a properly asymmetric SLA-window calibration. Actual result — shockwaves: **Phase 1 MAE 54.79 → Corrected Shockwave MAE 55.14 (−0.35 min), Acc15 +0.04pp**. Verdict: **`[FAIL]`** — even with the grouping bug fixed, dynamic congestion still doesn't help, a stronger and more trustworthy negative result than Cell 34's, since the known confound is now removed. Actual result — calibration: coverage improved from **67.62% (before padding) to 75.68% (after padding)** against an 80% target, with median window widening from **90 to 102 minutes** — real progress, though still short of the nominal target at this point in the notebook.

---

## Code Cell 36: Phase 4 — Physics-Informed Target Decomposition (Original, Feature-Starved Version)

**Inference of the Previous Output:**
Plain English: two dynamic/structural feature ideas have now been tested and failed; this cell tries a different kind of change — not a new feature, but a new *architecture* for how the prediction itself is built. Data science interpretation: rather than one model predicting the full wall-clock target directly, this decomposes it into two physically meaningful sub-targets predicted by separate specialist models.

**Explanation of the Upcoming Code:**
Two `HistGradientBoostingRegressor` models are trained: one predicting `driving_residual` (`actual_time − osrm_time`, i.e. how much slower/faster than OSRM's estimate the actual drive was) using only route-geometry features, and one predicting `dwell` (`wall_clock − actual_time`) using only facility/topology-flavored features. The final prediction is reconstructed as `osrm_time + predicted_driving_residual + predicted_dwell`.

**Deliverable and Expected Result:**
Deliverable: test the "Uber DeepETA"-style decomposition philosophy — that physics (driving) and operational chaos (dwell) are better modeled separately than jointly. Actual result: **Phase 1 MAE 54.79 → Decomposed Pipeline MAE 64.41 (−9.62 min), Acc15 51.60% → 46.05% (−5.55pp)**. Verdict: **`[FAIL]`**, and a severe one. The root cause, diagnosed elsewhere in this project's history and directly addressed in Cell 37: the `driving_features` list here excludes `corridor_factor_hist` — which is actually a driving-delay signal (built from `actual_time / osrm_time`), not a dwell signal — meaning the driving submodel was denied the single most powerful feature relevant to its own target.

---

## Code Cell 37: Phase 4 [Corrected] — Target Decomposition, Full Feature Set for Both Submodels

**Inference of the Previous Output:**
Plain English: the decomposition idea failed badly, but for a diagnosable reason — one of the two submodels was missing the exact information it needed most. Data science interpretation: this is a controlled re-test that removes the feature-starvation confound, isolating whether the decomposition architecture itself is flawed or whether the previous failure was purely an artifact of an unfair feature split.

**Explanation of the Upcoming Code:**
Same two-submodel structure as Cell 36, but now **both** the driving-residual model and the dwell model receive the complete expanded feature set (all 33 columns from Phase 1, not an artificially restricted subset). Each submodel's standalone MAE against its own target is explicitly printed before recombination — a diagnostic addition that Cell 36 lacked, allowing this cell to show *where* any remaining error is coming from rather than only the final blended number.

**Deliverable and Expected Result:**
Deliverable: a clean, fair test of whether target decomposition can beat the unified model once the earlier confound is removed. Actual result: submodel diagnostics show **Driving Residual submodel MAE 30.69 min (own target), Dwell submodel MAE 43.78 min (own target)** — both individually reasonable. But recombined: **Phase 1 MAE 54.79 → Decomposed MAE 56.70 (−1.91 min), Acc15 51.60% → 51.19% (−0.40pp)**. Verdict: **`[FAIL]`**, again — but now for a structural reason rather than a feature bug: summing two independently-erring predictions compounds their errors rather than letting one submodel's tree splits implicitly account for the other's behavior the way a single unified model can. This is a trustworthy negative result, since the confound from Cell 36 has been explicitly removed and the failure persists.

---

## Code Cell 38: Install PyTorch Geometric

**Inference of the Previous Output:**
Plain English: three feature/architecture ideas (shockwaves twice, decomposition twice) have now failed; the next planned step needs a genuine neural network library, which isn't installed by default in the Kaggle environment. Data science interpretation: no modeling logic in this cell — it's a pure environment-setup step, required before any `torch_geometric.nn` import can succeed.

**Explanation of the Upcoming Code:**
`!pip install torch-geometric` — a shell command (not Python) that installs the PyTorch Geometric library, which provides graph neural network building blocks including `SAGEConv`, the layer type used for the GraphSAGE model in the next cell.

**Deliverable and Expected Result:**
Deliverable: make the GraphSAGE architecture available to the notebook. Expected result: a successful pip install confirmation (`torch_geometric-2.8.0.post1` downloaded, dependencies already satisfied in this environment) — no model output, purely an environment dependency step.

---

## Code Cell 39: Phase 5 [Corrected] — GraphSAGE Inductive Embeddings

**Inference of the Previous Output:**
Plain English: the graph library is now installed, and everything before this point in the graph phase — flat centrality (helped), shockwaves (failed twice), decomposition (failed twice) — sets up the one remaining, most expensive idea: can a full graph neural network learn something the simpler graph features couldn't. Data science interpretation: this is the most architecturally significant cell in the graph phase, introducing message-passing rather than hand-computed graph statistics.

**Explanation of the Upcoming Code:**
A `torch_geometric.data.Data` object is built from `G_train` — 1,545 nodes, 2,349 edges, with each node described by 5 input features (the flat centrality metrics from Phase 1: betweenness, PageRank, in-degree, out-degree, component size). A 2-layer `SAGEConv` network (`DwellSAGE`) is defined and pretrained with an MSE loss against each node's historical dwell value as the training objective — 100 epochs, with loss printed every 20 epochs. After training, the model's second-layer output (16-dimensional) is extracted as a frozen embedding per facility — not retrained end-to-end with the downstream regressor, but computed once and then treated as 32 new static columns (16 for source, 16 for destination) appended to the tabular feature matrix, bringing the total to 65 features. The same HistGB evaluation gate as every previous phase is then run.

**Deliverable and Expected Result:**
Deliverable: test whether inductive, multi-hop graph embeddings — capable of pooling information from a facility's neighbors' neighbors, unlike the strictly local centrality metrics from Phase 1 — provide signal beyond the flat graph features. Actual result: training loss declines steadily (epoch 20: 1406.16 → epoch 100: 989.27 MSE). Evaluation: **Phase 1 MAE 54.79 → GraphSAGE MAE 54.23 (+0.57 min), Acc15 51.60% → 52.07% (+0.47pp)**. Verdict: **`[PASS]`** — both metrics improved together, a genuine (in this run) positive result. (Important caveat carried forward from elsewhere in this project's history: this specific improvement, from a single untested random seed, was later found to sit well within normal seed-to-seed noise when checked with a proper 3-seed stability test — Part 4 covers that verification directly.)

---

## Code Cell 40: Phase 5.6 & 6 — Final 4-Way Ensemble and First Executive Chokepoint Audit

**Inference of the Previous Output:**
Plain English: GraphSAGE embeddings just passed their own evaluation gate as a standalone addition — the next step is to fold them into the full blended ensemble (not just a single evaluator model) and use the graph's structure to produce the network bottleneck ranking the PS explicitly asks for. Data science interpretation: this moves from "does this feature help a single model" to "what's the final production blend, and what does the graph tell us operationally."

**Explanation of the Upcoming Code:**
Three things happen: **(6.1)** the single GraphSAGE-enhanced model from Cell 39 is broken down by `fallback_tier`, the same diagnostic used in Cell 23. **(6.2)** the full four-model family (base, log-transform, asymmetric-quantile, CatBoost) is retrained on the complete 65-feature graph-enhanced matrix, and SLSQP again solves for optimal blend weights, same procedure as Cell 22 but on the wider feature set. **(6.3)** a "chokepoint severity" score is computed per facility as `betweenness × median_historical_dwell × log(1 + inbound_volume)` and the top 20 facilities are ranked and printed as an executive audit table — directly working toward the PS's Task 2 deliverable.

**Deliverable and Expected Result:**
Deliverable: the graph-enhanced production blend, plus the first version of the ranked bottleneck-hub table the strategy memo will eventually be built around. Actual result — tier breakdown: **Tier 0: 48.24 min / 53.63%; Tier 1: 124.14 min / 28.28%; Tier 2: 157.04 min / 26.29%; Tier 3: 575.87 min / 7.69%** (this specific breakdown, computed from the single Cell-39 model rather than the final verified blend, is later re-audited in Part 4 for exactly this reason). Blend: **weights Base=0.265, Log=0.526, Asym=0.208, Cat=0.001** — CatBoost's contribution is essentially zero here — giving **Final Blend MAE 52.90 min / 52.98% within-15%**, a real improvement over the 54.17-minute tabular reference. Chokepoint table: topped by Bangalore_Nelmngla_H, Bhiwandi_Mankoli_HB, Delhi_Airport_H, Hyderabad_Shamshbd_H, and Sonipat_Kundli_H — this specific table is later found to contain a tier-labeling bug (from the facility-name regex, not from this cell's own logic) that gets diagnosed and fixed in Part 4.

---

**End of Part 3 (Code Cells 32–40).** This completes the initial graph-construction-through-first-blend arc: hygiene and stability auditing, flat centrality (the one clean win), two failed dynamic/architectural experiments (shockwaves, decomposition — each tested twice, once naively and once with a diagnosed bug fixed), GraphSAGE embeddings, and a first-pass ensemble plus chokepoint table that Part 4 goes on to rigorously re-verify. Say "continue" for Part 4 — Code Cells 41–47, covering the regex tier-extraction fix, the leakage verification suite, the 3-seed GraphSAGE stability check, the corrected tier audit, and the first two passes at the Track B business deliverables.
