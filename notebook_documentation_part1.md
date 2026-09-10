# Delhivery ETA Notebook — Cell-by-Cell Documentation
### Part 1 of N: Code Cells 1–20 (Tabular Baseline Pipeline)

**A note on scope before you paste anything:** the notebook actually has **66 code cells (97 cells total including markdown)**, not 45. The numbering below (`Code Cell #1`, `#2`, ...) counts **code cells only**, in execution order, matching what you'd click through in Kaggle. This document covers Code Cells 1–20, which is the complete non-graph tabular pipeline, ending at the 55.18-minute baseline that every later phase (Round 2 features, graph work, Phase 3 experiments) is measured against. Say "continue" for the next part.

Paste each `## Code Cell N` block as a **Markdown cell directly above** the corresponding code cell in Kaggle.

---

## Code Cell 1: Kaggle Environment Setup

**Inference of the Previous Output:**
There is no previous cell — this is the first cell in the notebook. Contextual baseline: no data has been loaded, no variables exist yet. Mathematically/statistically there is nothing to interpret; this is pure environment discovery.

**Explanation of the Upcoming Code:**
In plain English, this cell just looks around the Kaggle environment to see what data files are available, using `os.walk('/kaggle/input')` to print every file path under the input directory. It also imports `numpy`, `pandas`, and `kagglehub` for later use, but doesn't call `kagglehub.dataset_download()` — that line is commented out, meaning the dataset is already attached to this Kaggle session rather than being pulled programmatically.

**Deliverable and Expected Result:**
No modeling deliverable. This is a sanity check that the dataset (`delivery_data.csv`) is actually present and locate its exact path before anything else references it. Expected result: a single printed file path confirming the CSV exists at `/kaggle/input/datasets/ishanyash/delivery-data/delivery_data.csv`.

---

## Code Cell 2: Load the Raw Dataset

**Inference of the Previous Output:**
Plain English: the previous cell confirmed the CSV file exists at a specific path. Data science interpretation: we now have a verified, exact file path with zero risk of a `FileNotFoundError` on the next line.

**Explanation of the Upcoming Code:**
This cell does one thing: `pd.read_csv(...)` loads the entire raw delivery dataset into a DataFrame called `df`. No cleaning, filtering, or transformation happens here — this is the untouched, raw checkpoint-level data (144,867 rows in the full dataset), where each row is one GPS/scan checkpoint along a leg's journey, not one leg itself.

**Deliverable and Expected Result:**
Deliverable: get the raw data into memory as the starting point for every downstream step. No numeric result is expected or printed here — success is simply the absence of an error, since `df` needs to exist before Step 1A can run.

---

## Code Cell 3: Steps 1A–1D — Leg Construction, Terminal-Row Recovery, Cleaning, Timestamp Parsing

**Inference of the Previous Output:**
Plain English: the raw CSV is now sitting in memory as `df`, but every row is a checkpoint, not a leg — the same physical journey from Facility A to Facility B is spread across multiple rows. Data science interpretation: this is unaggregated, leg-level and trip-level structure has not yet been recovered; row count still equals the full raw checkpoint count.

**Explanation of the Upcoming Code:**
This is the single most important cell in the whole cleaning pipeline, doing four things back to back:
- **1A (leg identification):** builds a unique `leg_id` for every leg using `.cumsum()` on a boolean condition — a new leg starts whenever `trip_uuid`, `source_center`, or `destination_center` changes from the previous row. This is a classic "change-detection run-length" trick: the boolean OR condition is `True` at every boundary, and `.cumsum()` turns those boundary flags into a monotonically increasing group ID.
- **1B (terminal-row recovery):** the dataset stores each leg's checkpoints in *reverse* order, and `is_cutoff == False` marks the one row per leg holding the complete, final totals. But 251 legs have no such row at all (single-checkpoint legs), so those get explicitly recovered via `groupby('leg_id').tail(1)` on the missing group and concatenated back in — otherwise those legs would silently disappear.
- **1C (cleaning):** two known data-quality issues are fixed — `segment_factor == -1.0` (a divide-by-zero sentinel, not a real measurement) is converted to `NaN`, and physically impossible negative `segment_actual_time` values are also nulled out. A `is_placeholder_pin` flag is created to mark facilities coded `IND000000*`, which are aggregated placeholder PINs rather than real physical hubs.
- **1D (timestamps):** four timestamp columns are parsed with `format='mixed'`, which handles the fact that `cutoff_timestamp` mixes whole-second and microsecond string formats — a naive parser would silently treat the unrecognized format as missing data.

**Deliverable and Expected Result:**
Deliverable: a clean `legs` DataFrame — one row per leg (not per checkpoint) — with no silent data loss, no leaked sentinel values, and correctly typed timestamps. No metric is printed here (it's pure ETL); the expected result is a `legs` DataFrame ready for feature engineering, with row count equal to the true leg count (26,369 in the full audit) rather than the raw checkpoint count.

---

## Code Cell 4: Step 2 — Production Feature Engineering

**Inference of the Previous Output:**
Plain English: `legs` now exists as one clean row per leg. Data science interpretation: the DataFrame is at the correct grain for feature engineering — every row is now an independent observation for modeling, rather than a mix of intermediate and final checkpoints.

**Explanation of the Upcoming Code:**
This cell builds four families of engineered features directly on `legs`:
- **Dispatch timing:** `dispatch_hour`, `dispatch_dayofweek` from `od_start_time`; `is_night_dispatch` (8PM–4AM flag); `dispatch_planning_gap_min` (time between when the trip was logged and when it physically departed) — all legal features since they're known before or at the moment of dispatch.
- **Facility hierarchy:** a regex `_([A-Z]+)\s*\(` pulls the tier suffix (e.g. `HB`, `H`, `DC`) out of facility name strings, mapped to a numeric tier 0–4 via `tier_map`. From this, `is_hub_to_hub` and `is_feeder_drop` are derived as structural route-type flags.
- **Geography:** the first two digits after `IND` in the facility code are extracted as a PIN "circle" prefix, and `is_cross_region` flags whether source and destination fall in different circles.
- **Route geometry:** `osrm_speed_kmh` (implied driving speed from OSRM's own distance/time) and `circuity_index` (how much longer the OSRM road route is than the straight-line distance) are computed as physical descriptors of the corridor.

**Deliverable and Expected Result:**
Deliverable: a richer feature set describing *when*, *where*, and *what kind* of movement each leg represents, entirely from information known before or at dispatch (so nothing here can leak the outcome). No numeric result is printed; success is simply that these columns now exist without error on `legs`.

---

## Code Cell 5: Step 3 — Target Definition and Temporal Train/Test Split

**Inference of the Previous Output:**
Plain English: `legs` now carries dispatch, hierarchy, geography, and geometry features. Data science interpretation: the feature space is built; what's still missing is a clearly defined prediction target and a split that respects the fact that this is time-series-like operational data, not i.i.d. samples.

**Explanation of the Upcoming Code:**
Two targets are explicitly separated: `y_driving` (`actual_time` — pure movement time, the fair benchmark against OSRM) and `y_wall_clock` (`start_scan_to_end_scan` — the full customer-facing duration including facility dwell). `dwell` is computed as their difference, needed later for facility-level historical dwell encodings. The train/test split then uses the dataset's own pre-existing `data` column (`training` / `test`) — a **temporal** split, not a random one, so that the test set represents legs whose trips were created strictly after the training set's trips, simulating real deployment rather than shuffled cross-validation.

**Deliverable and Expected Result:**
Deliverable: two well-defined targets and a leakage-safe temporal split. Expected result: `train_legs` and `test_legs` now exist as separate DataFrames, with the split boundary sitting on the actual Sept 26/27 trip-creation cutoff in the underlying data — no explicit output is printed, but this split is what every later "train-only lookup" rule depends on.

---

## Code Cell 6: Step 4 — Target Encoding, Smoothed Lookups, and the 4-Tier Fallback Ladder

**Inference of the Previous Output:**
Plain English: we now have a target-safe split with `train_legs` and `test_legs` as separate, non-overlapping sets. Data science interpretation: it's now safe to compute historical aggregate statistics, since anything built from `train_legs` alone cannot leak test-set outcomes.

**Explanation of the Upcoming Code:**
`factor` (`actual_time / osrm_time`, the core delay ratio) is computed on training rows only. `build_lookup()` implements **Bayesian m-estimate smoothing**: for a grouping key (corridor, facility, schedule, or region), it blends that group's own median toward the global median, weighted by how many observations back it up (`m=20` acts as a virtual prior sample size) — this is exactly the formula `(group_median * group_size + global_median * m) / (group_size + m)`, which prevents a corridor seen only 2–3 times from getting an overconfident, noisy estimate. Five separate lookups are built this way: corridor-level delay factor, source-facility dwell, destination-facility dwell, schedule-level factor, and region-pair factor. `get_fallback_features()` then implements a **4-tier fallback ladder**: try the exact corridor first, then the schedule, then the region pair, then fall back to the global median — mirroring how a dispatcher would reason about an unfamiliar route ("I don't know this exact corridor, but I know this schedule / this region generally").

**Deliverable and Expected Result:**
Deliverable: the historical-encoding features that, per the original data audit, are the single biggest lever in the entire modeling pipeline, plus explicit visibility into cold-start coverage. Expected/actual result printed: **94.2% of test legs have an exact corridor match (Tier 0), 3.3% fall back to schedule (Tier 1), 2.4% to region (Tier 2), and only 0.2% (13 legs) hit the pure global fallback (Tier 3)** — meaning the ladder covers nearly the entire test set with the most specific information available.

---

## Code Cell 7: Step 5 — First Baseline Models (Driving Time and Wall-Clock)

**Inference of the Previous Output:**
Plain English: the fallback ladder confirmed that 94%+ of test legs have real corridor history to draw on — the historical features aren't mostly guesswork. Mathematically: the coverage table validates that `corridor_factor_hist` will carry real signal, not just the global median, for the overwhelming majority of rows.

**Explanation of the Upcoming Code:**
This trains the **first real models** in the project: two separate `HistGradientBoostingRegressor` instances with entirely default hyperparameters (just `random_state=42`), one predicting `y_driving`, one predicting `y_wall_clock`. The feature set here is deliberately small — 9 columns: OSRM time/distance, `is_ftl`, dispatch hour/day/gap, and the three historical encodings from Cell 6. `route_type` is encoded to a binary `is_ftl` flag. No leg-shape (hop-level) features are included yet.

**Deliverable and Expected Result:**
Deliverable: the project's very first model-vs-no-model comparison point — a "how much does just switching from OSRM to a basic ML model help" baseline. Actual result: **Driving Time MAE 35.90 min (42.43% within-15%); Wall-Clock MAE 59.73 min (43.84% within-15%)** — already a dramatic improvement over the original audit's finding that raw OSRM alone gets within-15% on wall-clock only 0.42% of the time.

---

## Code Cell 8: Rebuild Pipeline + Add Hop-Level Leg-Shape Features

**Inference of the Previous Output:**
Plain English: the first baseline already beats raw OSRM by a huge margin, but 59.73 minutes of average error is still a lot — the model has no way yet to tell "a leg that broke into many small hops" from "a leg that ran in one smooth stretch." Data science interpretation: the 9-feature set captures corridor identity and dispatch timing, but nothing about the *sub-leg structure* of the journey.

**Explanation of the Upcoming Code:**
This cell rebuilds the entire pipeline from the raw CSV (re-running leg construction, since it needs the original checkpoint-level `segment_osrm_time`/`segment_osrm_distance` columns that get collapsed away once legs are aggregated to terminal rows). New here: **hop-level aggregation** — for every leg, `n_hops` (how many checkpoints/segments it broke into), `hop_time_max` (the single longest hop), and `hop_speed_std` (how uneven the hop speeds were) are computed by grouping the raw checkpoint rows by `leg_id` before collapsing to terminal rows. These three "leg shape" features are then merged onto the terminal-row `legs` table, expanding the feature set from 9 to 12 columns.

**Deliverable and Expected Result:**
Deliverable: a richer 12-feature matrix that captures not just *what* corridor a leg is on, but *how* the journey was broken up — information no single-number OSRM estimate contains. Expected result: no model is trained in this cell; it prints `Data Prep Complete! X_train shape: (18948, 12)`, confirming the row count matches the training-set leg count and the feature count has grown from 9 to 12.

---

## Code Cell 9: GridSearchCV Hyperparameter Tuning

**Inference of the Previous Output:**
Plain English: we now have the expanded 12-feature training matrix ready, but the model trained on it so far still uses scikit-learn's default hyperparameters — nothing has been tuned for this specific dataset yet. Data science interpretation: default hyperparameters are a reasonable starting point but rarely optimal; there's likely room to improve MAE just by searching the hyperparameter space.

**Explanation of the Upcoming Code:**
A `GridSearchCV` exhaustively tries every combination across 5 hyperparameters (`learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`), each with 2 candidate values — 2⁵ = 32 combinations. Each combination is evaluated using `TimeSeriesSplit(n_splits=3)`, which — unlike standard k-fold — always trains on an earlier chronological block and validates on a later one, respecting the temporal nature of the data and avoiding the "training on the future to predict the past" leak that plain shuffled CV would introduce. 32 combinations × 3 folds = 96 total model fits, scored by negative MAE (since `GridSearchCV` maximizes by convention).

**Deliverable and Expected Result:**
Deliverable: a properly cross-validated, tuned HistGB model as an improvement over the untuned Cell 7 baseline. Actual result: best params `{l2_regularization: 1.0, learning_rate: 0.1, max_iter: 200, max_leaf_nodes: 63, min_samples_leaf: 20}`, giving **Wall-Clock MAE 56.60 min (51.27% within-15%)** — a meaningful jump in within-15% accuracy (43.84% → 51.27%) from the untuned baseline, showing tuning plus the leg-shape features together move the needle substantially.

---

## Code Cell 10: Optuna Hyperparameter Search

**Inference of the Previous Output:**
Plain English: grid search already found a real improvement over the untuned model, but it only tried 2 fixed values per hyperparameter — there could be better combinations sitting between the grid points that were never tested. Mathematically: grid search explores a coarse lattice of the hyperparameter space; a smarter, adaptive search could explore continuously and focus effort where results look promising.

**Explanation of the Upcoming Code:**
`optuna.create_study()` runs a **Tree-structured Parzen Estimator (TPE)** search — a Bayesian optimization method that models which regions of hyperparameter space tend to produce low error and samples more from those regions in later trials, rather than exploring uniformly like grid search. The search space here is continuous (`suggest_float` with `log=True` for `learning_rate` and `l2_regularization`, meaning it samples on a log scale appropriate for parameters that vary over orders of magnitude) and runs 20 trials, each internally scored the same way as Cell 9 (`TimeSeriesSplit(n_splits=3)`, MAE). `loss='absolute_error'` is fixed so the search optimizes MAE directly rather than squared error.

**Deliverable and Expected Result:**
Deliverable: a more thoroughly searched hyperparameter set than grid search could find in the same 20-ish evaluation budget. Actual result for this run: `{learning_rate: 0.0547, max_iter: 350, max_leaf_nodes: 50, min_samples_leaf: 10, l2_regularization: 0.0053}`, giving **Wall-Clock MAE 56.50 min (50.52% within-15%)** — roughly on par with grid search's MAE, slightly lower accuracy this run. (Note: because the Optuna study isn't given a seeded sampler, re-running this cell can produce a different result each time — this specific run's numbers shouldn't be treated as the single canonical Optuna outcome.)

---

## Code Cell 11: CatBoost Baseline

**Inference of the Previous Output:**
Plain English: two different tuning approaches on the same HistGB model family have now both landed in a similar 56–57 minute MAE range — suggesting HistGB itself may be near its ceiling on this feature set. Data science interpretation: it's worth testing a structurally different gradient-boosting implementation to see if a different tree-building algorithm captures different patterns in the same data.

**Explanation of the Upcoming Code:**
`CatBoostRegressor` is trained with `loss_function='MAE'` (directly matching the business metric) and simple, untuned settings (`iterations=400, learning_rate=0.1`). CatBoost differs from HistGB mainly in how it grows trees (ordered boosting, symmetric trees) and how it natively handles categorical variables — though here the input is fully numeric, so that categorical-handling advantage isn't being exercised. This is a first, untuned pass, purely to see where CatBoost lands before investing in tuning it.

**Deliverable and Expected Result:**
Deliverable: a second model family to diversify the eventual ensemble, and a baseline CatBoost number to compare its own tuning against later. Actual result: **CatBoost Wall-Clock MAE 57.53 min (49.59% within-15%)** — worse than both tuned HistGB variants so far, as expected for an untuned model.

---

## Code Cell 12: Untuned Quantile Regression (10th/90th Percentile ETA Windows)

**Inference of the Previous Output:**
Plain English: we now have a single best-guess ETA number, but a dispatcher often needs a *range* ("arrives between X and Y minutes"), not just one number — especially given how skewed delivery delays are. Data science interpretation: point predictions minimize average error but say nothing about the width of the plausible outcome distribution; a quantile model is needed to characterize that spread.

**Explanation of the Upcoming Code:**
Two separate `HistGradientBoostingRegressor` models are trained with `loss='quantile'` — one at `quantile=0.10` (predicts the value below which only 10% of outcomes fall — the optimistic/fast estimate) and one at `quantile=0.90` (the pessimistic/slow estimate). Together they define an 80% prediction interval: in theory, 80% of real outcomes should fall between the two predictions. Both use fully default hyperparameters — this is a first, untuned pass at the concept.

**Deliverable and Expected Result:**
Deliverable: a first working version of a dispatcher-facing ETA *range*, not just a point estimate — directly useful for the eventual SLA-window deliverable. Expected result: sample printed windows (e.g. "Leg 1: 60–88 min, actual 80 min") that look plausible leg-by-leg; formal coverage/width metrics aren't computed until Cell 17, after the tuned version is built.

---

## Code Cell 13: Optuna Tuning for CatBoost

**Inference of the Previous Output:**
Plain English: we now have a first rough sense of what an ETA range looks like, and separately we know untuned CatBoost (57.53 min) trails the tuned HistGB models. Data science interpretation: before drawing any conclusion about which model family is "better," CatBoost needs the same tuning investment HistGB already received, or the comparison is unfair.

**Explanation of the Upcoming Code:**
Same TPE-based Optuna search pattern as Cell 10, but now searching CatBoost's own hyperparameter space: `iterations`, `learning_rate`, `depth` (CatBoost's analogue to tree complexity), and `l2_leaf_reg`. 15 trials, each scored via the same `TimeSeriesSplit(n_splits=3)` cross-validation used throughout, so the comparison against HistGB's tuning stays apples-to-apples.

**Deliverable and Expected Result:**
Deliverable: a properly tuned CatBoost configuration to replace the untuned Cell 11 baseline. Actual result: best params found were `{iterations: 500, learning_rate: 0.0962, depth: 9, l2_leaf_reg: 2.264}` — note `depth=9` is notably deep, a hyperparameter choice not yet stress-tested for overfitting risk at this point in the notebook.

---

## Code Cell 14: Optuna Tuning for Quantile Models (q=0.10 and q=0.90)

**Inference of the Previous Output:**
Plain English: CatBoost now has tuned hyperparameters ready to be trained and evaluated in the next cell. Data science interpretation: the tuned CatBoost config exists but hasn't been used to train a final model yet — this cell runs in parallel on a separate, unrelated task (tuning the quantile models), not sequentially dependent on Cell 13's tuned model.

**Explanation of the Upcoming Code:**
Two more Optuna studies, using the same TPE search pattern again, this time tuning the untuned quantile models from Cell 12. `objective_quantile()` is parameterized by `quantile_level` and reused for both the 10th and 90th percentile searches (10 trials each). Note the code comment's own caveat: the objective evaluates plain MAE during cross-validation, not actual quantile/pinball loss — meaning trials are being selected on "closest to the median" performance, which is a reasonable proxy but not a perfect match to what quantile regression is actually trying to optimize.

**Deliverable and Expected Result:**
Deliverable: tuned hyperparameter sets for both boundary models, to replace the untuned quantile models from Cell 12. No metrics are printed in this cell — it just runs both searches silently (verbosity suppressed) and stores `study_q10` / `study_q90` for the next cell to consume.

---

## Code Cell 15: Extract and Evaluate the Tuned Quantile Models

**Inference of the Previous Output:**
Plain English: the two quantile searches from the previous cell have finished running, and their best-found hyperparameters are sitting in memory, unused until this cell trains real models with them. Data science interpretation: `study_q10.best_params` / `study_q90.best_params` are Optuna's search results, not yet materialized model objects.

**Explanation of the Upcoming Code:**
The best hyperparameters from each study are extracted and merged with the fixed settings (`loss='quantile'`, the specific quantile level, `random_state=42`). Both models are then retrained on the *full* training set (not just cross-validation folds) and used to predict on the real held-out test set — the standard "tune on CV, then refit on all training data" pattern.

**Deliverable and Expected Result:**
Deliverable: the tuned version of the ETA-range models from Cell 12, ready for formal coverage evaluation. Actual result: tuned params were `{learning_rate: 0.166, max_iter: 350, min_samples_leaf: 42}` for q10 and `{learning_rate: 0.188, max_iter: 350, min_samples_leaf: 30}` for q90; sample windows are printed for 10 test legs, visually tighter and more sensible than the untuned Cell 12 version (e.g., "Leg 1: 64–91 min, actual 80 min").

---

## Code Cell 16: Extract and Evaluate the Tuned CatBoost Model

**Inference of the Previous Output:**
Plain English: the quantile models are now tuned and generating ETA ranges — a separate thread of work from the point-estimate models. Data science interpretation: this cell now returns to the Cell 13 thread, materializing the tuned CatBoost hyperparameters into an actual trained model for the first time.

**Explanation of the Upcoming Code:**
`best_params_cat` from Cell 13's Optuna study is extracted, the required static arguments (`loss_function='MAE'`, `random_seed=42`, `verbose=False`) are added, and a full `CatBoostRegressor` is trained on all of `X_train` / `y_train_wall`, then evaluated on the real test set — same refit-then-predict pattern as Cell 15.

**Deliverable and Expected Result:**
Deliverable: the tuned CatBoost point-estimate model, to be compared against tuned HistGB and later included in the ensemble. Actual result: **Wall-Clock MAE 55.91 min (51.21% within-15%)** — CatBoost's biggest jump yet, closing much of the gap to HistGB's tuned numbers (56.50 min) and now sitting slightly ahead of it on MAE.

---

## Code Cell 17: Quantile Model Coverage and Window-Size Metrics

**Inference of the Previous Output:**
Plain English: tuned CatBoost is now a strong standalone model, and separately, the tuned quantile models exist but have only been eyeballed on 10 sample legs — no formal accuracy number for the *range* itself has been computed yet. Data science interpretation: sample-level printouts aren't a substitute for an aggregate coverage statistic across the full test set.

**Explanation of the Upcoming Code:**
Two metrics are computed across the entire test set: **coverage** (`np.mean(in_window) * 100`) — what fraction of actual outcomes fall inside the predicted 10th–90th percentile window — and **median window size** — how wide that range typically is. Coverage is the honesty check (does an "80% interval" actually contain the real answer 80% of the time?); window size is the usefulness check (a window that's always 800 minutes wide would have perfect coverage but be useless to a dispatcher).

**Deliverable and Expected Result:**
Deliverable: a single, trustworthy pair of numbers describing whether the tuned quantile models can actually be handed to operations as a real SLA window. Actual result: **Actual Coverage 69.5%** against an **80.0% target**, with a **94-minute median window** — a real shortfall (roughly 10.5 percentage points under target) that motivates the later conformal-calibration work to properly fix this gap rather than leave it under-covering.

---

## Code Cell 18: Ensembling — Simple Blend, Optimal Static Weight, Ridge Meta-Stacking

**Inference of the Previous Output:**
Plain English: at this point tuned HistGB and tuned CatBoost are two separately strong models sitting at roughly similar accuracy — the natural next question is whether combining their predictions beats either one alone. Mathematically: if the two models make somewhat different, not-fully-correlated errors, averaging their predictions can reduce variance and improve MAE even without any new information.

**Explanation of the Upcoming Code:**
Three ensembling strategies are tried, in increasing sophistication:
- **A. Simple 50/50 blend** — a flat average of `preds_hist` and `preds_cat`.
- **B. Optimal static weight search** — a brute-force `np.linspace(0, 1, 101)` sweep finds the single blend weight `w` (applied as `w*hist + (1-w)*cat`) that minimizes test MAE.
- **C. Ridge meta-learner stacking** — a more advanced technique: out-of-fold (OOF) predictions are generated across `TimeSeriesSplit(n_splits=4)` chronological folds (so the meta-learner never sees a fold's own model's in-sample prediction), then a `Ridge(alpha=10.0, positive=True, fit_intercept=False)` regression learns weights for combining the two models' OOF predictions, which are then applied to the real test-set predictions.

**Deliverable and Expected Result:**
Deliverable: identify whether blending helps at all, and if so, which blending strategy works best. Actual result: **Simple blend 55.40 min / 51.62%; Optimal weight (0.42 HistGB / 0.58 CatBoost) 55.38 min / 51.62%; Ridge stack 56.48 min / 50.34%.** The key finding, confirmed and flagged repeatedly throughout the rest of this project: **the sophisticated Ridge stacking approach performs *worse* than a simple weighted average**, likely because early `TimeSeriesSplit` folds train on very little data, making the OOF signal the meta-learner learns from noisier than what the fully-trained models produce at real test time.

---

## Code Cell 19: Advanced Targeting — Log Transform, Asymmetric Loss, and the "Super Blend"

**Inference of the Previous Output:**
Plain English: simple blending already beat both individual models, and Ridge stacking was confirmed as a dead end for this data. Data science interpretation: rather than combining two models trained the same way, this cell explores changing *what* the models are trained to optimize, which can produce genuinely different error patterns worth blending in.

**Explanation of the Upcoming Code:**
Three new techniques:
- **Log-transformed target:** `HistGradientBoostingRegressor(loss='squared_error')` is trained on `np.log1p(y_train_wall)` instead of the raw target, then predictions are transformed back with `np.expm1()`. Training in log-space compresses the influence of extreme outliers (very long delays), similar in spirit to RMSLE.
- **Asymmetric quantile loss:** `loss='quantile', quantile=0.52` — a deliberate, slight shift away from the true median (0.50), intentionally biasing the model to over-predict slightly, padding the ETA to protect against under-promising the customer.
- **Super Blend:** a fixed-weight combination (`0.50 * hist + 0.25 * log + 0.25 * asym`) of the tuned HistGB model with these two new variants.

**Deliverable and Expected Result:**
Deliverable: test whether changing the training objective (not just the model family) produces genuinely complementary predictions worth blending. Actual result: **Log-transformed 55.40 min / 51.18%; Asymmetric (q=0.52) 55.57 min / 51.52%; Super Blend 55.21 min / 51.73%** — the Super Blend is the best MAE seen in the tabular pipeline so far, confirming that objective diversity (not just model diversity) adds real value to the ensemble.

---

## Code Cell 20: Final Consolidated Pipeline — The 55.18-Minute Tabular Baseline

**Inference of the Previous Output:**
Plain English: individual experiments (Cells 7–19) have each been run somewhat independently, each rebuilding parts of the pipeline along the way — at this point it's worth consolidating everything into one clean, self-contained cell that rebuilds the full pipeline from scratch and produces a single trustworthy final number. Data science interpretation: this is a reproducibility and integration checkpoint, collapsing several experimental threads (base HistGB, log-transform, asymmetric quantile, tuned CatBoost) into one canonical blend.

**Explanation of the Upcoming Code:**
This cell rebuilds the entire pipeline end to end — leg construction, hop-shape features, cleaning, timestamp parsing, dispatch/target features, train-only historical lookups — then trains four models on the resulting 12-feature matrix: base absolute-error HistGB, log-transform HistGB, asymmetric quantile HistGB (q=0.52), and tuned CatBoost, each with hardcoded hyperparameters carried over from the earlier tuning cells. These four are combined with fixed weights: `0.40*base + 0.20*cat + 0.20*log + 0.20*asym`.

**Deliverable and Expected Result:**
Deliverable: the official, reproducible **tabular (non-graph) baseline** that every subsequent phase of this project — Round 2's expanded features, the entire graph/GraphSAGE effort, and all of Phase 3's experiments — is measured against. Actual result: **Wall-Clock MAE 55.18 minutes, Within-15% Accuracy 51.83%.** (Note for context carried forward in this documentation: this specific 0.40/0.20/0.20/0.20 weighting was **not** re-derived from an optimizer — it was hand-picked — and a later audit found the earlier Cell 19 Super Blend's weights actually produced a slightly *better* number, 55.21/51.73 vs this cell's 55.18/51.83 being roughly comparable; the real optimization of these weights doesn't happen until the SLSQP-based re-optimization in Round 2.)

---

**End of Part 1 (Code Cells 1–20).** This covers the complete tabular baseline pipeline, from raw CSV to the locked 55.18-minute reference number. Say "continue" for Part 2, which covers the Round 2 expanded-feature pipeline, the SLSQP blend re-optimization, the tier-by-tier audit, quantile calibration, Huber loss, two-stage classification+regression, and OOF target encoding (Code Cells 21 onward).
