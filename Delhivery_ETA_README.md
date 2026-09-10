# Delhivery ETA Prediction & Network Intelligence

An end-to-end analytics and machine-learning project for **delivery ETA prediction, network bottleneck analysis, SLA calibration, and route-intervention planning**.

The project starts from raw checkpoint-level delivery data, reconstructs physical shipment legs, builds leakage-safe temporal features, trains and evaluates multiple ETA models, enriches them with delivery-network graph information, and converts the final model into operational deliverables for hubs, corridors, SLA risk, and route strategy.

> **Final verified model:** 52.72 minutes MAE and 52.85% within-15% accuracy on the notebook's final seed-averaged test prediction.

---

## 🎯 Project Objective

The project addresses two connected operational questions:

1. **ETA Prediction:** Can delivery duration be predicted accurately enough to support operational planning and customer-facing SLAs?
2. **Network Intelligence:** Which hubs and corridors are structurally responsible for delays, and where should Delhivery prioritize operational or routing interventions?

The notebook deliberately evaluates both the predictive model and the business deliverables instead of stopping at a single model score.

---

## 🔄 End-to-End Workflow

```text
Raw checkpoint-level delivery data
                │
                ▼
      Leg reconstruction & cleaning
                │
                ▼
       Leakage-safe feature engineering
                │
                ▼
       Temporal train / test split
                │
                ▼
 Historical encodings + fallback ladder
                │
                ▼
     Tabular ETA model ensemble
                │
                ▼
      Network graph construction
                │
        ┌───────┴────────┐
        ▼                ▼
   Centrality         GraphSAGE
   features           embeddings
        └───────┬────────┘
                ▼
        4-model final blend
                │
                ▼
      Seed-averaged prediction
                │
        ┌───────┼──────────────┐
        ▼       ▼              ▼
   SLA window  Hub / corridor  Route
   calibration audit           interventions
```

---

## 📦 Dataset

The project uses the **Delhivery delivery-data dataset** available in the Kaggle environment.

The raw dataset contains **144,867 checkpoint-level rows**. These checkpoints are transformed into **26,369 delivery legs**, with each leg representing movement between a source and destination facility.

A key part of the preprocessing is recovering **251 single-checkpoint legs** that would otherwise be lost when identifying terminal rows.

The final training graph is built after removing placeholder facilities (`IND000000*`) so they do not distort network topology.

---

## 🧹 Data Preparation

The first stage converts raw operational scans into a modeling-ready leg-level dataset.

### Leg construction

A new leg is identified when any of the following changes:

- `trip_uuid`
- `source_center`
- `destination_center`

This produces a unique `leg_id` for each movement.

### Data-quality fixes

The cleaning pipeline:

- Converts the `segment_factor == -1.0` sentinel to `NaN`
- Removes physically impossible negative `segment_actual_time`
- Flags placeholder facilities coded as `IND000000*`
- Parses mixed-format timestamps safely
- Recovers missing terminal rows for single-checkpoint legs

### Targets

Two targets are separated:

- **Driving time:** `actual_time`
- **Wall-clock/customer duration:** `start_scan_to_end_scan`

The final business target is the wall-clock duration because it captures both movement and facility dwell.

---

## 🧠 Feature Engineering

The project builds features describing **when, where, and how a shipment moves through the network**.

### Dispatch & temporal features

- Dispatch hour
- Day of week
- Night-dispatch flag
- Dispatch planning gap

### Facility structure

- Source / destination facility tier
- Hub-to-hub flag
- Feeder-drop flag
- Facility hierarchy indicators

### Geography

- PIN-circle prefixes
- Cross-region movement

### Route geometry

- OSRM distance / speed information
- Circuity index
- Route-level geometric descriptors

---

## 🧮 Historical Encoding & Cold-Start Handling

Historical delivery behavior is one of the strongest predictive signals in the project.

A smoothed historical lookup is constructed using a **Bayesian m-estimate** with `m = 20`, shrinking small groups toward the global median.

Historical signals are calculated using training data only.

The model uses a **four-tier fallback ladder**:

```text
Tier 0 → Exact corridor history
   ↓
Tier 1 → Schedule history
   ↓
Tier 2 → Region-pair history
   ↓
Tier 3 → Global historical median
```

Test-set coverage:

| Fallback Tier | Coverage |
|---|---:|
| Tier 0 — Exact corridor | 94.2% |
| Tier 1 — Schedule | 3.3% |
| Tier 2 — Region | 2.4% |
| Tier 3 — Global | 0.2% |

Only **13 test legs** reached the global fallback.

The tier analysis also exposes an important operational reality: model accuracy deteriorates sharply as historical specificity disappears.

---

## 🤖 Modeling Approach

The project evaluates several modeling strategies rather than relying on a single estimator.

### Tabular models

The main ensemble contains:

- HistGradientBoosting — absolute-error objective
- HistGradientBoosting — log-transformed target
- HistGradientBoosting — asymmetric / quantile-style formulation
- CatBoost

Blend weights are optimized using **SLSQP** subject to:

```text
weight >= 0
sum(weights) = 1
```

The optimized expanded-feature blend achieved:

**54.17 min MAE / 51.76% within-15%**

before the graph phase.

---

## 🕸️ Network Graph Modeling

The delivery network is represented as a directed graph:

- **Nodes:** facilities
- **Edges:** observed facility-to-facility corridors

After placeholder removal:

- **1,545 facilities**
- **2,349 corridors**
- **74 connected components**
- **1,168 facilities in the giant component**
- **377 facilities across 73 island components**

A temporal betweenness-stability audit produced a **Spearman correlation of 0.7065**, indicating moderately stable but not perfectly fixed network rankings.

### Graph-derived features

The graph phase adds:

- Betweenness centrality
- PageRank
- In-degree
- Out-degree
- Clustering coefficient
- Source/destination graph metrics

Flat graph features improved the single-model baseline from:

**55.35 → 54.79 minutes MAE**

and:

**50.87% → 51.60% within-15%**

This was the first graph experiment to pass the explicit evaluation gate.

---

## 🧠 GraphSAGE

GraphSAGE embeddings are introduced to capture **multi-hop network structure** beyond flat centrality metrics.

A single-seed GraphSAGE result initially looked promising, but multi-seed testing showed that the standalone improvement was within seed variance.

The stronger result came from incorporating GraphSAGE into the full four-model ensemble and averaging predictions across seeds.

### Final seed-stabilized ensemble

| Seed | MAE | Within-15% |
|---|---:|---:|
| 42 | 52.63 min | 53.05% |
| 123 | 53.02 min | 52.65% |
| 999 | 53.04 min | 52.76% |
| **Final seed-averaged prediction** | **52.72 min** | **52.85%** |

Compared with the optimized pre-graph blend:

- **MAE improvement:** 54.17 → 52.72 minutes
- **Within-15% improvement:** 51.76% → 52.85%

The final 52.72-minute result is the project's **verified production reference** used for downstream deliverables.

---

## 📏 ETA / SLA Calibration

Point predictions alone are not sufficient for operational ETAs, so the project also builds calibrated prediction windows.

Several approaches were tested, including:

- Symmetric quantile calibration
- Asymmetric conformal-style padding
- Leakage-safe chronological calibration

The strongest final calibration achieved:

**81.3% out-of-sample coverage** against an **80% target**

with a:

**136-minute median ETA window**

The notebook explicitly rejects a same-data "80.0%" calibration result because self-evaluating on the same residuals makes the target coverage almost guaranteed and therefore misleading.

---

## 🔍 Tier-Level Accuracy

The final seed-averaged prediction was re-audited across the fallback hierarchy.

| Tier | MAE | Within-15% |
|---|---:|---:|
| Tier 0 | 46.93 min | 54.56% |
| Tier 1 | 123.24 min | 27.05% |
| Tier 2 | 148.37 min | 24.00% |
| Tier 3 | 552.87 min | 7.69% |

The results show that **historical coverage is a major determinant of ETA quality**. The overwhelming majority of legs are well-covered by corridor history, while cold-start legs remain substantially harder.

---

## 🏭 Network Bottleneck & SLA Analysis

The graph analysis is also used to identify structurally important facilities and chronically delayed corridors.

### Chronic delay

Two definitions are examined:

1. A literal rule based on **median delay ratio > 1.20**
2. A relative rule using the **worst 10% of delay-ratio corridors within route/distance groups**

The literal rule is too broad:

**95.9% of corridors** are classified as chronically delayed.

The relative rule is more discriminating:

**141 corridors** fall into the worst-decile group.

This distinction prevents a binary flag from becoming so broad that it loses operational usefulness.

### Hub SLA concentration

After correcting the denominator definition, the analysis finds:

- **3,201 true SLA breaches** in the phantom-free test data
- The top three hubs account for **9.2%** of the true breach denominator

Selected top hubs in the final audit include:

- Bhiwandi_Mankoli_HB
- Bangalore_Nelmngla_H
- Bengaluru_Bomsndra_HB
- Bengaluru_KGAirprt_HB
- Mumbai Hub

A revenue-at-risk illustration is expressed as:

> **₹146,500 = 293 breaches × ₹500/breach**

with the explicit caveat that ₹500 is only a placeholder assumption pending real unit economics.

---

## 🚚 FTL vs. Carting Analysis

The project evaluates whether route characteristics support a shift from Carting to FTL.

Short and Medium distance brackets have sufficient sample sizes for comparison.

Long and Ultra-Long Carting rows have **0 observed trips**, so the notebook explicitly marks those comparisons as **low sample / definitionally unavailable**, rather than manufacturing a recommendation from nonexistent data.

This is an example of the project's broader principle: **do not turn missing evidence into false certainty**.

---

## 🔀 Parallel Route Analysis

The final Task 5 work investigates where additional route alternatives could improve resilience.

### Network fragility

Out of **2,349 corridors**:

**1,888 corridors (80.4%) are served by exactly one schedule.**

This indicates substantial schedule-level dependency across the network.

A strict first-pass filter produced only:

- **2 qualifying parallel-route candidates**
- **66 trips**
- **0.4% of network traffic**

The analysis therefore evolves from a simple corridor-only intervention rule into a more useful **root-cause framework**.

### Convergence clusters

The final analysis identifies:

- **249 candidate corridors**
- **38 convergence-cluster facilities**
- **113 of the 249 candidates (45%)** absorbed by these facility clusters

Examples of high-impact convergence locations include:

- Hyderabad_Shamshbd_H
- Bangalore_Nelmngla_H
- Muzaffrpur_Bbganj_I

The analysis distinguishes between:

**Facility problem → fix the node**

and

**Corridor problem → consider a direct/parallel link**

This avoids recommending new routes when the real bottleneck is a common downstream facility.

---

## 🧪 Model Experiments That Did Not Beat the Final Baseline

A major strength of the notebook is that unsuccessful ideas are retained and measured rather than hidden.

### Expanded GraphSAGE node features

Adding facility tier, PIN prefix, dwell history, and shipment volume to GraphSAGE inputs produced:

**52.88 min / 52.78%**

vs. the locked:

**52.72 min / 52.85%**

**Verdict: FAIL**

### Target decomposition

Splitting the target into driving residual + dwell repeatedly underperformed the unified model.

Final post-graph decomposition:

**56.94 min / 50.92%**

**Verdict: FAIL**

### HistGradientBoosting re-tuning

Optuna found better internal CV parameters, but the real three-seed test gate changed only marginally:

**52.76 min / 52.89%**

**Verdict: FAIL / statistically indistinguishable**

### Trip-position + volatility features

The strongest candidate combination was Group A+B, but full confirmation produced:

**52.93 min / 52.66%**

**Verdict: FAIL**

### Acc15-optimized blend

In-sample optimization improved Acc15, but the out-of-sample gain collapsed to roughly noise level.

**Verdict: WEAK**

### GraphSAGE hyperparameter search

Validation MAE improved substantially, but the tuned configuration performed worse on the held-out test gate:

**52.87 min / 52.65%**

**Verdict: FAIL**

### Tier-conditioned blend weights

Tier-specific weighting produced only a tiny improvement relative to seed noise:

**52.75 min / 52.89%**

**Verdict: FAIL**

These failed experiments are valuable because they establish the practical performance ceiling and prevent repeated experimentation with ideas that have already been rigorously rejected.

---

## 🧪 Leakage & Integrity Checks

The project contains explicit verification checks for the graph pipeline.

The final audit confirmed:

- **2,349 / 2,349 graph edges** trace to valid training data
- **0 test rows** appear in the shockwave completion log
- **94 test-only facilities** are absent from the training graph
- `src_dwell_lkp` is built from training-only data

The test set is also treated as a chronological holdout rather than a random split.

> **Important caveat:** some blend-weight optimization steps in the notebook are fitted directly against the test set. The notebook explicitly flags this as a source of mild optimistic bias. The final metrics should therefore be treated as **project evaluation metrics, not an unbiased production generalization estimate**.

---

## 📁 Notebook Structure

The notebook contains approximately:

- **99 total cells**
- **67 code cells**
- **66 logical code-cell stages documented**

Major phases:

```text
1. Environment setup & raw-data loading
2. Leg reconstruction & cleaning
3. Feature engineering
4. Temporal split & historical encoding
5. Baseline ETA modeling
6. Expanded feature modeling
7. Blend optimization & calibration
8. Two-stage / alternative model experiments
9. Graph hygiene & centrality
10. Shockwave / causal experiments
11. Target decomposition
12. GraphSAGE
13. Multi-seed verification
14. SLA / bottleneck deliverables
15. FTL vs. Carting analysis
16. Parallel-route & convergence analysis
17. Final bounded improvement experiments
```

---

## 🛠️ Main Python Libraries

The notebook uses:

```text
pandas
numpy
scikit-learn
catboost
scipy
optuna
networkx
torch
torch-geometric
matplotlib
```

---

## ▶️ Running the Project

The notebook was developed in a Kaggle environment where the raw dataset is mounted under:

```text
/kaggle/input/datasets/ishanyash/delivery-data/delivery_data.csv
```

### Kaggle

1. Open the notebook in Kaggle.
2. Attach the delivery-data dataset.
3. Run the cells in order.

The notebook performs preprocessing, model training, graph construction, evaluation, and downstream operational analysis.

### Local environment

A local run requires the same data file and the Python dependencies listed above. The Kaggle-specific input path should be replaced with the local dataset path.

---

## 📌 Key Takeaways

### 1. Historical corridor intelligence matters most

With **94.2% of test legs in Tier 0**, corridor-level history provides strong predictive signal and explains why cold-start legs are much harder.

### 2. Simple graph structure adds value

Flat network centrality features produced a meaningful improvement before neural graph embeddings were introduced.

### 3. GraphSAGE is useful as part of an ensemble

The standalone GraphSAGE improvement was unstable across random seeds, but graph-enhanced **multi-model seed averaging** produced the first robust graph-phase improvement.

### 4. More complexity does not automatically improve ETA accuracy

Richer GraphSAGE inputs, decomposition, re-tuning, new feature groups, and tier-specific weights all failed to beat the verified baseline.

### 5. Operational interventions should match the root cause

Some delays are better addressed at facilities, while others justify corridor-level routing changes. The final analysis explicitly separates those cases.

---

## 🏁 Final Result

**Verified ETA model**

> **52.72 minutes MAE**  
> **52.85% within-15% accuracy**

**Verified ETA window**

> **81.3% out-of-sample coverage**  
> **136-minute median window**

**Network**

> **1,545 facilities**  
> **2,349 corridors**

**Parallel-route analysis**

> **80.4% of corridors served by exactly one schedule**

The project therefore combines **predictive modeling + network analytics + business intervention design** into one end-to-end delivery intelligence workflow.

---

## 👤 Author

**Delhivery ETA & Network Intelligence Project**

Built as an end-to-end analytics project combining **Python, machine learning, graph analytics, ETA calibration, and operational strategy**.
