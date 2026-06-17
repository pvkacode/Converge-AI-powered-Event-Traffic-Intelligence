# Converge- Day 1 Pipeline (ASTraM Bengaluru Traffic Disruption Intelligence)

**Framing:** From reactive patrol logs to a predictive, self-improving disruption-intelligence system for Bengaluru traffic.

This repository implements the **Day 1 foundation** plus **integrated advanced models** in the same Layer 1 and Layer 2 scripts: data cleaning with trust score, survival analysis (KM/Cox + frailty/AFT/RSF/RMST/GMM), and spatial intelligence (Gi* + severity/network/Hawkes/OBI).

## Problem statement

Political rallies, festivals, sports events, construction, and sudden gatherings create localized traffic breakdowns in Bengaluru. Today:

- **Event impact is not quantified in advance** — dispatch relies on experience, not data
- **Resource deployment is experience-driven** — no baseline for where to send manpower or barricades
- **No post-event learning system** — similar events are handled from scratch each time

**Goal:** Use historical and real-time ASTraM data to forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans.

Layers 1 and 2 are the **measurement layer** — they quantify *how long* and *where* disruption concentrates. Layer 3+ turn those numbers into deployment plans and learning loops.

## How Layers 1 & 2 address the problem

| Pain point | Layer | What it delivers |
|------------|-------|------------------|
| Impact not quantified | **Layer 1** | P50/P80/P95 duration quantiles by `(event_cause × corridor)` — e.g. breakdowns on Mysore Road: median ~39 min, P80 ~74 min |
| Deployment is guesswork (where) | **Layer 2** | 80 statistically significant junction hotspots (Gi*), not just a busy map — Silk Board, Goruguntepalya, Mysore Rd toll gate |
| Sparse planned events (191 true events / 8,173 rows) | **Layer 1** | `lookup_expected_duration()` returns `None` for sparse causes (e.g. protest) → triggers Layer 4 case retrieval |

### Layer 1 — time dimension (“how long will this corridor stay degraded?”)

Kaplan-Meier survival curves answer operational questions directly:

- **“How long should we block this lane?”** → P80 duration for that cause × corridor
- **“When can we redeploy this team?”** → P80 = time until 80% of similar incidents cleared
- **“Is this estimate trustworthy?”** → weighted by `trust_score`; sparse strata fall back or return `None`

Layer 1 feeds Layer 3 resource sizing:

```
manpower_needed   ≈ f(impact_score, expected_duration, requires_road_closure)
barricade_window  ≈ [start, start + P80_duration]
diversion_window  ≈ same window
```

### Layer 2 — spatial dimension (“where should we pre-position resources?”)

Getis-Ord Gi* tests whether a junction is **anomalously hot**, not merely busy. Trust-weighted intensity (`sum(trust_score)`) prevents low-quality rows from inflating hotspots.

Layer 2 feeds Layer 3 placement:

```
pre_position_manpower(junction) ∝ Gi* significance × weighted_intensity
priority_barricade_points       = hotspots ∩ planned_event_route
diversion_candidates            = corridors adjacent to significant hotspots
```

### Worked example — breakdown at a known hotspot

| Input | Layer | Output |
|-------|-------|--------|
| `vehicle_breakdown`, Mysore Road | L1 | P50 ~39 min, P80 ~74 min |
| Silk Board Junction | L2 | Significant hotspot (p_sim = 0.006) |
| **Layer 3 (next)** | — | Pre-position tow + patrol; plan ~40–80 min disruption window; prioritize alternate ORR arm |

### What Layers 1 & 2 do not do yet

| Problem piece | Status |
|---------------|--------|
| Forecast duration / spatial risk | **Done** (L1 + L2) |
| Recommend specific manpower counts | Layer 3 (planned) |
| Barricade coordinates / diversion routes | Layer 3 (planned) |
| Post-event learning loop | Layer 6 (planned) |

## Dataset

- **Source:** ASTraM / Bengaluru Traffic Police operational log (~8,170 incidents, Nov 2023 – Apr 2024)
- **Raw file:** `data/events_raw.csv`
- **Cleaned output:** `data/events_clean.parquet` (+ `.csv`)

## What is `trust_score`?

A single row-level confidence weight in **[0, 1]**, computed as a **noisy-OR** over four independent evidence flags:

```
trust_i = ∏_k (1 − w_k · flag_k,i)
```

| Flag | Weight | Meaning |
|------|--------|---------|
| `duration_anomaly` | 0.30 | Stratified MAD outlier within (cause × corridor) |
| NOT `geo_valid` | 0.40 | Missing or placeholder (0,0) coordinates |
| MNAR censored | 0.30 | Censored + logistic P(missing) > 0.7 |
| `iso_flagged` | 0.30 | Bottom 5% Isolation Forest anomaly score |

**Why it replaces global truncation:** A single 1440-minute cutoff cannot distinguish "12-hour ORR East 2 construction is normal" from "12-hour vehicle breakdown is bad data." Stratified MAD (`|modified_z| > 3.5`) flags anomalies *within context*.

Low-trust rows are **not deleted** — they contribute proportionally less to Kaplan-Meier fits (`weights=`) and Gi* intensity (`sum(trust_score)`).

## Missingness test

The pipeline fits a logistic regression predicting `P(no end timestamp | corridor, priority, cause, closure, hour)` and runs a likelihood-ratio test against an intercept-only model. Results are saved to:

```
outputs/missingness_test.txt
```

**Key finding:** ~3,500+ rows marked `status=closed` have **no** end timestamp at all — status and timestamp fields are inconsistently maintained in the source system. This is reported explicitly in the pipeline summary.

## Layer 1 — Survival analysis

- **Kaplan-Meier** stratified by `(event_cause, corridor)` with `MIN_GROUP_SIZE=15`
- Uses **trust-weighted** fits on rows with observed `duration_min` only (~3,500 resolved incidents)
- Censored rows (no end timestamp) are excluded from KM quantiles but down-weighted via `trust_score` in the cleaned table; including them with administrative censoring would inflate quantiles to study-horizon (~160 days) and destroy operational readability
- Fallback table by `event_cause` only for sparse strata
- **Cox PH** on priority, closure flag, cyclical time, top-8 corridor dummies (concordance ~0.56 — weak covariate signal; cause×corridor matters more)
- Outputs: `outputs/layer1_survival_quantiles.csv`, `layer1_survival_fallback.csv`, `layer1_cox_summary.txt`

Public API: `lookup_expected_duration(cause, corridor, km_table, km_fallback, quantile="p50")`

## Layer 2 — Getis-Ord Gi* hotspots

- Trust-weighted junction intensity (not raw counts)
- KNN spatial weights (k=6), permutation `p_sim < 0.05` as **primary** significance test
- Output: `outputs/layer2_hotspots.csv`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (in order)

```bash
python src/data_pipeline.py
python src/layer1_survival.py    # baseline KM/Cox + advanced survival models
python src/layer2_hotspots.py    # baseline Gi* + advanced hotspot intelligence
python src/layer1_research_upgrades.py  # frailty LRT + stacked ensemble (additive)
python src/layer2_research_upgrades.py  # MSHI + Monte Carlo OBI stability (additive)
python src/layer3_resource_optimization.py
python src/layer4_event_intelligence.py
python src/layer3_corridor_fragility.py    # additive: Hawkes corridor fragility
python src/layer4_planned_event_retrieval.py  # additive: prototype retrieval
python src/layer3_methodology_upgrades.py   # PCA stability + log fragility (additive)
python src/layer4_methodology_upgrades.py   # leakage-free retrieval + K-Medoids (additive)
python src/layer4_operational_upgrades.py   # evidence tiers + quantiles + L3 fallback (final L4)
python src/frontend_exports.py              # dashboard-ready copies → outputs/frontend/
python src/validate_consistency.py
```

Each layer script runs **baseline first, then advanced**, writing all outputs to `outputs/layer1_*`, `outputs/layer2_*`, `outputs/layer3_*`, and `outputs/layer4_*`.

## Layer 1 outputs (`layer1_survival.py`)

| Section | Models | Output files |
|---------|--------|--------------|
| **Baseline** | Kaplan-Meier (cause×corridor), Cox PH | `layer1_survival_quantiles.csv`, `layer1_survival_fallback.csv`, `layer1_cox_summary.txt` |
| **Advanced** | Frailty, AFT, RSF, RMST, GMM | `layer1_frailty_scores.csv`, `layer1_duration_predictions.csv`, `layer1_survival_risk_scores.csv`, `layer1_rmst_summary.csv`, `layer1_incident_archetypes.csv`, … |
| **Research upgrades** | Frailty LRT, stacked ensemble | `layer1_frailty_validation.csv`, `layer1_frailty_interpretation.txt`, `layer1_stacked_survival_predictions.csv`, `layer1_stacked_survival_metrics.csv` |

## Layer 2 outputs (`layer2_hotspots.py`)

| Section | Models | Output files |
|---------|--------|--------------|
| **Baseline** | Trust-weighted Getis-Ord Gi* | `layer2_hotspots.csv` |
| **Advanced** | Severity, spatiotemporal Gi*, network Gi*, Hawkes, persistence, future risk, OBI | `layer2_severity_hotspots.csv`, `layer2_network_hotspots.csv`, `layer2_operational_burden_index.csv`, … |
| **Research upgrades** | Multi-scale Gi* (MSHI), Monte Carlo OBI stability | `layer2_multiscale_hotspots.csv`, `layer2_obi_stability.csv`, `layer2_obi_stable_top25.csv` |

## Advanced models (integrated into Layer 1 & 2)

### Why advanced models beat baseline KM + Gi*

| Limitation of baseline | Advanced remedy |
|------------------------|-----------------|
| KM ignores covariate interactions | Random Survival Forest (C-index ~0.70 vs Cox ~0.56) |
| Cox gives hazard ratios, not minutes | AFT models predict median/P90 duration directly |
| Corridors differ in hidden ops capacity | Frailty / clearance multipliers by corridor |
| Median hides tail risk for planners | RMST(τ) = expected occupation time up to τ |
| One-size duration bucket | GMM latent archetypes (quick / moderate / severe) |
| Gi* counts incidents, not burden | Severity = Trust × Duration × Priority |
| Euclidean KNN ignores road topology | Network-constrained Gi* on corridor graph |
| Static map misses rush-hour patterns | Spatiotemporal Gi* by hour × day-of-week |
| No cascade awareness | Hawkes self-exciting intensity per junction |
| Hotspot today ≠ hotspot always | Weekly persistence index (transient / chronic) |
| Reactive not predictive | Graph ML future hotspot risk score |
| Many metrics, one decision | **Operational Burden Index (OBI)** |

### Layer 1 formulas (all in `layer1_survival.py`)

| Model | Formula | Output |
|-------|---------|--------|
| **Kaplan-Meier** | \(\hat S(t) = \prod(1 - d_i/n_i)\), trust-weighted | `layer1_survival_quantiles.csv` |
| **Cox PH** | \(h(t|X) = h_0(t) e^{\beta X}\) | `layer1_cox_summary.txt` |
| **Frailty** | Corridor clearance multiplier = global median / corridor median | `layer1_frailty_scores.csv` |
| **AFT** | Weibull + LogNormal; lowest AIC wins | `layer1_duration_predictions.csv` |
| **RSF** | Random Survival Forest (`scikit-survival`) | `layer1_survival_risk_scores.csv` |
| **RMST** | \(\int_0^\tau S(t)\,dt\) for τ ∈ {60,180,360,720} min | `layer1_rmst_summary.csv` |
| **GMM** | Latent duration archetypes (BIC selects K) | `layer1_incident_archetypes.csv` |

### Layer 2 formulas (all in `layer2_hotspots.py`)

| Model | Formula | Output |
|-------|---------|--------|
| **Baseline Gi*** | \(x_j = \sum \text{Trust}_i\), Euclidean KNN | `layer2_hotspots.csv` |
| **Severity** | \(x_j = \sum \text{Trust}_i \times \text{Duration}_i \times \text{Priority}_i\) | `layer2_severity_hotspots.csv` |
| **Spatiotemporal Gi*** | Gi* per hour × dow slice | `layer2_spatiotemporal_hotspots.csv` |
| **Network Gi*** | Corridor-graph adjacency (≤2 hops) | `layer2_network_hotspots.csv` |
| **Hawkes** | \(\lambda(t) = \mu + \sum \alpha e^{-\beta(t-t_i)}\) | `layer2_hawkes_cascade_risk.csv` |
| **Persistence** | HPI = significant weeks / total weeks | `layer2_hotspot_persistence.csv` |
| **Future risk** | XGBoost/GBC on graph features | `layer2_future_hotspot_risk.csv` |
| **OBI** | Weighted composite of above | `layer2_operational_burden_index.csv` |

### Research-grade upgrades (additive modules)

Run **after** the main layer scripts. These modules do not retrain RSF, SHAP, HDBSCAN, or other base models — they read existing outputs.

#### Layer 1 — `layer1_research_upgrades.py`

| Upgrade | Formula | Output | Interpretation |
|---------|---------|--------|----------------|
| **Frailty LRT** | \(LR = 2(\ell_{\text{frailty}} - \ell_{\text{Cox nested}})\), df = 1 | `layer1_frailty_validation.csv`, `layer1_frailty_interpretation.txt` | Tests whether shared gamma frailty (\(u_j \sim \text{Gamma}(\theta,\theta)\)) improves fit over nested Cox (\(\theta \to \infty\)). `frailty_supported=True` when \(p < 0.05\). |
| **Stacked ensemble** | \(\text{Risk}_{\text{stack}} = \sum_k w_k \cdot \text{risk}_k\) via Elastic Net CV on standardized Cox / frailty / AFT / RSF scores | `layer1_stacked_survival_predictions.csv`, `layer1_stacked_survival_metrics.csv`, `layer1_stacked_interpretation.txt` | RSF remained best; stacking did not improve C-index (overlapping signal). Honest negative result documented. |
| **RSF reliability** | ECE \(= \frac{1}{N}\sum_k | \text{obs}_k - \text{pred}_k | \) by risk-score decile at τ=180 min | `layer1_rsf_reliability.csv`, `layer1_rsf_calibration_summary.csv` | Calibration more informative than another model; ECE < 0.10 is reasonable. |

#### Layer 2 — `layer2_research_upgrades.py`

| Upgrade | Formula | Output | Interpretation |
|---------|---------|--------|----------------|
| **MSHI → SPS / NHI** | \(\text{SPS}_i = \frac{1}{|H|}\sum_h G_i^*(h)\); \(\text{NHI}_i = \frac{1}{|H|}\sum_h \text{Percentile}(G_i^*(h))\), \(H=\{1,2,3,5\}\) | `layer2_multiscale_hotspots.csv` | Replaces collapsed binary MSHI; NHI ranks junctions when significance tests saturate on dense corridor graphs. |
| **Hawkes validation** | Branching ratio \(R = \alpha/\beta\); weak \(<0.3\), moderate \(0.3–0.7\), strong \(>0.7\) | `layer2_hawkes_validation.csv` | Operational cascade intensity per junction (reads existing Hawkes fit). |
| **OBI stability** | \(x' = x + \epsilon\), \(\epsilon \sim N(0, 0.05)\); 1000 Monte Carlo OBI recomputations | `layer2_obi_stability.csv`, `layer2_obi_stable_top25.csv` | `prob_top25` = fraction of simulations in top 25; high values = robust priority junctions under metric noise. |

### Methodology fixes (pre-frontend — additive, no model retraining)

Run **after** main Layer 3/4 scripts. These address judge-facing weaknesses without retraining Layer 1/2 models.

#### Layer 3 — `layer3_methodology_upgrades.py`

| Fix | Formula | Output | Rationale |
|-----|---------|--------|-----------|
| **PCA loading stability** | Bootstrap \(B=500\) junction resamples; 95% CI on PC1 loadings | `layer3_pca_loading_stability.csv`, `layer3_pca_stability_summary.txt` | Defends DIS = PC1: which drivers (OBI, cascade, etc.) are stable. |
| **Log fragility** | \(\text{fragility\_log} = \log(1 + (\lambda-\mu)/(\mu+\varepsilon))\), \(\varepsilon=0.01\) | `layer3_corridor_fragility.csv` (adds `fragility_raw`, `fragility_log`) | Bounded ranking when \(\mu \to 0\); Hawkes fit unchanged. |

#### Layer 4 — `layer4_methodology_upgrades.py`

| Fix | Formula | Output | Rationale |
|-----|---------|--------|-----------|
| **Leakage-free retrieval** | Gower \(d_G(q,p)\) uses only pre-event features: cause, corridor, closure, hour, dow, priority, month | `layer4_retrieval_validation.csv` | Duration/trust/OBI never enter similarity — only outcomes after retrieval. |
| **K-Medoids prototypes** | Medoid \(= \arg\min_{x_i}\sum_j d_G(x_i,x_j)\) on Gower matrix | `layer4_planned_event_prototypes.csv` | Real event prototypes; mixed categorical + numeric geometry. |
| **Calibrated confidence** | \(\text{Conf} = \frac{n_{eff}}{n_{eff}+2}\cdot\bar s\cdot\max(s)\); abstain if Conf \(<0.4\) or \(\max s<0.3\) or \(n_{eff}<3\) | `layer4_retrieval_diagnostics.csv` | Principled abstention when evidence is weak (~50% on LOO evaluation). |

#### Layer 4 — `layer4_operational_upgrades.py` (final pre-frontend)

| Feature | Behavior | Output |
|---------|----------|--------|
| **Evidence bands** | HIGH (Conf≥0.70), MEDIUM (0.40–0.70), LOW (<0.40) | `confidence_band` in retrieval + diagnostics |
| **Uncertainty** | Weighted \(Q_{50}, Q_{80}, Q_{95}\) for duration + impact from retrieved analogs | `pred_duration_p*`, `pred_impact_p*` |
| **Fallback** | LOW → Layer 3 DIS/ODS/manpower; MEDIUM → HYBRID; HIGH → RETRIEVAL durations | `recommendation_source` |
| **Analytics** | Band counts, confidence histogram | `layer4_retrieval_quality_summary.csv`, `layer4_confidence_distribution.csv` |

#### Frontend — `frontend_exports.py`

Copies canonical outputs to `outputs/frontend/` — **the only path the dashboard should read**:

`duration_lookup.csv`, `risk_scores.csv`, `hotspot_rankings.csv`, `operational_burden.csv`, `top25_locations.csv`, `corridor_fragility.csv`, `planned_event_recommendations.csv`

### Computational complexity (approximate, n=8k incidents)

| Module | Complexity | Runtime (local) |
|--------|------------|-----------------|
| Frailty / AFT | O(n·p) | < 5 s |
| RSF (500 trees) | O(n log n · trees) | ~3 min |
| RMST / GMM | O(n·strata) | < 5 s |
| Spatiotemporal Gi* | O(hours × junctions × perm) | ~1 min |
| Network Gi* | O(j²) shortest paths | ~30 s |
| Hawkes (40 junctions) | O(n²) per junction MLE | ~30 s |
| OBI composite | O(junctions) | instant |

### Operational use cases

- **Pre-position tow trucks:** top OBI junctions + high Hawkes cascade_risk
- **Staff shift planning:** spatiotemporal Gi* morning vs evening hotspots
- **Barricade duration:** RMST(180) + AFT predicted_p90 per cause×corridor
- **VIP route planning:** avoid chronic persistence_class junctions
- **Dashboard KPI:** OBI ranked list replaces raw incident counts

## Design notes for judges / concept note

1. **Censoring is real:** 4,500+ rows lack end timestamps; naive averages are biased.
2. **Data quality is a first-class finding:** closed-without-timestamp is systematic, not random.
3. **Cox concordance ~0.56:** priority/time-of-day weakly predict duration; cause×corridor matter more (KM captures this).
4. **Gi* z > 1.96 vs p_sim:** asymptotic cutoff failed on this sample; permutation test is documented and used.
5. **Silk Board, Mekhri Circle, etc.** emerge as significant hotspots — real-world sanity check.

## Layer 3 — Resource Optimization Engine (v2)

Implemented in `src/layer3_resource_optimization.py`. Consumes Layer 1 and Layer 2 outputs exclusively (no re-training).

### Learned DIS via PCA

DIS is no longer a fixed-weight sum. A `sklearn.PCA` is fitted on the standardised 5-component matrix `[OBI, cascade_risk, future_risk, RMST_mean, persistence]` for all 294 junctions. PC1 (55.7 % of variance) is used as the DIS axis; its sign is corrected so that DIS correlates positively with OBI.

```
X_scaled[294×5] = StandardScaler().fit_transform([OBI, cascade_risk, future_risk, RMST, persistence])
DIS_raw          = X_scaled @ PC1_loadings          # sign-checked vs OBI
DIS              = 100 · (DIS_raw − min) / (max − min)
```

The fitted PCA is saved to `outputs/layer3_pca_model.pkl` for reproducibility.

Risk tiers: **Low** (0–30) · **Moderate** (30–60) · **High** (60–80) · **Critical** (80–100)

### Operational Demand Score (ODS)

ODS is a multiplicative demand signal that drives continuous resource sizing:

```
ODS = DIS × DurationFactor × ClosureFactor × CascadeFactor

DurationFactor = 1 + P80_capped / 120          # P80 capped at 360 min
ClosureFactor  = 1.5 if requires_road_closure else 1.0
CascadeFactor  = 1 + R                          # R = Hawkes branching_ratio
```

Resource quantities derived continuously from ODS:

```
officers    = ceil(ODS / 30), capped at 25
barricades  = ceil(ODS / 20), capped at 40
tow_units   = ceil(ODS / 80), capped at 5
supervisors = ceil(officers / 6)
qru_units   = 1 if DIS ≥ 70 else 0
```

### Linear Programming resource allocation

`scipy.optimize.linprog` (HiGHS) maximises total DIS served subject to city-wide budget constraints across top-50 DIS≥30 junctions:

```
maximise   Σ DIS_i · x_i
subject to Σ officers_i · x_i    ≤ 120
           Σ tow_i · x_i          ≤ 15
           Σ barricades_i · x_i  ≤ 100
           Σ supervisors_i · x_i ≤ 20
           0 ≤ x_i ≤ 1
```

`allocation_fraction` = LP solution. Junctions outside the LP budget keep `x_i = 1` (recommended = allocated).

### Dijkstra diversion routing

A `networkx.DiGraph` is built from all events-clean corridor-junction pairs. Edge weight:

```
w(u,v) = 0.5 · (node_cost(u) + node_cost(v))
node_cost(j) = 0.4·norm(OBI_j) + 0.3·norm(FutureRisk_j) + 0.2·norm(Hawkes_j) + 0.1·norm(RMST_j) + 0.01
```

For each of the top-30 DIS junctions, the blocked junction is removed, and `nx.single_source_dijkstra` finds the 3 lowest-cost diversion targets. Zone-based fallback is used only when the graph is disconnected.

### Resource efficiency simulation

```
clearance_predicted = base_clearance × (1 − reduction)
reduction           = (1 − exp(−0.08 · N_officers)) × 0.40    # max 40% improvement
```

Simulated for N_officers multipliers [1.0, 1.1, 1.2, 1.3, 1.5, 2.0] across the top-20 junctions.

### Barricading strategy (ODS-driven)

| Risk level | Strategy | Barricades | Closure type |
|------------|----------|------------|--------------|
| Low | none | 0 | none |
| Moderate | partial_closure | ceil(ODS/20) | partial_lane |
| High | full_barricading | ceil(ODS/20) | full_road |
| Critical | emergency_closure | ceil(ODS/20) | full_road |

### Public API

```python
generate_deployment_blueprint(junction_name: str, event_type: str | None = None) -> dict
```

Returns: `{dis_score, risk_level, ods_score, pca_loadings, allocated_officers, allocated_supervisors, allocated_tow, allocated_barricades, qru_units, patrol_vehicles, barricade_strategy, closure_type, diversion_routes, tow_unit_assigned, efficiency_note}`.

### Layer 3 outputs

| File | Rows | Description |
|------|------|-------------|
| `layer3_disruption_impact_scores.csv` | 294 | PCA-learned DIS + risk level + PC1 loadings per junction |
| `layer3_pca_model.pkl` | — | Fitted StandardScaler + PCA (5 components) |
| `layer3_pca_explained_variance.csv` | 5 | Explained variance ratio per PC |
| `layer3_manpower_recommendations.csv` | 294 | ODS-derived officers, tow, barricades + LP-allocated quantities |
| `layer3_lp_resource_allocation.csv` | 50 | LP allocation fractions and quantities for top-50 DIS≥30 junctions |
| `layer3_barricading_plan.csv` | 294 | ODS-driven strategy, barricades, closure type, teams |
| `layer3_diversion_recommendations.csv` | 90 | Dijkstra Route A/B/C for top-30 DIS junctions |
| `layer3_tow_placement.csv` | ~98 | Tow unit IDs, assigned junctions, shift |
| `layer3_resource_efficiency_simulation.csv` | 120 | Clearance vs officer count for top-20 junctions |
| `layer3_efficiency_scenarios.json` | — | City-wide clearance improvement at each multiplier |
| `layer3_deployment_blueprints.json` | 5 | Full blueprint dicts for top-5 junctions |
| `layer3_full_dashboard.csv` | 294 | Wide table joining all Layer 3 metrics + LP status |

---

## Layer 4 — Event Intelligence Engine (v2)

Implemented in `src/layer4_event_intelligence.py`. Focuses on the 191 true planned events (`is_true_planned_event == True`).

### Gower similarity (mixed-type, 191 × 8,173)

Categorical features use exact-match similarity (0 or 1). Continuous features use:

```
sim_cont(xi, xj) = 1 − |xi − xj| / range
Gower(xi, xj)    = (Σ cat_sims + Σ cont_sims) / n_features
```

Features: `event_cause` (cat), `corridor` (cat), `requires_road_closure` (cat), `hour_local` (cont), `dow_local` (cont), `duration_min_filled` (cont), `priority_code` (cont), `trust_score` (cont), `month` (cont).

### Hybrid similarity

Operationally-weighted categorical + continuous blend:

| Feature | Weight |
|---------|--------|
| cause | 0.35 |
| corridor | 0.25 |
| time (hour + dow) | 0.15 |
| duration | 0.10 |
| closure | 0.10 |
| priority | 0.05 |

Final blended similarity:

```
SIM = 0.6 × Hybrid + 0.4 × Gower
```

### Retrieval confidence tiers

Mean of top-k blended scores per planned event:

| Tier | Score |
|------|-------|
| Strong | ≥ 0.80 |
| Moderate | 0.60–0.80 |
| Weak | < 0.60 |

### Institutional Memory Score (IMS)

```
n_meaningful = |{j : SIM(query, j) ≥ 0.60}|
mean_sim     = mean of SIM scores above threshold
IMS          = log(1 + n_meaningful) × mean_sim
```

Higher IMS → more reliable historical evidence → stronger weight on history vs Layer 3 rules.

### Evidence-based resource recommendation

```
evidence_weight  = min(1, IMS / max_IMS)
blended_officers = evidence_weight × historical_est + (1 − evidence_weight) × l3_rule_est
```

### Impact forecast

```
impact_score = (0.35·norm(L3_DIS) + 0.25·norm(pred_duration)
               + 0.20·closure_prob + 0.20·norm(IMS)) × trust_score × 100
```

### Public API

```python
# Retrieve k most similar historical events for a planned event
retrieval_df  # layer4_retrieval_results.csv
# Evidence weight for resource blending
evid_df       # layer4_evidence_based_recommendations.csv
# IMS per planned event
ims_df        # layer4_institutional_memory_scores.csv
```

### Knowledge base structure

`layer4_event_knowledge_base.json` contains (v2 enriched):
- `knowledge_entries[]` — one entry per event; planned events additionally carry `ims_score`, `confidence_tier`, `evidence_weight`, `impact_score`, `impact_tier`, `recommended_officers`
- `corridor_summaries{}` — incident count, mean duration, closure rate, top causes, n_planned
- `cause_summaries{}` — incident count, mean duration, n_planned
- `metadata{}` — similarity method, IMS threshold, k, hybrid weights

### Layer 4 outputs

| File | Description |
|------|-------------|
| `layer4_event_features.csv` | Raw + encoded feature matrix (8,173 rows) |
| `layer4_event_index_metadata.csv` | Row-index → event metadata mapping |
| `layer4_nn_index.pkl` | sklearn NearestNeighbors index (8 features) |
| `layer4_encoders.pkl` | Fitted LabelEncoders + StandardScaler (pickle) |
| `layer4_gower_similarity_sample.csv` | Gower similarity sample (top-5 per first 10 planned events) |
| `layer4_similarity_weights.json` | Hybrid weights + blend alphas |
| `layer4_institutional_memory_scores.csv` | IMS + n_meaningful + confidence tier (191 rows) |
| `layer4_retrieval_results.csv` | Top-5 hybrid-retrieved events per planned event (955 rows) |
| `layer4_event_outcome_predictions.csv` | Predicted duration, closure prob, trust (191 rows) |
| `layer4_evidence_based_recommendations.csv` | IMS-blended officer recommendations (191 rows) |
| `layer4_explainable_recommendations.txt` | Human-readable Gower/Hybrid breakdown for top-20 |
| `layer4_explainable_recommendations.json` | Structured version of the above |
| `layer4_scenario_simulations.json` | 5 what-if scenario results |
| `layer4_event_knowledge_base.json` | Full enriched institutional memory (~2.2 MB) |

---

## Layer 3 — Corridor Fragility (Hierarchical Marked Hawkes)

New additive module (`src/layer3_corridor_fragility.py`). Fits a marked Hawkes process per corridor with zone-level pooling and empirical Bayes shrinkage.

**Model:**
```
λ_c(t) = μ_c + Σ_{t_i<t} α_{z(c)} · m_i · exp(−β_{z(c)} · (t − t_i))

where m_i = trust_i × (1 + 0.5·closure_i) × (priority_i / max_priority)
```

**Zone pooling** (first-token of corridor name): ORR North 1/2 + ORR East 1/2 + ORR West 1 → zone ORR; Bellary Road 1/2 → zone BELLARY; etc.

**Shrinkage** (for sparse corridors with n < 20):
```
θ_c = (n/(n+κ)) · θ̂_c + (κ/(n+κ)) · θ_{z(c)}
```

**Key outputs:**
| Metric | Interpretation |
|--------|---------------|
| Branching ratio R = α/β | < 0.3: baseline-dominated · 0.3–0.7: moderate excitation · ≥ 0.7: cascade-prone |
| current_fragility = λ(t)/μ − 1 | 0: at baseline · > 0: above baseline · > 2: critically elevated |
| fragility_practical | Capped at 100 for dispatch (raw fragility can be unbounded when μ → 0) |
| fragility_reliable | True when μ ≥ 5% of Poisson rate; False flags near-zero-mu artifacts |

**Results (22 corridors):** 21/22 show Hawkes > Poisson (LR test p < 0.05). Median branching ratio = 0.26 — fragility is baseline-driven (historical incident rate), not cascade-driven. ORR corridors show the highest reliable fragility (R ≈ 1.08).

**Outputs:** `layer3_corridor_fragility.csv`, `layer3_fragility_validation.csv`, `layer3_zone_fragility_summary.csv`

---

## Layer 4 — Planned Event Prototype Retrieval

New additive module (`src/layer4_planned_event_retrieval.py`). Trust-weighted Gower similarity retrieval with prototype compression and principled abstention for sparse planned events (n = 191).

**Design decisions:**
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Prototype compression | KMeans → 47 clusters | Prevents degenerate retrieval; one prototype can't dominate |
| Feature weights | IG-shrinkage: w_k = ρ·IG_k/ΣIG + (1−ρ)/P | Adapts to which features predict duration; ρ = 0.95 |
| Similarity | s(q,p) = exp(−d_G/h) · τ_p | Trust-weighted; bandwidth h = 0.5 |
| Confidence | Conf = min(1, n_eff/k₀) · s̄ | ESS-normalised mean similarity |
| Abstention | max_sim < 0.15 or n_eff < 3.0 | Declines to predict when evidence is thin |

**Results (191 planned events):** 0% abstention rate; mean confidence = 0.786; mean n_eff = 4.97; MAE = 40.9 min; 57.6% of predictions within 20 min of actual. Top feature weight: `duration_clean` (57.2%).

**Outputs:** `layer4_planned_event_retrieval.csv`, `layer4_planned_event_prototypes.csv`, `layer4_retrieval_diagnostics.csv`, `layer4_prototype_utilization.csv`, `layer4_retrieval_feature_weights.json`, `layer4_retrieval_encoders.pkl`, `layer4_example_retrievals.json`, `layer4_simulation_demos.json`

---

## End-to-End Decision Pipeline

```
Historical ASTraM Data (8,173 incidents, Nov 2023 – Apr 2024)
        |
        v
Layer 1 — Duration Intelligence
(KM, Cox PH, Frailty, AFT, RSF, RMST, GMM archetypes)
        |
        v
Layer 2 — Spatial Intelligence
(Gi*, OBI, Hawkes self-excitation, Future Risk, Persistence index)
        |
        v
Layer 3 — Resource Optimization (v2)         Layer 3 — Corridor Fragility (Hawkes)
(PCA-learned DIS, LP allocation, Dijkstra)   (marked Hawkes + EB shrinkage + LR test)
        |                                              |
        +──────────────────────┬────────────────────+
                               v
Layer 4 — Event Intelligence (v3)            Layer 4 — Prototype Retrieval
(Gower+Hybrid, IMS, KG, counterfactuals)     (trust-weighted Gower + ESS confidence)
        |
        v
Operational Action Plan
(Deployment Blueprint + Growing Knowledge Base)
```

## Project structure

```
converge/
├── data/
│   ├── events_raw.csv
│   └── events_clean.parquet
├── outputs/
│   ├── missingness_test.txt
│   ├── layer1_survival_quantiles.csv
│   ├── layer1_frailty_scores.csv
│   ├── layer1_rmst_summary.csv
│   ├── layer1_duration_predictions.csv
│   ├── layer1_survival_risk_scores.csv
│   ├── layer1_incident_archetypes.csv
│   ├── layer1_stacked_survival_predictions.csv
│   ├── layer2_hotspots.csv
│   ├── layer2_operational_burden_index.csv
│   ├── layer2_hawkes_cascade_risk.csv
│   ├── layer2_future_hotspot_risk.csv
│   ├── layer2_hotspot_persistence.csv
│   ├── layer2_severity_hotspots.csv
│   ├── layer2_network_hotspots.csv
│   ├── layer2_obi_stability.csv
│   ├── layer3_disruption_impact_scores.csv
│   ├── layer3_manpower_recommendations.csv
│   ├── layer3_barricading_plan.csv
│   ├── layer3_diversion_recommendations.csv
│   ├── layer3_tow_placement.csv
│   ├── layer3_deployment_blueprints.json
│   ├── layer3_full_dashboard.csv
│   ├── layer4_event_features.csv
│   ├── layer4_nn_index.pkl
│   ├── layer4_event_index_metadata.csv
│   ├── layer4_encoders.pkl
│   ├── layer4_retrieval_results.csv
│   ├── layer4_event_outcome_predictions.csv
│   ├── layer4_explainable_recommendations.txt
│   ├── layer4_explainable_recommendations.json
│   ├── layer4_scenario_simulations.json
│   ├── layer4_event_knowledge_base.json
│   ├── layer3_corridor_fragility.csv       # Hawkes fragility per corridor
│   ├── layer3_fragility_validation.csv     # LR test results
│   ├── layer3_zone_fragility_summary.csv   # zone-aggregated fragility
│   ├── layer4_planned_event_retrieval.csv  # prototype retrieval results (191 events)
│   ├── layer4_planned_event_prototypes.csv # 47 KMeans prototypes
│   ├── layer4_retrieval_diagnostics.csv    # per-event confidence + error
│   ├── layer4_prototype_utilization.csv    # prototype usage statistics
│   ├── layer4_retrieval_feature_weights.json # IG-shrinkage feature weights
│   ├── layer4_example_retrievals.json      # 5 annotated examples
│   └── layer4_simulation_demos.json        # 3 demo simulations
├── src/
│   ├── data_pipeline.py
│   ├── layer1_survival.py              # baseline + advanced survival
│   ├── layer2_hotspots.py              # baseline + advanced hotspots
│   ├── layer1_research_upgrades.py     # frailty LRT + stacked ensemble
│   ├── layer2_research_upgrades.py     # MSHI + OBI stability
│   ├── layer3_resource_optimization.py # DIS, manpower, barricades, diversions, tow
│   ├── layer3_corridor_fragility.py    # marked Hawkes, zone pooling, EB shrinkage, LR test
│   ├── layer4_event_intelligence.py    # retrieval, prediction, simulation, knowledge base
│   ├── layer4_planned_event_retrieval.py # prototype retrieval (legacy KMeans)
│   ├── layer4_methodology_upgrades.py    # leakage-free Gower + K-Medoids + confidence
│   ├── layer4_operational_upgrades.py    # evidence tiers, quantiles, L3 fallback
│   ├── layer3_methodology_upgrades.py  # PCA bootstrap + log fragility
│   ├── frontend_exports.py             # dashboard export layer
│   └── validate_consistency.py
├── requirements.txt
├── HANDOFF.md
└── README.md
```

## Next layers (planned)

- Layer 6: Bayesian post-event learning loop (updates priors after each incident closes)
