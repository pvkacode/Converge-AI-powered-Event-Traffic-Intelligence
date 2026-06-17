# Converge — Day 1 Pipeline (ASTraM Bengaluru Traffic Disruption Intelligence)

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
python src/validate_consistency.py
```

Each layer script runs **baseline first, then advanced**, writing all outputs to `outputs/layer1_*` and `outputs/layer2_*`.

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
│   ├── layer2_hotspots.csv
│   ├── layer2_operational_burden_index.csv
│   └── … (all layer1_* and layer2_* outputs)
├── src/
│   ├── data_pipeline.py
│   ├── layer1_survival.py       # baseline + advanced survival
│   ├── layer2_hotspots.py       # baseline + advanced hotspots
│   ├── layer1_research_upgrades.py  # frailty LRT + stacked ensemble
│   ├── layer2_research_upgrades.py  # MSHI + OBI stability
│   └── validate_consistency.py
├── requirements.txt
├── HANDOFF.md
└── README.md
```

## Design notes for judges / concept note

1. **Censoring is real:** 4,500+ rows lack end timestamps; naive averages are biased.
2. **Data quality is a first-class finding:** closed-without-timestamp is systematic, not random.
3. **Cox concordance ~0.56:** priority/time-of-day weakly predict duration; cause×corridor matter more (KM captures this).
4. **Gi* z > 1.96 vs p_sim:** asymptotic cutoff failed on this sample; permutation test is documented and used.
5. **Silk Board, Mekhri Circle, etc.** emerge as significant hotspots — real-world sanity check.

## Next layers (planned)

- Layer 3: Resource optimization (manpower / barricades / diversions) — consumes OBI + RMST + frailty
- Layer 4: Case-based retrieval for sparse `is_true_planned_event` rows (191 / 8,173)
- Layer 6: Bayesian post-event learning loop
