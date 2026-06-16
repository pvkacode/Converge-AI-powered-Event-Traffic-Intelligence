# Converge — Day 1 Pipeline (ASTraM Bengaluru Traffic Disruption Intelligence)

**Framing:** From reactive patrol logs to a predictive, self-improving disruption-intelligence system for Bengaluru traffic.

This repository implements the **Day 1 foundation**: data cleaning with a composite **Data Trust Score**, **Layer 1** survival-based duration forecasting, and **Layer 2** spatial hotspot detection.

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
python src/layer1_survival.py
python src/layer2_hotspots.py
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
│   ├── layer1_survival_fallback.csv
│   ├── layer1_cox_summary.txt
│   └── layer2_hotspots.csv
├── src/
│   ├── data_pipeline.py
│   ├── layer1_survival.py
│   ├── layer2_hotspots.py
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

- Layer 3: Resource optimization (manpower / barricades / diversions)
- Layer 4: Case-based retrieval for sparse planned events (protest, VIP)
- Layer 5: Hawkes process for unplanned incident cascades
- Layer 6: Bayesian post-event learning loop
