# ASTraM Day 1 — Cursor Handoff Context

**Framing:** From reactive patrol logs to a predictive, self-improving disruption-intelligence system for Bengaluru traffic.

This document is the single source of truth for continuing the project in Cursor with no prior chat context.

---

## Problem & dataset

- **Source:** ASTraM / Bengaluru Traffic Police operational log (~8,170 rows, Nov 2023 – Apr 2024)
- **Raw:** `data/events_raw.csv` (column `id` renamed to `event_id` on load)
- **Clean:** `data/events_clean.parquet`

| Split | Count | Notes |
|-------|------:|-------|
| Unplanned | 7,706 | breakdowns, water-logging, accidents, etc. |
| Planned (`event_type=planned`) | 467 | includes standing construction re-logs |
| **`is_true_planned_event`** | **191 (2.3%)** | public_event, procession, vip_movement, protest — **headline scarcity finding for Layer 4** |

Both planned and unplanned rows are modeled as **duration + spatial impact** problems, not two disconnected pipelines.

---

## Architecture (Day 1 complete)

```
events_raw.csv
    └─► data_pipeline.py  ──► events_clean.parquet (trust_score spine)
            ├─► layer1_survival.py  ──► KM quantiles + Cox PH
            └─► layer2_hotspots.py  ──► Gi* hotspot table
```

Run order:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/data_pipeline.py
python src/layer1_survival.py
python src/layer2_hotspots.py
python src/validate_consistency.py   # sanity checks
```

---

## File 1: `src/data_pipeline.py`

### Pipeline order

1. `load_raw()` — rename `id` → `event_id`
2. `clean_categoricals()` — **before** duration (see below)
3. `parse_datetimes()` — UTC → `start_local` (Asia/Kolkata), `hour_local`, `dow_local`
4. `build_duration_and_censoring()` — end marker priority: closed → end → resolved
5. `add_geo_valid()` — reject null/(0,0) coords
6. `flag_duration_anomalies()` — stratified MAD, `|modified_z| > 3.5`
7. `run_missingness_test()` — logistic LR test → `outputs/missingness_test.txt`
8. `run_isolation_forest()` — multivariate anomaly, bottom 5% flagged
9. `compute_trust_score()` — noisy-OR composite
10. `add_category_codes()` — stable integer codes
11. `print_summary()` + save parquet/csv

### `clean_categoricals()` (load-bearing)

- Strip whitespace; `corridor` null → `"Non-corridor"`
- **Typo fixes:** `"Debris"`/`"debris"` → `debris`; `"Fog / Low Visibility"` → `fog_low_visibility`
- `event_type` → lowercase
- `requires_road_closure` → real bool (handles string TRUE/FALSE and native bool)
- **`is_true_planned_event`** column:

```python
TRUE_PLANNED_CAUSES = {"public_event", "procession", "vip_movement", "protest"}
df["is_true_planned_event"] = df["event_cause"].isin(TRUE_PLANNED_CAUSES)
```

### `trust_score`

```
trust_i = ∏_k (1 − w_k · flag_k,i)
```

| Flag | Weight |
|------|--------|
| duration_anomaly (stratified MAD) | 0.30 |
| NOT geo_valid | 0.40 |
| is_censored AND P(missing) > 0.7 | 0.30 |
| iso_flagged (bottom 5%) | 0.30 |

**Replaces** global 1440-min duration truncation from v1.

### Key data-quality findings (quote in concept note)

1. **4,523** rows lack any end timestamp (`is_censored=True`)
2. **3,526** of those are `status=closed` — status/timestamp inconsistency in source system
3. Missingness is **NOT MCAR** (LR ≈ 955, p ≈ 6e-187); corridor and cause predict gaps
4. Mean `trust_score` ≈ 0.95

---

## File 2: `src/layer1_survival.py`

### Survival table design (consistent with original v1)

```python
surv = df[df["duration_min"].notna() & (df["duration_min"] >= 0]]
surv["T"] = surv["duration_min"]
surv["E"] = (~surv["is_censored"]).astype(int)
surv["weights"] = surv["trust_score"].clip(0.01, 1.0)
```

**Important:** Censored rows have **no** `duration_min` → excluded from KM table. Result: ~3,498 rows, **E=1 for all**, censored count in survival table = 0. This matches original v1 behavior; it is **not** a regression.

| v1 | Current |
|----|---------|
| Hard-drop `duration_min > 1440` | Stratified MAD + `trust_score` weights |
| Unweighted KM | Weighted KM (`weights=trust_score`) |
| Same inclusion rule | Same inclusion rule |

**Do not** admin-censor at study end — pushes P50 to ~160 days and destroys operational quantiles.

### Outputs

- `outputs/layer1_survival_quantiles.csv` — KM by (cause × corridor), min n=15
- `outputs/layer1_survival_fallback.csv` — cause-only fallback
- `outputs/layer1_cox_summary.txt` — Cox PH, concordance ~0.56 (weak; state honestly)

### Public API

```python
lookup_expected_duration(cause, corridor, km_table, km_fallback, quantile="p50")
# → {duration_min, source, n, confidence} or None for sparse causes (e.g. protest)
```

### Verified quantiles (real data)

| Stratum | P50 |
|---------|-----|
| vehicle_breakdown × Mysore Road | ~39 min |
| vehicle_breakdown × Non-corridor | ~45 min |
| construction × Non-corridor | ~662 min |

---

## File 3: `src/layer2_hotspots.py`

- Junction intensity = **sum(trust_score)**, not raw count
- Getis-Ord Gi* via `esda.G_Local`, KNN k=6
- **Primary significance:** permutation `p_sim < 0.05` (NOT z > 1.96)
- Output: `outputs/layer2_hotspots.csv` (~80 significant hotspots on real data; Silk Board top)

---

## Planned layers (not yet built)

| Layer | Purpose |
|-------|---------|
| 3 | Resource optimization (manpower, barricades, diversions) |
| 4 | Case-based retrieval for sparse `is_true_planned_event` rows |
| 5 | Hawkes process for unplanned cascades |
| 6 | Bayesian post-event learning |

Layer 4 **depends on** `is_true_planned_event` + Layer 1 `lookup_expected_duration`.

---

## Dependencies

```
pandas numpy scipy scikit-learn statsmodels lifelines esda libpysal pyarrow
```

See `requirements.txt`.

---

## What NOT to change without explicit reason

1. **`is_true_planned_event` definition** — cause-based, 191 rows; drives Layer 4 framing
2. **Survival inclusion rule** — `duration_min.notna()` only; do not admin-censor
3. **Gi* significance** — use `p_sim`, not asymptotic z
4. **trust_score as weight** — do not hard-drop anomaly rows from KM/Gi*

---

## Validation

`python src/validate_consistency.py` checks:

- `is_true_planned_event` ≈ 191
- debris merged to 13 rows
- censored ⇒ null duration_min
- survival table E=0 count = 0 (by design)
- no missing trust_score
