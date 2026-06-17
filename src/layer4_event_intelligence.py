"""
Layer 4 v3 -- Event Intelligence Engine
ASTraM Bengaluru Traffic Disruption Intelligence

Upgrades over v2:
  - Two-stage retrieval (planned-event-first, full fallback)
  - Retrieval diversity score
  - networkx Knowledge Graph (corridor->cause->outcome->resource_plan)
  - Bayesian graph-prior for closure probability in simulate_event
  - Counterfactual what-if scenario comparisons
  - Evidence-weighted IMS blending for resource recommendations
  - Feature-level contribution explainability per retrieved match

Column mapping (from audit):
  CAUSE_COL     = 'event_cause'
  CORRIDOR_COL  = 'corridor'
  CLOSURE_COL   = 'requires_road_closure'
  DURATION_COL  = 'duration_min'
  PRIORITY_COL  = 'priority'
  TRUST_COL     = 'trust_score'
  PLANNED_COL   = 'is_true_planned_event'
  START_COL     = 'start_local'
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "outputs"
DATA = ROOT / "data"

# ── helpers ───────────────────────────────────────────────────────────────────

def safe_load(path, **kwargs):
    try:
        df = pd.read_csv(path, **kwargs)
        print(f"  Loaded {path}: {df.shape}, cols={list(df.columns)}")
        return df
    except Exception as e:
        print(f"  WARNING: Could not load {path}: {e}")
        return pd.DataFrame()


def weighted_median(values, weights):
    if len(values) == 0:
        return np.nan
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    vals = np.array([p[0] for p in pairs], dtype=float)
    wts  = np.array([p[1] for p in pairs], dtype=float)
    cumw = np.cumsum(wts)
    total = cumw[-1]
    if total <= 0:
        return float(np.mean(vals))
    for i, cw in enumerate(cumw):
        if cw >= 0.5 * total:
            return float(vals[i])
    return float(vals[-1])


def weighted_quantile(values, weights, q=0.80):
    """Weighted quantile with linear interpolation."""
    if len(values) == 0:
        return np.nan
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    vals = np.array([p[0] for p in pairs], dtype=float)
    wts  = np.array([p[1] for p in pairs], dtype=float)
    cumw = np.cumsum(wts)
    total = cumw[-1]
    if total <= 0:
        return float(np.percentile(vals, q * 100))
    target = q * total
    for i, cw in enumerate(cumw):
        if cw >= target:
            if i == 0:
                return float(vals[0])
            prev_cw = cumw[i - 1]
            frac = (target - prev_cw) / max(cw - prev_cw, 1e-9)
            return float(vals[i - 1] + frac * (vals[i] - vals[i - 1]))
    return float(vals[-1])


def _js(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):   return bool(obj)
    raise TypeError(type(obj))


def minmax_norm(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA PREPARATION AND PLANNED EVENT IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 1: Data Preparation and Planned Event Identification")

df_all = pd.read_parquet(DATA / "events_clean.parquet")
df_all = df_all.reset_index(drop=True)
print(f"  events_clean: {df_all.shape}")

# Confirmed column names from audit
CAUSE_COL    = 'event_cause'
CORRIDOR_COL = 'corridor'
CLOSURE_COL  = 'requires_road_closure'
DURATION_COL = 'duration_min'
PRIORITY_COL = 'priority'
TRUST_COL    = 'trust_score'
PLANNED_COL  = 'is_true_planned_event'
START_COL    = 'start_local'
JUNCTION_COL = 'junction'

# hour_local already in df_all as float; dow_local has 116 NaN → fill with median
df_all['hour_of_day'] = df_all['hour_local'].fillna(df_all['hour_local'].median()).astype(int)
df_all['day_of_week'] = df_all['dow_local'].fillna(df_all['dow_local'].median()).astype(int)
df_all['month'] = pd.to_datetime(df_all[START_COL], utc=True, errors='coerce').dt.month.fillna(1).astype(int)

# closure_binary from bool column
df_all['closure_binary'] = df_all[CLOSURE_COL].fillna(False).astype(int)

# priority_numeric: High=3 (most demanding), Low=1, Unknown=2
# (priority_code in data is High=0, Low=1 — alphabetical, operationally inverted)
PRIORITY_MAP = {'High': 3, 'Low': 1}
df_all['priority_numeric'] = df_all[PRIORITY_COL].map(PRIORITY_MAP).fillna(2.0)

# duration_clean: clip to [0, 1440 min = 24h]
df_all['duration_clean'] = df_all[DURATION_COL].clip(lower=0, upper=1440)
GLOBAL_DUR_MEDIAN = float(df_all['duration_clean'].dropna().median())
df_all['duration_clean'] = df_all['duration_clean'].fillna(GLOBAL_DUR_MEDIAN)

# Identify planned events using primary PLANNED_COL
planned_mask    = df_all[PLANNED_COL] == True
n_planned_found = planned_mask.sum()

if n_planned_found < 10:
    # Cause-keyword fallback
    print(f"  WARNING: PLANNED_COL gave {n_planned_found} rows; using cause-keyword fallback")
    kw = ['rally','festival','procession','vip','protest','event','celebr','march','gather','sports','concert']
    planned_mask = df_all[CAUSE_COL].str.lower().str.contains('|'.join(kw), na=False)
    print(f"  Keyword fallback: {planned_mask.sum()} planned events")
else:
    print(f"  Primary PLANNED_COL: {n_planned_found} planned events")

planned_positions = list(df_all[planned_mask].index)   # positions in df_all (0-8172)
df_planned = df_all.loc[planned_positions].reset_index(drop=True).copy()

print(f"  Planned events: {len(df_planned)}")
print(f"  Cause dist: {df_planned[CAUSE_COL].value_counts().to_dict()}")

# Stage 1 pool: event_type == 'planned' (467 events, broader operational context)
STAGE1_MASK    = (df_all['event_type'] == 'planned').to_numpy()
stage1_pos_set = set(np.where(STAGE1_MASK)[0].tolist())
print(f"  Stage 1 pool (event_type=='planned'): {len(stage1_pos_set)} events")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FEATURE ENGINEERING FOR SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 2: Feature Engineering for Similarity")

CATEGORICAL_COLS = [CAUSE_COL, CORRIDOR_COL]
BINARY_COLS      = ['closure_binary']
CONTINUOUS_COLS  = ['hour_of_day', 'day_of_week', 'duration_clean', 'priority_numeric', TRUST_COL, 'month']
ALL_FEATURE_COLS = CATEGORICAL_COLS + BINARY_COLS + CONTINUOUS_COLS

HYBRID_WEIGHTS = {
    CAUSE_COL:          0.35,
    CORRIDOR_COL:       0.25,
    'closure_binary':   0.15,
    'priority_numeric': 0.10,
    'hour_of_day':      0.10,
    'month':            0.05,
}
assert abs(sum(HYBRID_WEIGHTS.values()) - 1.0) < 1e-9, \
    f"Hybrid weights must sum to 1.0, got {sum(HYBRID_WEIGHTS.values())}"

# Fill NaN in numeric feature columns
for col in CONTINUOUS_COLS:
    med = float(pd.to_numeric(df_all[col], errors='coerce').dropna().median())
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(med)
    df_planned[col] = pd.to_numeric(df_planned[col], errors='coerce').fillna(med)

# Label-encode categoricals on full dataset
label_encoders: dict[str, LabelEncoder] = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    le.fit(df_all[col].fillna('__NA__').astype(str))
    df_all[f'{col}_enc']     = le.transform(df_all[col].fillna('__NA__').astype(str))
    df_planned[f'{col}_enc'] = le.transform(df_planned[col].fillna('__NA__').astype(str))
    label_encoders[col] = le

# Feature ranges (for Gower and hybrid continuous similarity)
feature_ranges: dict[str, float] = {}
for col in CONTINUOUS_COLS + BINARY_COLS:
    vals = pd.to_numeric(df_all[col], errors='coerce').dropna()
    feature_ranges[col] = max(float(vals.max() - vals.min()), 1.0)

print(f"  Feature ranges: {feature_ranges}")

# Save event features CSV
ef_save_cols = ['event_id'] + ALL_FEATURE_COLS + [f'{c}_enc' for c in CATEGORICAL_COLS]
ef_df = df_all[ef_save_cols].copy()
ef_df['is_planned'] = df_all[PLANNED_COL].astype(int)
ef_df['row_idx']    = range(len(ef_df))
ef_df.to_csv(OUT / 'layer4_event_features.csv', index=False)
print(f"  Saved: layer4_event_features.csv ({len(ef_df)} rows)")

# Save event index metadata (backward-compat)
meta_df = df_all[['event_id', CAUSE_COL, CORRIDOR_COL, DURATION_COL, TRUST_COL, PLANNED_COL]].copy()
meta_df['start_timestamp'] = (
    pd.to_datetime(df_all[START_COL], utc=True, errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
)
meta_df = meta_df.reset_index().rename(columns={
    'index': 'row_idx', CAUSE_COL: 'event_cause', CORRIDOR_COL: 'corridor',
    DURATION_COL: 'duration_min', TRUST_COL: 'trust_score', PLANNED_COL: 'is_true_planned_event',
})
meta_df.to_csv(OUT / 'layer4_event_index_metadata.csv', index=False)
print(f"  Saved: layer4_event_index_metadata.csv ({len(meta_df)} rows)")

# Save encoders pickle
with open(OUT / 'layer4_encoders.pkl', 'wb') as f:
    pickle.dump({
        'label_encoders':  label_encoders,
        'feature_ranges':  feature_ranges,
        'ALL_FEATURE_COLS': ALL_FEATURE_COLS,
        'CATEGORICAL_COLS': CATEGORICAL_COLS,
        'BINARY_COLS':      BINARY_COLS,
        'CONTINUOUS_COLS':  CONTINUOUS_COLS,
        'CAUSE_COL':        CAUSE_COL,
        'CORRIDOR_COL':     CORRIDOR_COL,
        'CLOSURE_COL':      CLOSURE_COL,
        'DURATION_COL':     DURATION_COL,
        'PRIORITY_COL':     PRIORITY_COL,
        'GLOBAL_DUR_MEDIAN': GLOBAL_DUR_MEDIAN,
    }, f)
print(f"  Saved: layer4_encoders.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: GOWER + HYBRID SIMILARITY ENGINE  (vectorized 191 x 8173)
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 3: Gower + Hybrid Similarity Engine")

n_q  = len(df_planned)
n_db = len(df_all)
print(f"  Computing similarity matrices: {n_q} x {n_db}")


def _to_str_arr(series: pd.Series) -> np.ndarray:
    return series.fillna('__NA__').astype(str).to_numpy(dtype=str)


def _to_float_arr(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors='coerce').fillna(0).to_numpy(dtype=np.float32)


def _build_hybrid_gower_matrices(
    df_q: pd.DataFrame,
    df_db: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (H, G) both shape (n_q, n_db) float32."""
    nq, ndb = len(df_q), len(df_db)

    # Hybrid: weighted sum over HYBRID_WEIGHTS features
    H = np.zeros((nq, ndb), dtype=np.float32)
    for col, w in HYBRID_WEIGHTS.items():
        if col in CATEGORICAL_COLS:
            qa  = _to_str_arr(df_q[col])
            dba = _to_str_arr(df_db[col])
            H  += w * (qa[:, None] == dba[None, :]).astype(np.float32)
        elif col in BINARY_COLS:
            qa  = _to_float_arr(df_q[col])
            dba = _to_float_arr(df_db[col])
            H  += w * (qa[:, None] == dba[None, :]).astype(np.float32)
        else:
            qa  = _to_float_arr(df_q[col])
            dba = _to_float_arr(df_db[col])
            r   = feature_ranges[col]
            H  += w * np.clip(1.0 - np.abs(qa[:, None] - dba[None, :]) / r, 0.0, 1.0)

    # Gower: mean over ALL_FEATURE_COLS
    n_feat = len(ALL_FEATURE_COLS)
    G = np.zeros((nq, ndb), dtype=np.float32)
    for col in ALL_FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            qa  = _to_str_arr(df_q[col])
            dba = _to_str_arr(df_db[col])
            G  += (qa[:, None] == dba[None, :]).astype(np.float32)
        elif col in BINARY_COLS:
            qa  = _to_float_arr(df_q[col])
            dba = _to_float_arr(df_db[col])
            G  += (qa[:, None] == dba[None, :]).astype(np.float32)
        else:
            qa  = _to_float_arr(df_q[col])
            dba = _to_float_arr(df_db[col])
            r   = feature_ranges[col]
            G  += np.clip(1.0 - np.abs(qa[:, None] - dba[None, :]) / r, 0.0, 1.0)
    G /= n_feat

    return H, G


try:
    H_mat, G_mat = _build_hybrid_gower_matrices(df_planned, df_all)
    SIM_mat = 0.60 * H_mat + 0.40 * G_mat
    print(f"  H:   mean={H_mat.mean():.3f}, std={H_mat.std():.3f}")
    print(f"  G:   mean={G_mat.mean():.3f}, std={G_mat.std():.3f}")
    print(f"  SIM: mean={SIM_mat.mean():.3f}, max={SIM_mat.max():.3f}")
except MemoryError:
    print("  WARNING: MemoryError — falling back to row-by-row chunked computation")
    H_mat   = np.zeros((n_q, n_db), dtype=np.float32)
    G_mat   = np.zeros((n_q, n_db), dtype=np.float32)
    CHUNK   = 20
    for start in range(0, n_q, CHUNK):
        end = min(start + CHUNK, n_q)
        h_chunk, g_chunk = _build_hybrid_gower_matrices(
            df_planned.iloc[start:end], df_all
        )
        H_mat[start:end] = h_chunk
        G_mat[start:end] = g_chunk
    SIM_mat = 0.60 * H_mat + 0.40 * G_mat


# ── Two-stage retrieval ──────────────────────────────────────────────────────
K_RETRIEVAL = 5
SIM_THRESHOLD = 0.50   # Stage 1 quality gate

retrieval_results = []
retrieval_rows    = []
gower_sample_rows = []

all_positions = np.arange(n_db)

for qi in range(n_q):
    self_pos = planned_positions[qi]
    sim_row  = SIM_mat[qi].copy()

    # Stage 1: planned-event pool (event_type=='planned') excluding self
    s1_idx = np.array([j for j in sorted(stage1_pos_set) if j != self_pos])
    s1_sim  = sim_row[s1_idx]
    n_s1_good = int(np.sum(s1_sim >= SIM_THRESHOLD))

    if n_s1_good >= 3:
        # Stage 1 only: top K from planned-event pool
        top_k_pos = s1_idx[np.argsort(-s1_sim)[:K_RETRIEVAL]]
        stage_label = 'stage1_only'
    else:
        # Stage 2 fallback: top K from all events
        sim_excl = sim_row.copy()
        sim_excl[self_pos] = -1.0
        top_k_pos = np.argsort(-sim_excl)[:K_RETRIEVAL]
        n_from_s1 = sum(1 for j in top_k_pos if j in stage1_pos_set)
        stage_label = 'stage1+stage2' if n_from_s1 > 0 else 'stage2_only'

    top_k_sims    = sim_row[top_k_pos]
    top_k_hybrid  = H_mat[qi][top_k_pos]
    top_k_gower   = G_mat[qi][top_k_pos]
    mean_top_k    = float(top_k_sims.mean())

    # Feature contributions for best match
    feat_contribs: dict[str, float] = {}
    if len(top_k_pos) > 0:
        best_j = int(top_k_pos[0])
        for col, w in HYBRID_WEIGHTS.items():
            if col in CATEGORICAL_COLS:
                qa  = str(df_planned.iloc[qi][col]) if pd.notna(df_planned.iloc[qi][col]) else '__NA__'
                dba = str(df_all.iloc[best_j][col]) if pd.notna(df_all.iloc[best_j][col]) else '__NA__'
                feat_contribs[col] = round(w * (1.0 if qa == dba else 0.0), 4)
            elif col in BINARY_COLS:
                qa  = int(df_planned.iloc[qi].get(col, 0) or 0)
                dba = int(df_all.iloc[best_j].get(col, 0) or 0)
                feat_contribs[col] = round(w * (1.0 if qa == dba else 0.0), 4)
            else:
                qa  = float(pd.to_numeric(df_planned.iloc[qi].get(col, 0), errors='coerce') or 0)
                dba = float(pd.to_numeric(df_all.iloc[best_j].get(col, 0), errors='coerce') or 0)
                r   = feature_ranges[col]
                feat_contribs[col] = round(w * max(0.0, 1.0 - abs(qa - dba) / r), 4)

    retrieval_results.append({
        'query_idx':         qi,
        'query_event_id':    df_planned.iloc[qi]['event_id'],
        'query_cause':       df_planned.iloc[qi][CAUSE_COL],
        'query_corridor':    df_planned.iloc[qi][CORRIDOR_COL],
        'top_k_indices':     top_k_pos.tolist(),
        'top_k_similarities': top_k_sims.tolist(),
        'top_k_hybrid':      top_k_hybrid.tolist(),
        'top_k_gower':       top_k_gower.tolist(),
        'retrieval_stage':   stage_label,
        'mean_top_k_sim':    round(mean_top_k, 4),
        'feature_contributions_best': feat_contribs,
    })

    # Flat CSV rows
    for rank, (j, sim_v, hyb_v, gow_v) in enumerate(
        zip(top_k_pos, top_k_sims, top_k_hybrid, top_k_gower)
    ):
        db_row = df_all.iloc[int(j)]
        retrieval_rows.append({
            'query_event_id':       df_planned.iloc[qi]['event_id'],
            'query_cause':          df_planned.iloc[qi][CAUSE_COL],
            'query_corridor':       df_planned.iloc[qi][CORRIDOR_COL],
            'query_is_planned':     True,
            'retrieved_event_id':   db_row['event_id'],
            'retrieved_cause':      db_row[CAUSE_COL],
            'retrieved_corridor':   db_row[CORRIDOR_COL],
            'final_similarity_score': round(float(sim_v), 4),
            'gower_component':      round(float(gow_v), 4),
            'hybrid_component':     round(float(hyb_v), 4),
            'retrieved_duration_min': round(float(db_row.get('duration_clean', GLOBAL_DUR_MEDIAN) or GLOBAL_DUR_MEDIAN), 1),
            'retrieved_closure':    int(db_row.get('closure_binary', 0) or 0),
            'retrieved_priority':   db_row.get(PRIORITY_COL, 'Low'),
            'rank':                 rank + 1,
            'retrieval_stage':      stage_label,
        })

    # Gower sample: top-10 per event
    top10_pos  = np.argsort(-sim_row)
    top10_pos  = [j for j in top10_pos if j != self_pos][:10]
    for rank, j in enumerate(top10_pos):
        db_row = df_all.iloc[int(j)]
        gower_sample_rows.append({
            'planned_event_id':   df_planned.iloc[qi]['event_id'],
            'planned_cause':      df_planned.iloc[qi][CAUSE_COL],
            'retrieved_event_id': db_row['event_id'],
            'retrieved_cause':    db_row[CAUSE_COL],
            'retrieved_corridor': db_row[CORRIDOR_COL],
            'gower_score':        round(float(G_mat[qi, j]), 4),
            'hybrid_score':       round(float(H_mat[qi, j]), 4),
            'final_score':        round(float(SIM_mat[qi, j]), 4),
            'rank':               rank + 1,
        })

retrieval_df = pd.DataFrame(retrieval_rows)
retrieval_df.to_csv(OUT / 'layer4_retrieval_results.csv', index=False)
print(f"  Saved: layer4_retrieval_results.csv ({len(retrieval_df)} rows)")

pd.DataFrame(gower_sample_rows).to_csv(OUT / 'layer4_gower_similarity_sample.csv', index=False)
print(f"  Saved: layer4_gower_similarity_sample.csv ({len(gower_sample_rows)} rows)")

# Save nn_index.pkl as pickled retrieval_results (new format)
with open(OUT / 'layer4_nn_index.pkl', 'wb') as f:
    pickle.dump(retrieval_results, f)
print(f"  Saved: layer4_nn_index.pkl (retrieval cache, {len(retrieval_results)} entries)")

with open(OUT / 'layer4_similarity_weights.json', 'w', encoding='utf-8') as f:
    json.dump({
        'hybrid_weights':  HYBRID_WEIGHTS,
        'gower_weight':    0.40,
        'hybrid_weight':   0.60,
        'rationale': (
            'Cause and corridor dominate because same cause+corridor predicts'
            ' outcome most reliably in traffic operations'
        ),
        'categorical_features': CATEGORICAL_COLS,
        'binary_features':      BINARY_COLS,
        'continuous_features':  CONTINUOUS_COLS,
    }, f, indent=2)
print(f"  Saved: layer4_similarity_weights.json")

# Build per-query lookup by event_id
rr_by_qid = {r['query_event_id']: r for r in retrieval_results}
retrieval_by_qid = retrieval_df.groupby('query_event_id')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RETRIEVAL CONFIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 4: Retrieval Confidence Score")


def confidence_level(score: float) -> str:
    if score >= 0.85: return "Very High"
    if score >= 0.70: return "High"
    if score >= 0.55: return "Moderate"
    return "Weak"


conf_lu: dict[str, tuple[float, str]] = {}
for rr in retrieval_results:
    s = rr['mean_top_k_sim']
    conf_lu[rr['query_event_id']] = (round(s, 4), confidence_level(s))

conf_dist: dict[str, int] = {}
for _, (_, cl) in conf_lu.items():
    conf_dist[cl] = conf_dist.get(cl, 0) + 1
print(f"  Confidence distribution (191 events): {conf_dist}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: RETRIEVAL DIVERSITY SCORE
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 5: Retrieval Diversity Score")


def diversity_interpretation(score: float) -> str:
    if score >= 0.70: return "High diversity -- historical outcomes vary widely; predictions uncertain"
    if score >= 0.40: return "Moderate diversity -- mixed historical outcomes"
    return "Low diversity -- retrieved events strongly agree; predictions reliable"


div_lu: dict[str, tuple[float, str]] = {}
for rr in retrieval_results:
    top_k = rr['top_k_indices']
    if len(top_k) < 2:
        div_lu[rr['query_event_id']] = (0.0, diversity_interpretation(0.0))
        continue
    pairs_sim = []
    for a in range(len(top_k)):
        for b in range(a + 1, len(top_k)):
            ja, jb = int(top_k[a]), int(top_k[b])
            # Use precomputed SIM row of a against b: need full pairwise
            # Cheaper: Gower between two db rows
            row_a = df_all.iloc[ja][ALL_FEATURE_COLS]
            row_b = df_all.iloc[jb][ALL_FEATURE_COLS]
            s = 0.0
            for col in ALL_FEATURE_COLS:
                va = str(row_a[col]) if col in CATEGORICAL_COLS else float(row_a[col] or 0)
                vb = str(row_b[col]) if col in CATEGORICAL_COLS else float(row_b[col] or 0)
                if col in CATEGORICAL_COLS or col in BINARY_COLS:
                    s += 1.0 if va == vb else 0.0
                else:
                    r = feature_ranges[col]
                    s += max(0.0, 1.0 - abs(float(va) - float(vb)) / r)
            pairs_sim.append(s / len(ALL_FEATURE_COLS))
    mean_pair = float(np.mean(pairs_sim)) if pairs_sim else 0.0
    div_score = round(1.0 - mean_pair, 4)
    div_lu[rr['query_event_id']] = (div_score, diversity_interpretation(div_score))

print(f"  Mean diversity score: {np.mean([v[0] for v in div_lu.values()]):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: INSTITUTIONAL MEMORY SCORE
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 6: Institutional Memory Score (IMS)")

IMS_THRESHOLD = 0.50
IMS_LEVELS = [
    (1.5, "Strong institutional memory -- high reliability"),
    (0.8, "Moderate institutional memory"),
    (0.3, "Weak institutional memory -- use Layer 3 rules as primary"),
    (0.0, "Sparse -- no reliable historical analogue; rules only"),
]


def ims_interpretation(score: float) -> str:
    for threshold, label in IMS_LEVELS:
        if score >= threshold:
            return label
    return IMS_LEVELS[-1][1]


ims_rows = []
ims_lu: dict[str, dict] = {}

for qi, rr in enumerate(retrieval_results):
    sim_row        = SIM_mat[qi]
    self_pos       = planned_positions[qi]
    meaningful_idx = [j for j in range(n_db) if j != self_pos and sim_row[j] >= IMS_THRESHOLD]
    n_mean         = len(meaningful_idx)
    mean_sim       = float(np.mean([sim_row[j] for j in meaningful_idx])) if n_mean > 0 else 0.0
    ims_score      = math.log1p(n_mean) * mean_sim
    conf_s, conf_l = conf_lu[rr['query_event_id']]
    div_s,  div_i  = div_lu[rr['query_event_id']]

    rec = {
        'event_id':            rr['query_event_id'],
        'event_cause':         rr['query_cause'],
        'corridor':            rr['query_corridor'],
        'n_meaningful_matches': n_mean,
        'mean_similarity':     round(mean_sim, 4),
        'ims_score':           round(ims_score, 4),
        'ims_interpretation':  ims_interpretation(ims_score),
        'confidence_score':    conf_s,
        'confidence_level':    conf_l,
        'diversity_score':     div_s,
        'diversity_interpretation': div_i,
    }
    ims_rows.append(rec)
    ims_lu[rr['query_event_id']] = rec

ims_df = pd.DataFrame(ims_rows)
ims_df.to_csv(OUT / 'layer4_institutional_memory_scores.csv', index=False)

IMS_MAX = float(ims_df['ims_score'].max())
print(f"  IMS stats: mean={ims_df['ims_score'].mean():.3f}, max={IMS_MAX:.3f}")
ims_dist = {'Strong': 0, 'Moderate': 0, 'Weak': 0, 'Sparse': 0}
for v in ims_df['ims_score']:
    if v >= 1.5: ims_dist['Strong'] += 1
    elif v >= 0.8: ims_dist['Moderate'] += 1
    elif v >= 0.3: ims_dist['Weak'] += 1
    else: ims_dist['Sparse'] += 1
print(f"  IMS distribution: {ims_dist}")
print(f"  Saved: layer4_institutional_memory_scores.csv ({len(ims_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: WEIGHTED OUTCOME PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 7: Weighted Outcome Prediction")

# Load Layer 2 for OBI, persistence, Hawkes
obi_df     = safe_load(OUT / 'layer2_operational_burden_index.csv')
persist_df = safe_load(OUT / 'layer2_hotspot_persistence.csv')
hawkes_df  = safe_load(OUT / 'layer2_hawkes_cascade_risk.csv')

# Build junction-level lookups
obi_lu   = dict(zip(obi_df['junction'], obi_df['operational_burden_index'])) if len(obi_df) else {}
pers_lu  = dict(zip(persist_df['junction'], persist_df['hotspot_persistence_index'])) if len(persist_df) else {}
haw_lu   = dict(zip(hawkes_df['junction'], hawkes_df['cascade_risk'])) if len(hawkes_df) else {}

# Build corridor → best junction mapping via df_all
junc_corr = (
    df_all[df_all[JUNCTION_COL].notna()][[CORRIDOR_COL, JUNCTION_COL, 'duration_clean']]
    .groupby([CORRIDOR_COL, JUNCTION_COL])['duration_clean'].count()
    .reset_index(name='n')
    .sort_values('n', ascending=False)
    .drop_duplicates(CORRIDOR_COL)
    .set_index(CORRIDOR_COL)[JUNCTION_COL]
    .to_dict()
)

GLOBAL_OBI   = float(obi_df['operational_burden_index'].mean()) if len(obi_df) else 0.5
GLOBAL_PERS  = float(persist_df['hotspot_persistence_index'].mean()) if len(persist_df) else 0.5
GLOBAL_HAWKES = float(hawkes_df['cascade_risk'].mean()) if len(hawkes_df) else 0.05

outcome_rows = []
outcome_lu: dict[str, dict] = {}

for qi, rr in enumerate(retrieval_results):
    qid          = rr['query_event_id']
    top_k_idx    = [int(j) for j in rr['top_k_indices']]
    top_k_sims   = [float(s) for s in rr['top_k_similarities']]
    W_total      = max(sum(top_k_sims), 1e-12)
    w            = [s / W_total for s in top_k_sims]

    dur_vals  = [float(df_all.iloc[j].get('duration_clean', GLOBAL_DUR_MEDIAN) or GLOBAL_DUR_MEDIAN) for j in top_k_idx]
    cl_vals   = [float(df_all.iloc[j].get('closure_binary', 0) or 0) for j in top_k_idx]
    pr_vals   = [float(df_all.iloc[j].get('priority_numeric', 2) or 2) for j in top_k_idx]
    ts_vals   = [float(df_all.iloc[j].get(TRUST_COL, 0.7) or 0.7) for j in top_k_idx]

    p50          = weighted_quantile(dur_vals, top_k_sims, q=0.50)
    p80          = weighted_quantile(dur_vals, top_k_sims, q=0.80)
    closure_prob = float(np.dot(w, cl_vals))
    pr_weighted  = float(np.dot(w, pr_vals))

    min_p, max_p = 1.0, 3.0
    congestion_mult = 1.0 + (pr_weighted - min_p) / max(max_p - min_p, 1.0) * 3.0

    # OBI / persistence / hawkes via query corridor → junction
    q_corridor = rr['query_corridor']
    q_junction = (
        df_planned.iloc[qi].get(JUNCTION_COL)
        or junc_corr.get(q_corridor)
    )
    if pd.isna(q_junction) if q_junction is not None else True:
        q_junction = None

    area_obi  = obi_lu.get(q_junction, GLOBAL_OBI) if q_junction else GLOBAL_OBI
    pers_sc   = pers_lu.get(q_junction, GLOBAL_PERS) if q_junction else GLOBAL_PERS
    hawkes_sc = haw_lu.get(q_junction, GLOBAL_HAWKES) if q_junction else GLOBAL_HAWKES

    conf_s, conf_l = conf_lu[qid]
    ims_rec  = ims_lu[qid]
    div_s,   _ = div_lu[qid]

    rec = {
        'event_id':                qid,
        'event_cause':             rr['query_cause'],
        'corridor':                rr['query_corridor'],
        'predicted_duration_p50':  round(p50, 1),
        'predicted_duration_p80':  round(p80, 1),
        'closure_probability':     round(closure_prob, 4),
        'congestion_multiplier_estimate': round(congestion_mult, 3),
        'area_influence_obi':      round(area_obi, 4),
        'persistence_score':       round(pers_sc, 4),
        'hawkes_score':            round(hawkes_sc, 4),
        'confidence_score':        conf_s,
        'confidence_level':        conf_l,
        'ims_score':               ims_rec['ims_score'],
        'diversity_score':         div_s,
        'diversity_interpretation': div_lu[qid][1],
        'retrieval_stage':         rr['retrieval_stage'],
        'n_retrieved':             len(top_k_idx),
        'junction_used':           q_junction or 'None',
    }
    outcome_rows.append(rec)
    outcome_lu[qid] = rec

outcome_df = pd.DataFrame(outcome_rows)
print(f"  Duration P50: mean={outcome_df['predicted_duration_p50'].mean():.0f} min")
print(f"  Closure prob: mean={outcome_df['closure_probability'].mean():.3f}")
print(f"  Stage dist: {outcome_df['retrieval_stage'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: EVENT IMPACT FORECAST (EVENT SEVERITY INDEX)
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 8: Event Impact Forecast (Event Severity Index)")


def impact_level(score: float) -> str:
    if score < 25:  return "Low"
    if score < 50:  return "Moderate"
    if score < 75:  return "High"
    return "Critical"


norm_dur  = minmax_norm(outcome_df['predicted_duration_p50'])
norm_cl   = minmax_norm(outcome_df['closure_probability'])
norm_obi  = minmax_norm(outcome_df['area_influence_obi'])
norm_pers = minmax_norm(outcome_df['persistence_score'])
norm_haw  = minmax_norm(outcome_df['hawkes_score'])

outcome_df['impact_score'] = (
    0.35 * norm_dur +
    0.25 * norm_cl  +
    0.20 * norm_obi +
    0.10 * norm_pers +
    0.10 * norm_haw
) * 100.0

outcome_df['impact_level'] = outcome_df['impact_score'].apply(impact_level)
print(f"  Impact score: {outcome_df['impact_score'].min():.1f}–{outcome_df['impact_score'].max():.1f}")
print(f"  Tier dist: {outcome_df['impact_level'].value_counts().to_dict()}")

# Also save per-event predicted_duration_min alias for backward-compat
outcome_df['predicted_duration_min']   = outcome_df['predicted_duration_p50']
outcome_df['predicted_trust_score']    = [
    float(np.dot(
        [s / max(sum(rr['top_k_similarities']), 1e-12) for s in rr['top_k_similarities']],
        [float(df_all.iloc[j].get(TRUST_COL, 0.7) or 0.7) for j in rr['top_k_indices']]
    ))
    for rr in retrieval_results
]
outcome_df['road_closure_probability'] = outcome_df['closure_probability']
outcome_df['retrieval_confidence']     = outcome_df['confidence_level']

outcome_df.to_csv(OUT / 'layer4_event_outcome_predictions.csv', index=False)
print(f"  Saved: layer4_event_outcome_predictions.csv ({len(outcome_df)} rows)")

# Update outcome_lu with impact
for _, row in outcome_df.iterrows():
    outcome_lu[row['event_id']].update({
        'impact_score': float(row['impact_score']),
        'impact_level': row['impact_level'],
    })


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: FEATURE CONTRIBUTION EXPLAINABILITY
# (already computed per-query in Section 3; embedded in retrieval_results)
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 9: Feature Contribution Explainability")
print("  Feature contributions stored in layer4_retrieval_results.csv and retrieval_results cache.")
print("  Weights:", HYBRID_WEIGHTS)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: RETRIEVAL-BASED RESOURCE RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 10: Retrieval-Based Resource Recommendation")

mp_df  = safe_load(OUT / 'layer3_manpower_recommendations.csv')
dis_df = safe_load(OUT / 'layer3_disruption_impact_scores.csv')

# Build corridor→L3 resource lookup via junction
TIER_DEFAULTS = {
    'Critical': {'officers': 14, 'tow': 3, 'barricades': 24},
    'High':     {'officers': 10, 'tow': 1, 'barricades': 16},
    'Moderate': {'officers':  6, 'tow': 1, 'barricades':  8},
    'Low':      {'officers':  2, 'tow': 0, 'barricades':  0},
}

if len(mp_df):
    mp_lu = mp_df.set_index('junction')[['allocated_officers', 'allocated_tow', 'allocated_barricades', 'ods_score']].to_dict('index')
else:
    mp_lu = {}

if len(dis_df):
    dis_lu_junc = dis_df.set_index('junction')[['dis_score', 'risk_level']].to_dict('index')
else:
    dis_lu_junc = {}

# Corridor → best junction (highest ODS in L3)
corr_best_junc: dict[str, str] = {}
if len(mp_df):
    ev_jc = df_all[df_all[JUNCTION_COL].notna()][[CORRIDOR_COL, JUNCTION_COL]].drop_duplicates()
    ev_jc = ev_jc.merge(mp_df[['junction', 'ods_score']], on='junction', how='inner')
    best  = ev_jc.sort_values('ods_score', ascending=False).drop_duplicates(CORRIDOR_COL)
    corr_best_junc = dict(zip(best[CORRIDOR_COL], best['junction']))


def _l3_resources(junction: str | None, corridor: str | None, tier: str = 'Moderate') -> dict:
    if junction and junction in mp_lu:
        r = mp_lu[junction]
        return {'officers': int(r['allocated_officers']), 'tow': int(r['allocated_tow']), 'barricades': int(r['allocated_barricades'])}
    best_j = corr_best_junc.get(corridor or '')
    if best_j and best_j in mp_lu:
        r = mp_lu[best_j]
        return {'officers': int(r['allocated_officers']), 'tow': int(r['allocated_tow']), 'barricades': int(r['allocated_barricades'])}
    return TIER_DEFAULTS.get(tier, TIER_DEFAULTS['Moderate']).copy()


evid_rows = []
evid_lu: dict[str, dict] = {}

for qi, rr in enumerate(retrieval_results):
    qid        = rr['query_event_id']
    top_k_idx  = [int(j) for j in rr['top_k_indices']]
    top_k_sims = [float(s) for s in rr['top_k_similarities']]
    ims_rec    = ims_lu[qid]
    ims_val    = ims_rec['ims_score']
    ilevel     = outcome_lu.get(qid, {}).get('impact_level', 'Moderate')

    # Evidence resources from retrieved events
    hist_off, hist_tow, hist_bar = [], [], []
    for j, sim_v in zip(top_k_idx, top_k_sims):
        db_row   = df_all.iloc[j]
        j_junc   = db_row.get(JUNCTION_COL) if pd.notna(db_row.get(JUNCTION_COL)) else None
        j_corr   = str(db_row.get(CORRIDOR_COL, ''))
        j_res    = _l3_resources(j_junc, j_corr, ilevel)
        hist_off.append(j_res['officers'])
        hist_tow.append(j_res['tow'])
        hist_bar.append(j_res['barricades'])

    ev_off = weighted_median(hist_off, top_k_sims)
    ev_tow = weighted_median(hist_tow, top_k_sims)
    ev_bar = weighted_median(hist_bar, top_k_sims)

    # L3 rules for query
    q_junc  = df_planned.iloc[qi].get(JUNCTION_COL) if pd.notna(df_planned.iloc[qi].get(JUNCTION_COL)) else None
    q_corr  = rr['query_corridor']
    l3_res  = _l3_resources(q_junc, q_corr, ilevel)

    # IMS-driven evidence weight
    if ims_val >= 1.5:   ev_w = 0.70
    elif ims_val >= 0.8: ev_w = 0.50
    elif ims_val >= 0.3: ev_w = 0.30
    else:                ev_w = 0.10

    final_off = max(2, min(25, round(ev_w * ev_off + (1 - ev_w) * l3_res['officers'])))
    final_tow = max(0, min(10, round(ev_w * ev_tow + (1 - ev_w) * l3_res['tow'])))
    final_bar = max(0, min(40, round(ev_w * ev_bar + (1 - ev_w) * l3_res['barricades'])))

    rec = {
        'event_id':       qid,
        'event_cause':    rr['query_cause'],
        'corridor':       q_corr,
        'evidence_officers': round(float(ev_off), 1),
        'evidence_tow':      round(float(ev_tow), 1),
        'evidence_barricades': round(float(ev_bar), 1),
        'l3_officers':    l3_res['officers'],
        'l3_tow':         l3_res['tow'],
        'l3_barricades':  l3_res['barricades'],
        'final_officers': final_off,
        'final_tow':      final_tow,
        'final_barricades': final_bar,
        'evidence_weight': round(ev_w, 2),
        'ims_score':      round(ims_val, 4),
        'confidence_level': ims_rec['confidence_level'],
        'impact_level':   ilevel,
    }
    evid_rows.append(rec)
    evid_lu[qid] = rec

evid_df = pd.DataFrame(evid_rows)
evid_df.to_csv(OUT / 'layer4_evidence_based_recommendations.csv', index=False)
print(f"  Officers: {evid_df['final_officers'].min()}–{evid_df['final_officers'].max()} "
      f"(mean={evid_df['final_officers'].mean():.1f})")
print(f"  Saved: layer4_evidence_based_recommendations.csv ({len(evid_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 11: Knowledge Graph")

G = nx.DiGraph()

# Pre-collect outcomes for planned events
med_p50 = float(outcome_df['predicted_duration_p50'].median())

def _outcome_nodes(qid: str) -> list[str]:
    oc = outcome_lu.get(qid, {})
    nodes = []
    if oc.get('closure_probability', 0) >= 0.5:
        nodes.append('High_Closure_Probability')
    else:
        nodes.append('Low_Closure_Probability')
    if oc.get('predicted_duration_p50', med_p50) >= med_p50:
        nodes.append('Long_Duration')
    else:
        nodes.append('Short_Duration')
    return nodes

OUTCOME_NODES = ['High_Closure_Probability', 'Low_Closure_Probability', 'Long_Duration', 'Short_Duration']
for on in OUTCOME_NODES:
    G.add_node(on, node_type='outcome')

# Add corridor and event_type nodes + edges
corr_cause_events: dict[tuple[str, str], list[str]] = {}
for qi in range(n_q):
    qid   = df_planned.iloc[qi]['event_id']
    corr  = rr['query_corridor'] if (rr := rr_by_qid.get(qid)) else df_planned.iloc[qi][CORRIDOR_COL]
    cause = df_planned.iloc[qi][CAUSE_COL]
    key   = (str(corr), str(cause))
    corr_cause_events.setdefault(key, []).append(qid)

for (corr_n, cause_n), event_ids in corr_cause_events.items():
    G.add_node(corr_n,  node_type='corridor')
    G.add_node(cause_n, node_type='event_type')
    G.add_edge(corr_n, cause_n,
               weight=len(event_ids),
               edge_label='corridor_to_cause',
               events=event_ids)

# event_type → outcome edges
for cause_n in [n for n, d in G.nodes(data=True) if d.get('node_type') == 'event_type']:
    cause_events = [
        qid for (c, ca), eids in corr_cause_events.items()
        if ca == cause_n for qid in eids
    ]
    n_total = max(len(cause_events), 1)
    for on in OUTCOME_NODES:
        count = sum(1 for qid in cause_events if on in _outcome_nodes(qid))
        if count > 0:
            G.add_edge(cause_n, on,
                       weight=count / n_total,
                       edge_label='cause_to_outcome')

# outcome → resource_plan edges
for on in OUTCOME_NODES:
    on_events = [qid for qi in range(n_q)
                 for qid in [df_planned.iloc[qi]['event_id']]
                 if on in _outcome_nodes(qid)]
    if not on_events:
        continue
    med_off = float(np.median([evid_lu[qid]['final_officers'] for qid in on_events if qid in evid_lu]))
    med_tow = float(np.median([evid_lu[qid]['final_tow']      for qid in on_events if qid in evid_lu]))
    med_bar = float(np.median([evid_lu[qid]['final_barricades'] for qid in on_events if qid in evid_lu]))
    rp_node = f"Officers_{int(med_off)}_Tow_{int(med_tow)}_Barricades_{int(med_bar)}"
    G.add_node(rp_node, node_type='resource_plan', officers=int(med_off), tow=int(med_tow), barricades=int(med_bar))
    mean_ims = float(np.mean([ims_lu[qid]['ims_score'] for qid in on_events if qid in ims_lu])) if on_events else 0.0
    G.add_edge(on, rp_node, weight=round(mean_ims, 4), edge_label='outcome_to_plan')

n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
print(f"  Graph: {n_nodes} nodes, {n_edges} edges")
_nt_dist: dict[str, int] = {}
for _, _nd in G.nodes(data=True):
    _t = _nd.get('node_type', 'unknown')
    _nt_dist[_t] = _nt_dist.get(_t, 0) + 1
print(f"  Node types: {_nt_dist}")

# Save graph (NetworkX >= 3.0 removed write_gpickle → use pickle.dump)
graph_path = OUT / 'layer4_knowledge_graph.gpickle'
with open(graph_path, 'wb') as f:
    pickle.dump(G, f)
print(f"  Saved: layer4_knowledge_graph.gpickle ({graph_path.stat().st_size // 1024} KB)")

# Human-readable summary CSV
graph_rows = []
for u, v, data in G.edges(data=True):
    graph_rows.append({
        'source_node':  u,
        'source_type':  G.nodes[u].get('node_type', 'unknown'),
        'target_node':  v,
        'target_type':  G.nodes[v].get('node_type', 'unknown'),
        'edge_weight':  round(float(data.get('weight', 0)), 4),
        'edge_label':   data.get('edge_label', ''),
    })
pd.DataFrame(graph_rows).to_csv(OUT / 'layer4_knowledge_graph_summary.csv', index=False)
print(f"  Saved: layer4_knowledge_graph_summary.csv ({len(graph_rows)} edges)")

# Build graph-prior closure lookup for simulate_event
def _graph_prior_closure(cause: str, corridor: str) -> float:
    """Traverse corridor→cause→outcome to get prior closure probability."""
    # Check if cause node has outgoing edges to outcome nodes
    if cause not in G:
        return float(outcome_df['closure_probability'].mean())
    out_edges = [(v, d['weight']) for u, v, d in G.out_edges(cause, data=True)
                 if d.get('edge_label') == 'cause_to_outcome']
    if not out_edges:
        return float(outcome_df['closure_probability'].mean())
    total_w = max(sum(w for _, w in out_edges), 1e-9)
    hi_w = sum(w for v, w in out_edges if 'High_Closure' in v)
    return round(hi_w / total_w, 4)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: EXPLAINABLE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 12: Explainable Recommendations")

top20_df = outcome_df.nlargest(20, 'impact_score').reset_index(drop=True)
recs_text, recs_json = [], []

SEP_LONG  = chr(0x2501) * 39   # box-drawing = row
SEP_SHORT = '-' * 39

for _, row in top20_df.iterrows():
    qid   = row['event_id']
    rr    = rr_by_qid.get(qid, {})
    ims_r = ims_lu.get(qid, {})
    ev_r  = evid_lu.get(qid, {})
    oc_r  = outcome_lu.get(qid, row.to_dict())

    cause  = str(row['event_cause'])
    corr   = str(row['corridor'])
    conf_s = float(row.get('confidence_score', 0))
    conf_l = str(row.get('confidence_level', 'Unknown'))
    ims_v  = float(ims_r.get('ims_score', 0))
    n_mean = int(ims_r.get('n_meaningful_matches', 0))
    div_s  = float(row.get('diversity_score', 0))
    div_i  = str(ims_r.get('diversity_interpretation', ''))
    stage  = str(rr.get('retrieval_stage', 'stage2_only'))

    p50   = float(row['predicted_duration_p50'])
    p80   = float(row.get('predicted_duration_p80', p50 * 1.3))
    cl_pr = float(row['closure_probability'])
    cong  = float(row.get('congestion_multiplier_estimate', 1.0))
    imp   = float(row['impact_score'])
    ilev  = str(row['impact_level'])

    ev_w      = float(ev_r.get('evidence_weight', 0.5))
    final_off = int(ev_r.get('final_officers', 6))
    final_tow = int(ev_r.get('final_tow', 1))
    final_bar = int(ev_r.get('final_barricades', 8))
    final_sup = max(0, math.ceil(final_off / 6))
    qru       = 1 if imp >= 75 else 0

    # Graph prior
    gp_cl    = _graph_prior_closure(cause, corr)
    final_cl = round(0.3 * gp_cl + 0.7 * cl_pr, 4)

    # Best match details
    best_cause, best_corr = '', ''
    best_gower = best_hybrid = best_final = 0.0
    feat_contribs = rr.get('feature_contributions_best', {})
    if rr.get('top_k_indices'):
        bj = int(rr['top_k_indices'][0])
        bd = df_all.iloc[bj]
        best_cause  = str(bd[CAUSE_COL])
        best_corr   = str(bd[CORRIDOR_COL])
        best_final  = float(rr['top_k_similarities'][0])
        best_hybrid = float(rr['top_k_hybrid'][0]) if rr.get('top_k_hybrid') else 0.0
        best_gower  = float(rr['top_k_gower'][0])  if rr.get('top_k_gower')  else 0.0

    text_block = (
        f"\nDEPLOYMENT RECOMMENDATION -- {cause} at {corr}\n"
        f"{SEP_LONG}\n"
        f"RETRIEVAL METHOD: Two-Stage (Stage: {stage})\n"
        f"  Similarity: Gower (40%) + Domain-Weighted Hybrid (60%)\n"
        f"  Hybrid weights: Cause=0.35, Corridor=0.25, Closure=0.15, Priority=0.10, Time=0.10, Month=0.05\n\n"
        f"RETRIEVAL QUALITY:\n"
        f"  Confidence: {conf_l} (score={conf_s:.3f})\n"
        f"  Institutional Memory Score: {ims_v:.2f} ({n_mean} meaningful matches)\n"
        f"  Diversity: {div_i}\n\n"
        f"BEST HISTORICAL MATCH (rank 1):\n"
        f"  Event: {best_cause} at {best_corr}\n"
        f"  Final similarity: {best_final:.3f} (Gower={best_gower:.3f}, Hybrid={best_hybrid:.3f})\n"
        f"  Feature contributions: "
        + " | ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in feat_contribs.items()) + "\n\n"
        f"PREDICTED OUTCOMES:\n"
        f"  Duration: {p50:.0f} min (P50) / {p80:.0f} min (P80)\n"
        f"  Road closure probability: {cl_pr*100:.0f}%\n"
        f"  Congestion: {cong:.1f}x normal\n"
        f"  Impact Score: {imp:.0f}/100 -- {ilev}\n\n"
        f"KNOWLEDGE GRAPH CONTEXT:\n"
        f"  Graph-prior closure probability: {gp_cl:.2f}\n"
        f"  Final blended closure probability: {final_cl:.2f}\n\n"
        f"RESOURCE RECOMMENDATION (evidence blend: {ev_w*100:.0f}% historical / {(1-ev_w)*100:.0f}% rules):\n"
        f"  Officers: {final_off} | Supervisors: {final_sup} | Barricades: {final_bar}\n"
        f"  Tow vehicles: {final_tow} | QRU: {qru}\n"
        f"{SEP_LONG}"
    )
    recs_text.append(text_block)
    recs_json.append({
        'event_id':               qid,
        'event_cause':            cause,
        'corridor':               corr,
        'retrieval_stage':        stage,
        'confidence_score':       conf_s,
        'confidence_level':       conf_l,
        'ims_score':              round(ims_v, 4),
        'n_meaningful_matches':   n_mean,
        'diversity_score':        div_s,
        'diversity_interpretation': div_i,
        'best_match_cause':       best_cause,
        'best_match_corridor':    best_corr,
        'best_match_final_sim':   round(best_final, 4),
        'best_match_gower':       round(best_gower, 4),
        'best_match_hybrid':      round(best_hybrid, 4),
        'feature_contributions':  feat_contribs,
        'predicted_duration_p50': round(p50, 1),
        'predicted_duration_p80': round(p80, 1),
        'closure_probability':    round(cl_pr, 4),
        'graph_prior_closure':    gp_cl,
        'final_closure_probability': final_cl,
        'congestion_multiplier':  round(cong, 3),
        'impact_score':           round(imp, 2),
        'impact_level':           ilev,
        'evidence_weight':        round(ev_w, 2),
        'final_officers':         final_off,
        'final_supervisors':      final_sup,
        'final_barricades':       final_bar,
        'final_tow':              final_tow,
        'qru_units':              qru,
        'top_retrieved': [
            {
                'rank':       r + 1,
                'event_id':   df_all.iloc[int(j)]['event_id'],
                'cause':      str(df_all.iloc[int(j)][CAUSE_COL]),
                'corridor':   str(df_all.iloc[int(j)][CORRIDOR_COL]),
                'final_sim':  round(float(rr['top_k_similarities'][r]), 4),
                'gower_sim':  round(float(rr['top_k_gower'][r]), 4),
                'hybrid_sim': round(float(rr['top_k_hybrid'][r]), 4),
            }
            for r, j in enumerate(rr.get('top_k_indices', []))
        ],
    })

with open(OUT / 'layer4_explainable_recommendations.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(recs_text))
with open(OUT / 'layer4_explainable_recommendations.json', 'w', encoding='utf-8') as f:
    json.dump(recs_json, f, indent=2, default=_js)
print(f"  Saved: layer4_explainable_recommendations.txt | .json (top-{len(recs_json)} events)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: COUNTERFACTUAL + WHAT-IF SCENARIO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 13: Counterfactual + What-If Scenario Simulator")


def _find_closest(value: str, classes: list[str]) -> str:
    v_low = str(value).lower()
    for c in classes:
        if v_low in c.lower() or c.lower() in v_low:
            return c
    # Try keyword containment
    for kw in v_low.split():
        for c in classes:
            if kw in c.lower():
                return c
    return classes[0]   # fallback: most frequent (first in sorted classes)


def _build_scenario_row(
    event_type: str,
    corridor: str,
    day_of_week: int,
    hour_of_day: int,
    expected_crowd_size: int,
    requires_closure: bool | None = None,
) -> pd.DataFrame:
    matched_cause  = _find_closest(event_type, list(label_encoders[CAUSE_COL].classes_))
    matched_corr   = _find_closest(corridor,    list(label_encoders[CORRIDOR_COL].classes_))
    cl_val         = 1 if (requires_closure is True) else (0 if requires_closure is False else 0)
    crowd_factor   = min(math.log1p(expected_crowd_size / 5000), 2.0) if expected_crowd_size > 0 else 1.0
    dur_est        = GLOBAL_DUR_MEDIAN * max(1.0, crowd_factor)
    priority_n     = 3.0  # High by default for scenarios

    row = {
        CAUSE_COL:         matched_cause,
        CORRIDOR_COL:      matched_corr,
        'closure_binary':  cl_val,
        'hour_of_day':     hour_of_day,
        'day_of_week':     day_of_week,
        'duration_clean':  min(dur_est, 1440),
        'priority_numeric': priority_n,
        TRUST_COL:         0.7,
        'month':           3,
    }
    return pd.DataFrame([row])


def simulate_event(
    event_type: str,
    corridor: str,
    day_of_week: int,
    hour_of_day: int,
    expected_crowd_size: int,
    requires_closure: bool | None = None,
    counterfactual_overrides: dict | None = None,
) -> dict:
    def _run_once(
        ev_type: str, corr: str, dow: int, hod: int, crowd: int,
        cl: bool | None, cf_label: str
    ) -> dict:
        sc_df = _build_scenario_row(ev_type, corr, dow, hod, crowd, cl)
        matched_cause = sc_df.iloc[0][CAUSE_COL]
        matched_corr  = sc_df.iloc[0][CORRIDOR_COL]

        # Compute similarity row (1 x n_db)
        try:
            H_sc, G_sc = _build_hybrid_gower_matrices(sc_df, df_all)
            SIM_sc = (0.60 * H_sc + 0.40 * G_sc)[0]
        except Exception as e:
            print(f"    WARNING: similarity failed for {cf_label}: {e}")
            SIM_sc = np.full(n_db, 0.5, dtype=np.float32)

        # Two-stage retrieval
        s1_idx  = np.array(sorted(stage1_pos_set))
        s1_sim  = SIM_sc[s1_idx]
        n_s1_g  = int(np.sum(s1_sim >= SIM_THRESHOLD))
        if n_s1_g >= 3:
            top_k_pos = s1_idx[np.argsort(-s1_sim)[:K_RETRIEVAL]]
            sc_stage  = 'stage1_only'
        else:
            top_k_pos = np.argsort(-SIM_sc)[:K_RETRIEVAL]
            n_s1_tk   = sum(1 for j in top_k_pos if j in stage1_pos_set)
            sc_stage  = 'stage1+stage2' if n_s1_tk > 0 else 'stage2_only'

        top_k_sims = SIM_sc[top_k_pos]
        W_tot      = max(float(top_k_sims.sum()), 1e-12)
        w          = top_k_sims / W_tot
        dur_vals_s = np.array([float(df_all.iloc[int(j)].get('duration_clean', GLOBAL_DUR_MEDIAN) or GLOBAL_DUR_MEDIAN) for j in top_k_pos])
        cl_vals_s  = np.array([float(df_all.iloc[int(j)].get('closure_binary', 0) or 0) for j in top_k_pos])

        crowd_factor_s  = min(math.log1p(crowd / 5000), 2.0) if crowd > 0 else 1.0
        p50_base        = weighted_quantile(dur_vals_s.tolist(), top_k_sims.tolist(), q=0.50)
        p80_base        = weighted_quantile(dur_vals_s.tolist(), top_k_sims.tolist(), q=0.80)
        adj_p50         = p50_base * max(1.0, crowd_factor_s)
        adj_p80         = p80_base * max(1.0, crowd_factor_s)
        closure_prob_s  = float(np.dot(w, cl_vals_s))
        if cl is not None:
            closure_prob_s = 0.95 if cl else 0.05
        gp_cl    = _graph_prior_closure(matched_cause, matched_corr)
        fin_cl   = round(0.3 * gp_cl + 0.7 * closure_prob_s, 4)

        # Resources
        sc_l3 = _l3_resources(None, matched_corr, 'Moderate')
        conf_m = confidence_level(float(top_k_sims.mean()))
        n_mean_s = int(np.sum(SIM_sc >= IMS_THRESHOLD))
        ims_s  = math.log1p(n_mean_s) * float(SIM_sc[SIM_sc >= IMS_THRESHOLD].mean()) if n_mean_s > 0 else 0.0
        if ims_s >= 1.5:   ev_w_s = 0.70
        elif ims_s >= 0.8: ev_w_s = 0.50
        elif ims_s >= 0.3: ev_w_s = 0.30
        else:              ev_w_s = 0.10

        hist_off_s = weighted_median([float(df_all.iloc[int(j)].get('priority_numeric', 2) or 2) * 3 for j in top_k_pos], top_k_sims.tolist())
        hist_off_s = max(2, min(25, round(hist_off_s)))
        fin_off  = max(2, min(25, round(ev_w_s * hist_off_s + (1 - ev_w_s) * sc_l3['officers'])))
        fin_tow  = max(0, min(10, round(ev_w_s * sc_l3['tow'] + (1 - ev_w_s) * sc_l3['tow'])))
        fin_bar  = max(0, min(40, round(ev_w_s * sc_l3['barricades'] + (1 - ev_w_s) * sc_l3['barricades'])))
        clearance = adj_p50 * (1 - (1 - math.exp(-0.08 * fin_off)) * 0.4)

        return {
            'label':                  cf_label,
            'matched_cause':          matched_cause,
            'matched_corridor':       matched_corr,
            'retrieval_stage':        sc_stage,
            'confidence':             conf_m,
            'mean_top_k_sim':         round(float(top_k_sims.mean()), 4),
            'ims_score':              round(ims_s, 4),
            'predicted_duration_p50': round(adj_p50, 1),
            'predicted_duration_p80': round(adj_p80, 1),
            'closure_probability':    round(closure_prob_s, 4),
            'graph_prior_closure':    gp_cl,
            'final_closure_probability': fin_cl,
            'crowd_factor':           round(crowd_factor_s, 3),
            'final_officers':         fin_off,
            'final_tow':              fin_tow,
            'final_barricades':       fin_bar,
            'estimated_clearance_min': round(clearance, 1),
            'retrieved_events': [
                {
                    'rank':       r + 1,
                    'event_id':   df_all.iloc[int(j)]['event_id'],
                    'cause':      str(df_all.iloc[int(j)][CAUSE_COL]),
                    'corridor':   str(df_all.iloc[int(j)][CORRIDOR_COL]),
                    'sim':        round(float(SIM_sc[j]), 4),
                }
                for r, j in enumerate(top_k_pos)
            ],
        }

    baseline = _run_once(event_type, corridor, day_of_week, hour_of_day, expected_crowd_size, requires_closure, 'baseline')

    result = {
        'scenario_name':   f"{event_type}_{corridor}",
        'inputs': {
            'event_type':           event_type,
            'corridor':             corridor,
            'day_of_week':          day_of_week,
            'hour_of_day':          hour_of_day,
            'expected_crowd_size':  expected_crowd_size,
            'requires_closure':     requires_closure,
        },
        'baseline': baseline,
    }

    if counterfactual_overrides:
        cf_event   = counterfactual_overrides.get('event_type',           event_type)
        cf_corr    = counterfactual_overrides.get('corridor',             corridor)
        cf_dow     = counterfactual_overrides.get('day_of_week',          day_of_week)
        cf_hod     = counterfactual_overrides.get('hour_of_day',          hour_of_day)
        cf_crowd   = counterfactual_overrides.get('expected_crowd_size',  expected_crowd_size)
        cf_cl      = counterfactual_overrides.get('requires_closure',     requires_closure)
        cf_res     = _run_once(cf_event, cf_corr, cf_dow, cf_hod, cf_crowd, cf_cl, 'counterfactual')
        result['counterfactual_inputs']  = counterfactual_overrides
        result['counterfactual']         = cf_res
        result['counterfactual_comparison'] = {
            'delta_duration_p50':   round(cf_res['predicted_duration_p50'] - baseline['predicted_duration_p50'], 1),
            'delta_closure_prob':   round(cf_res['closure_probability'] - baseline['closure_probability'], 4),
            'delta_officers':       cf_res['final_officers'] - baseline['final_officers'],
            'delta_clearance_min':  round(cf_res['estimated_clearance_min'] - baseline['estimated_clearance_min'], 1),
        }

    return result


# ── Baseline scenarios ────────────────────────────────────────────────────────
SCENARIOS = [
    dict(event_type='political rally',   corridor='Silk Board',               day_of_week=5, hour_of_day=17, expected_crowd_size=20000, requires_closure=True),
    dict(event_type='festival',          corridor='MG Road',                  day_of_week=6, hour_of_day=19, expected_crowd_size=50000, requires_closure=True),
    dict(event_type='vehicle breakdown', corridor='Hebbal',                   day_of_week=1, hour_of_day=8,  expected_crowd_size=0,     requires_closure=False),
    dict(event_type='construction',      corridor='Outer Ring Road',          day_of_week=2, hour_of_day=10, expected_crowd_size=0,     requires_closure=True),
    dict(event_type='sports event',      corridor='Kanteerava',               day_of_week=5, hour_of_day=15, expected_crowd_size=30000, requires_closure=False),
]
CF_SCENARIOS = [
    dict(event_type='political rally', corridor='Silk Board', day_of_week=5, hour_of_day=17, expected_crowd_size=20000, requires_closure=True,  counterfactual_overrides={'expected_crowd_size': 40000}),
    dict(event_type='festival',        corridor='MG Road',    day_of_week=6, hour_of_day=19, expected_crowd_size=50000, requires_closure=True,  counterfactual_overrides={'requires_closure': False}),
    dict(event_type='construction',    corridor='Outer Ring Road', day_of_week=2, hour_of_day=10, expected_crowd_size=0, requires_closure=True,  counterfactual_overrides={'hour_of_day': 7}),
]

all_scenarios = []
print("\n  --- Baseline Scenarios ---")
for sc in SCENARIOS:
    res = simulate_event(**sc)
    b   = res['baseline']
    print(f"  {res['scenario_name'][:50]:50s} | dur={b['predicted_duration_p50']:5.0f} min | "
          f"cl={b['closure_probability']:.2f} | off={b['final_officers']} | conf={b['confidence']}")
    all_scenarios.append(res)

print("\n  --- Counterfactual Scenarios ---")
for sc in CF_SCENARIOS:
    res = simulate_event(**sc)
    b   = res['baseline']
    c   = res.get('counterfactual', {})
    cmp = res.get('counterfactual_comparison', {})
    print(f"  {res['scenario_name'][:45]:45s} | baseline_dur={b['predicted_duration_p50']:.0f} | "
          f"cf_dur={c.get('predicted_duration_p50',0):.0f} | delta={cmp.get('delta_duration_p50',0):+.0f}")
    all_scenarios.append(res)

with open(OUT / 'layer4_scenario_simulations.json', 'w', encoding='utf-8') as f:
    json.dump(all_scenarios, f, indent=2, default=_js)
print(f"  Saved: layer4_scenario_simulations.json ({len(all_scenarios)} scenarios)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14: EVENT KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 14: Event Knowledge Base")

print("  Building knowledge base entries...")
kb_entries = []
for i, ev_row in df_all.iterrows():
    eid    = ev_row['event_id']
    is_p   = bool(ev_row.get(PLANNED_COL, False))

    entry: dict = {
        'event_id':              eid,
        'event_cause':           str(ev_row.get(CAUSE_COL, '')),
        'corridor':              str(ev_row.get(CORRIDOR_COL, '')),
        'junction':              str(ev_row.get(JUNCTION_COL, '')) if pd.notna(ev_row.get(JUNCTION_COL)) else '',
        'day_of_week':           int(ev_row.get('day_of_week', 0) or 0),
        'hour_of_day':           int(ev_row.get('hour_of_day', 0) or 0),
        'actual_duration_min':   float(ev_row.get('duration_clean', GLOBAL_DUR_MEDIAN) or GLOBAL_DUR_MEDIAN),
        'trust_score':           round(float(ev_row.get(TRUST_COL, 0.7) or 0.7), 4),
        'priority':              str(ev_row.get(PRIORITY_COL, 'Low')),
        'requires_road_closure': bool(int(ev_row.get('closure_binary', 0) or 0)),
        'is_planned_event':      is_p,
        'month':                 int(ev_row.get('month', 1) or 1),
    }

    if is_p:
        oc  = outcome_lu.get(eid, {})
        im  = ims_lu.get(eid, {})
        ef  = evid_lu.get(eid, {})
        rr  = rr_by_qid.get(eid, {})
        entry.update({
            'retrieval_stage':           str(rr.get('retrieval_stage', '')),
            'confidence_score':          round(float(im.get('confidence_score', 0)), 4),
            'confidence_level':          str(im.get('confidence_level', '')),
            'ims_score':                 round(float(im.get('ims_score', 0)), 4),
            'ims_interpretation':        str(im.get('ims_interpretation', '')),
            'diversity_score':           round(float(im.get('diversity_score', 0)), 4),
            'diversity_interpretation':  str(im.get('diversity_interpretation', '')),
            'predicted_duration_p50':    round(float(oc.get('predicted_duration_p50', GLOBAL_DUR_MEDIAN)), 1),
            'predicted_duration_p80':    round(float(oc.get('predicted_duration_p80', GLOBAL_DUR_MEDIAN * 1.3)), 1),
            'closure_probability':       round(float(oc.get('closure_probability', 0)), 4),
            'congestion_multiplier':     round(float(oc.get('congestion_multiplier_estimate', 1.0)), 3),
            'impact_score':              round(float(oc.get('impact_score', 0)), 2),
            'impact_level':              str(oc.get('impact_level', '')),
            'final_officers':            int(ef.get('final_officers', 4)),
            'final_barricades':          int(ef.get('final_barricades', 8)),
            'final_tow':                 int(ef.get('final_tow', 0)),
            'evidence_weight':           round(float(ef.get('evidence_weight', 0.5)), 2),
            'top_similar_event_id':      str(df_all.iloc[int(rr['top_k_indices'][0])]['event_id']) if rr.get('top_k_indices') else '',
            'top_similarity_score':      round(float(rr['top_k_similarities'][0]), 4) if rr.get('top_k_similarities') else 0.0,
            'feature_contributions_top_match': rr.get('feature_contributions_best', {}),
        })

    kb_entries.append(entry)

print(f"  Built {len(kb_entries)} entries")

# Corridor summaries
corr_summaries: dict[str, dict] = {}
for corr_v, grp in df_all.groupby(CORRIDOR_COL):
    dur_c   = grp['duration_clean'].dropna()
    cl_rate = float(grp['closure_binary'].fillna(0).mean())
    n_pl    = int(grp[PLANNED_COL].sum())
    pl_grp  = [e for e in grp['event_id'] if e in outcome_lu]
    corr_summaries[str(corr_v)] = {
        'n_events':          len(grp),
        'mean_duration_min': round(float(dur_c.mean()), 1) if len(dur_c) else 0.0,
        'closure_rate':      round(cl_rate, 3),
        'top_causes':        grp[CAUSE_COL].value_counts().head(3).to_dict(),
        'n_planned':         n_pl,
        'avg_impact_score':  round(float(np.mean([outcome_lu[e]['impact_score'] for e in pl_grp])), 2) if pl_grp else 0.0,
        'avg_ims':           round(float(np.mean([ims_lu[e]['ims_score'] for e in pl_grp if e in ims_lu])), 4) if pl_grp else 0.0,
    }

# Cause summaries
cause_summaries: dict[str, dict] = {}
for cause_v, grp in df_all.groupby(CAUSE_COL):
    dur_ca  = grp['duration_clean'].dropna()
    pl_grp  = [e for e in grp['event_id'] if e in outcome_lu]
    cause_summaries[str(cause_v)] = {
        'n_events':          len(grp),
        'mean_duration_min': round(float(dur_ca.mean()), 1) if len(dur_ca) else 0.0,
        'n_planned':         int(grp[PLANNED_COL].sum()),
        'avg_confidence':    round(float(np.mean([conf_lu[e][0] for e in pl_grp if e in conf_lu])), 4) if pl_grp else 0.0,
        'retrieval_stage_distribution': dict(
            pd.Series([rr_by_qid[e]['retrieval_stage'] for e in pl_grp if e in rr_by_qid]).value_counts().to_dict()
        ) if pl_grp else {},
    }

# Graph summary
out_dist = {on: sum(1 for qi in range(n_q) if on in _outcome_nodes(df_planned.iloc[qi]['event_id'])) for on in OUTCOME_NODES}
top_corr_nodes = sorted(
    [(n, G.in_degree(n)) for n, d in G.nodes(data=True) if d.get('node_type') == 'corridor'],
    key=lambda x: -x[1]
)[:5]

kb = {
    'generated_at':         datetime.utcnow().isoformat() + 'Z',
    'layer_version':        '4.3-upgraded',
    'total_events':         len(kb_entries),
    'planned_events':       sum(1 for e in kb_entries if e['is_planned_event']),
    'retrieval_method':     'Two-Stage: Planned-Event-First (event_type==planned) + Full-Fallback',
    'similarity_method':    'Gower (40%) + Domain-Weighted Hybrid (60%)',
    'knowledge_entries':    kb_entries,
    'corridor_summaries':   corr_summaries,
    'cause_summaries':      cause_summaries,
    'graph_summary': {
        'n_nodes':                  n_nodes,
        'n_edges':                  n_edges,
        'top_connected_corridors':  [c for c, _ in top_corr_nodes],
        'outcome_distribution':     out_dist,
    },
}

with open(OUT / 'layer4_event_knowledge_base.json', 'w', encoding='utf-8') as f:
    json.dump(kb, f, separators=(',', ':'), default=_js)

kb_size = (OUT / 'layer4_event_knowledge_base.json').stat().st_size
print(f"  Knowledge base: {len(kb_entries)} entries | {kb_size // 1024} KB")
print(f"  Saved: layer4_event_knowledge_base.json")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15: FINAL VALIDATION AND SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 15: Final Validation and Summary")

required_outputs = [
    'layer4_event_features.csv',
    'layer4_event_index_metadata.csv',
    'layer4_encoders.pkl',
    'layer4_nn_index.pkl',
    'layer4_retrieval_results.csv',
    'layer4_gower_similarity_sample.csv',
    'layer4_similarity_weights.json',
    'layer4_institutional_memory_scores.csv',
    'layer4_event_outcome_predictions.csv',
    'layer4_evidence_based_recommendations.csv',
    'layer4_knowledge_graph.gpickle',
    'layer4_knowledge_graph_summary.csv',
    'layer4_explainable_recommendations.txt',
    'layer4_explainable_recommendations.json',
    'layer4_scenario_simulations.json',
    'layer4_event_knowledge_base.json',
]

all_ok = True
for fname in required_outputs:
    p = OUT / fname
    if p.exists():
        sz = p.stat().st_size
        suffix = ''
        if fname.endswith('.csv'):
            rc = len(pd.read_csv(p))
            suffix = f'{rc} rows | '
        print(f"  [OK] {fname}: {suffix}{sz // 1024} KB")
    else:
        print(f"  [MISSING] {fname}")
        all_ok = False

print(f"\n=== LAYER 4 COMPLETE ===")
print(f"Planned events processed: {n_q}")
print(f"Retrieval method: Two-Stage Gower+Hybrid")
print(f"Confidence distribution: {conf_dist}")
print(f"IMS distribution: {ims_dist}")
print(f"Knowledge graph: {n_nodes} nodes, {n_edges} edges")
print(f"All required outputs present: {all_ok}")
