"""
Layer 4 — Planned Event Prototype Retrieval
ASTraM Bengaluru Traffic Disruption Intelligence

New additive module. Does NOT modify any existing Layer 1/2/3/4 file.
Reads: data/events_clean.parquet, outputs/layer2_operational_burden_index.csv

Writes:
  outputs/layer4_planned_event_prototypes.csv
  outputs/layer4_planned_event_retrieval.csv
  outputs/layer4_retrieval_feature_weights.json
  outputs/layer4_retrieval_encoders.pkl
  outputs/layer4_retrieval_diagnostics.csv
  outputs/layer4_prototype_utilization.csv
  outputs/layer4_example_retrievals.json
  outputs/layer4_simulation_demos.json

Design:
  Prototype compression: 191 planned events → K KMeans prototypes
  Shrinkage weights:     w_k = rho*IG_k/sum_IG + (1-rho)/P, rho = n/(n+eta)
  Trust-weighted sim:    s(q,p) = exp(-d_G(q,p) / h) * tau_p
  Effective sample size: n_eff = (sum s_i)^2 / sum s_i^2
  Confidence:            Conf = min(1, n_eff/k0) * mean_sim
  Abstention:            max_sim < S_MIN or n_eff < N_EFF_MIN
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(42)
warnings.filterwarnings('ignore')

OUTPUTS   = Path('outputs')
DATA      = Path('data')

K_NEIGHBORS   = 5      # top-k prototypes for retrieval
K0            = 3.0    # ESS threshold for confidence normalisation
S_MIN         = 0.15   # minimum similarity to avoid abstention
N_EFF_MIN     = 3.0    # minimum ESS to avoid abstention
H_BANDWIDTH   = 0.5    # kernel bandwidth
ETA           = 10.0   # shrinkage strength for feature weight learning
RHO_DEFAULT   = 0.5    # fallback rho if IG undefined


def safe_load(path, **kwargs):
    try:
        if str(path).endswith('.csv'):
            df = pd.read_csv(path, **kwargs)
        else:
            df = pd.read_parquet(path, **kwargs)
        print(f'  Loaded {path}: {df.shape}')
        return df
    except Exception as e:
        print(f'  WARNING cannot load {path}: {e}')
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
print("=== SECTION 1: DATA PREPARATION ===")

df_raw = safe_load(DATA / 'events_clean.parquet')
df_all = df_raw.copy().reset_index(drop=True)

CAUSE_COL    = 'event_cause'
CORRIDOR_COL = 'corridor'
CLOSURE_COL  = 'requires_road_closure'
DURATION_COL = 'duration_min'
PRIORITY_COL = 'priority'
TRUST_COL    = 'trust_score'
PLANNED_COL  = 'is_true_planned_event'
START_COL    = 'start_local'

# Parse timestamps
df_all[START_COL] = pd.to_datetime(df_all[START_COL], errors='coerce')
df_all['hour_of_day'] = df_all[START_COL].dt.hour.fillna(12).astype(int)
df_all['day_of_week'] = df_all[START_COL].dt.dayofweek.fillna(2).astype(int)
df_all['month']       = df_all[START_COL].dt.month.fillna(3).astype(int)

# Closure binary
df_all['closure_binary'] = df_all[CLOSURE_COL].astype(bool).astype(int)

# Priority numeric: High=3, Low=1, Unknown/NaN=2
PRIORITY_MAP = {'High': 3, 'Low': 1, 'Unknown': 2}
prio_str = df_all[PRIORITY_COL].astype(str).fillna('Unknown')
df_all['priority_numeric'] = prio_str.map(PRIORITY_MAP).fillna(2.0)
med_priority = float(df_all['priority_numeric'].median())

# Duration
dur_med = float(df_all[DURATION_COL].clip(lower=0, upper=1440).dropna().median())
df_all['duration_clean'] = (
    df_all[DURATION_COL].clip(lower=0, upper=1440).fillna(dur_med)
)

# Trust
df_all['trust_clean'] = (
    pd.to_numeric(df_all[TRUST_COL], errors='coerce').clip(0.1, 1.0).fillna(0.7)
)

# Identify planned events
planned_mask = df_all[PLANNED_COL] == True
n_primary    = int(planned_mask.sum())

if n_primary >= 10:
    print(f'  Using PLANNED_COL primary flag: {n_primary} planned events')
else:
    print(f'  PLANNED_COL gave only {n_primary} rows; using cause-keyword fallback')
    kw = ['rally', 'festival', 'procession', 'vip', 'protest', 'event',
          'celebr', 'march', 'gather', 'sports', 'concert', 'parade',
          'public_event']
    planned_mask = df_all[CAUSE_COL].str.lower().str.contains('|'.join(kw), na=False)
    print(f'  Keyword fallback: {planned_mask.sum()} planned events')

df_planned = df_all[planned_mask].copy().reset_index(drop=True)
n_planned  = len(df_planned)
print(f'  Planned events: {n_planned}')
print(f'  Cause dist: {df_planned[CAUSE_COL].value_counts().to_dict()}')
print(f'  Date range: {df_planned[START_COL].min()} – {df_planned[START_COL].max()}')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION 2: FEATURE ENGINEERING ===")

CATEGORICAL_COLS = [CAUSE_COL, CORRIDOR_COL]
BINARY_COLS      = ['closure_binary']
CONTINUOUS_COLS  = ['hour_of_day', 'day_of_week', 'duration_clean', 'priority_numeric', 'month']
ALL_COLS         = CATEGORICAL_COLS + BINARY_COLS + CONTINUOUS_COLS

# Fill NaN in continuous columns on both frames
for col in CONTINUOUS_COLS:
    med_v = float(pd.to_numeric(df_all[col], errors='coerce').dropna().median())
    df_all[col]     = pd.to_numeric(df_all[col],     errors='coerce').fillna(med_v)
    df_planned[col] = pd.to_numeric(df_planned[col], errors='coerce').fillna(med_v)

# Label-encode categoricals (fit on full dataset)
label_encoders: dict[str, LabelEncoder] = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    le.fit(df_all[col].fillna('__NA__').astype(str))
    df_all[f'{col}_enc']     = le.transform(df_all[col].fillna('__NA__').astype(str))
    df_planned[f'{col}_enc'] = le.transform(df_planned[col].fillna('__NA__').astype(str))
    label_encoders[col] = le

# Feature ranges for Gower (on full dataset)
feature_ranges: dict[str, float] = {}
for col in CONTINUOUS_COLS + BINARY_COLS:
    vals = pd.to_numeric(df_all[col], errors='coerce').dropna()
    feature_ranges[col] = max(float(vals.max() - vals.min()), 1e-6)

# Shrinkage feature weights via Information Gain (correlation proxy)
def compute_ig(feature_series: pd.Series, target_series: pd.Series) -> float:
    try:
        if feature_series.nunique() <= 1:
            return 0.0
        if feature_series.dtype == object or str(feature_series.dtype).startswith('str'):
            encoded = LabelEncoder().fit_transform(feature_series.fillna('__NA__').astype(str))
        else:
            encoded = pd.to_numeric(feature_series, errors='coerce').fillna(0).to_numpy()
        target  = pd.to_numeric(target_series, errors='coerce').fillna(0).to_numpy()
        if len(encoded) < 2 or np.std(encoded) < 1e-9 or np.std(target) < 1e-9:
            return 0.0
        c = float(np.corrcoef(encoded.astype(float), target.astype(float))[0, 1])
        return abs(c) if not np.isnan(c) else 0.0
    except Exception:
        return 0.0

ig_scores: dict[str, float] = {}
for col in ALL_COLS:
    ig_scores[col] = compute_ig(df_planned[col], df_planned['duration_clean'])

ig_total = max(sum(ig_scores.values()), 1e-6)
rho      = n_planned / (n_planned + ETA)
print(f'  Shrinkage rho = {rho:.4f} (n={n_planned}, eta={ETA})')

feature_weights: dict[str, float] = {}
for col in ALL_COLS:
    ig_w   = ig_scores[col] / ig_total
    unif_w = 1.0 / len(ALL_COLS)
    feature_weights[col] = rho * ig_w + (1 - rho) * unif_w

w_total = sum(feature_weights.values())
feature_weights = {k: v / w_total for k, v in feature_weights.items()}

print('  Feature weights (descending):')
for col, w in sorted(feature_weights.items(), key=lambda x: -x[1]):
    print(f'    {col:25s}: {w:.4f}  (IG={ig_scores[col]:.4f})')

# Save feature weights
with open(OUTPUTS / 'layer4_retrieval_feature_weights.json', 'w', encoding='utf-8') as f:
    json.dump({'rho': rho, 'eta': ETA, 'n_planned': n_planned,
               'ig_scores': ig_scores, 'feature_weights': feature_weights}, f, indent=2)
print('  Saved: layer4_retrieval_feature_weights.json')

# Save encoders
with open(OUTPUTS / 'layer4_retrieval_encoders.pkl', 'wb') as f:
    pickle.dump({
        'label_encoders': label_encoders,
        'feature_ranges': feature_ranges,
        'feature_weights': feature_weights,
        'ALL_COLS': ALL_COLS,
        'CATEGORICAL_COLS': CATEGORICAL_COLS,
        'BINARY_COLS': BINARY_COLS,
        'CONTINUOUS_COLS': CONTINUOUS_COLS,
        'CAUSE_COL': CAUSE_COL,
        'CORRIDOR_COL': CORRIDOR_COL,
    }, f)
print('  Saved: layer4_retrieval_encoders.pkl')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PROTOTYPE COMPRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION 3: PROTOTYPE COMPRESSION ===")

if n_planned < 40:
    n_clusters = max(5, n_planned // 3)
else:
    n_clusters = max(10, min(50, n_planned // 4))

print(f'  {n_planned} planned events → {n_clusters} prototypes')

# Build numeric matrix for KMeans clustering
enc_cols  = [f'{col}_enc' for col in CATEGORICAL_COLS]
cont_cols = BINARY_COLS + CONTINUOUS_COLS

X_parts = []
for col in enc_cols:
    v = df_planned[col].to_numpy(dtype=np.float64)
    X_parts.append(v.reshape(-1, 1))
for col in cont_cols:
    v  = pd.to_numeric(df_planned[col], errors='coerce').fillna(0).to_numpy(dtype=np.float64)
    r  = feature_ranges.get(col, 1.0)
    mn = float(df_all[col].min())
    X_parts.append(((v - mn) / r).reshape(-1, 1))

X_planned_num = np.hstack(X_parts)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_planned_num)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

prototypes: list[dict] = []
for k in range(n_clusters):
    mask = labels == k
    ce   = df_planned[mask].copy()
    if len(ce) == 0:
        continue

    X_k = X_scaled[mask]
    dists_to_centroid = np.linalg.norm(X_k - kmeans.cluster_centers_[k], axis=1)
    medoid_pos = int(np.argmin(dists_to_centroid))

    # Store all ALL_COLS feature values for Gower distance
    proto: dict = {
        'prototype_id':            k,
        'support_count':           int(len(ce)),
        'representative_event_id': str(ce.iloc[medoid_pos]['event_id']),
        'cause':                   str(ce[CAUSE_COL].mode()[0]),
        'corridor':                str(ce[CORRIDOR_COL].mode()[0]),
        'median_duration':         float(ce['duration_clean'].median()),
        'p80_duration':            float(ce['duration_clean'].quantile(0.80)),
        'p95_duration':            float(ce['duration_clean'].quantile(0.95)),
        'mean_severity':           float(ce['priority_numeric'].mean()),
        'closure_rate':            float(ce['closure_binary'].mean()),
        'trust_mean':              float(ce['trust_clean'].mean()),
        'hour_mode':               int(ce['hour_of_day'].mode()[0]),
        'dow_mode':                int(ce['day_of_week'].mode()[0]),
        'cause_diversity':         int(ce[CAUSE_COL].nunique()),
        'corridor_diversity':      int(ce[CORRIDOR_COL].nunique()),
        'outcome_summary':         (f"Duration: {ce['duration_clean'].median():.0f}min, "
                                    f"Closure: {ce['closure_binary'].mean():.0%}"),
    }
    # Store representative values for ALL feature columns (for Gower)
    for col in CATEGORICAL_COLS:
        proto[col] = str(ce[col].mode()[0])
    for col in BINARY_COLS:
        proto[col] = int(round(float(ce[col].mean())))
    for col in CONTINUOUS_COLS:
        proto[col] = float(ce[col].median())

    prototypes.append(proto)

prototypes_df = pd.DataFrame(prototypes)
# Save CSV with just the summary columns (not all feature duplicates)
save_cols = ['prototype_id', 'support_count', 'representative_event_id', 'cause', 'corridor',
             'median_duration', 'p80_duration', 'p95_duration', 'mean_severity',
             'closure_rate', 'trust_mean', 'hour_mode', 'dow_mode',
             'cause_diversity', 'corridor_diversity', 'outcome_summary']
prototypes_df[save_cols].to_csv(OUTPUTS / 'layer4_planned_event_prototypes.csv', index=False)
print(f'  Saved: layer4_planned_event_prototypes.csv ({len(prototypes_df)} prototypes)')

supp = prototypes_df['support_count']
print(f'  Prototype support: min={supp.min()}, median={supp.median():.0f}, max={supp.max()}')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: GOWER DISTANCE AND TRUST-WEIGHTED SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION 4: GOWER + TRUST SIMILARITY ===")


def gower_distance(
    query_row: dict,
    proto_row: dict,
    feature_weights: dict[str, float],
    feature_ranges: dict[str, float],
    categorical_cols: list[str],
    binary_cols: list[str],
    continuous_cols: list[str],
) -> float:
    total_weight   = 0.0
    total_distance = 0.0

    for col in categorical_cols + binary_cols + continuous_cols:
        w     = feature_weights.get(col, 1.0 / max(len(feature_weights), 1))
        q_val = query_row.get(col)
        p_val = proto_row.get(col)

        if q_val is None or p_val is None:
            continue
        if isinstance(q_val, float) and np.isnan(q_val):
            continue
        if isinstance(p_val, float) and np.isnan(p_val):
            continue

        if col in categorical_cols or col in binary_cols:
            d_k = 0.0 if str(q_val) == str(p_val) else 1.0
        else:
            r   = feature_ranges.get(col, 1.0)
            d_k = min(1.0, abs(float(q_val) - float(p_val)) / r)

        total_distance += w * d_k
        total_weight   += w

    if total_weight < 1e-9:
        return 1.0
    return total_distance / total_weight


def trust_weighted_similarity(gower_dist: float, trust_score: float, bandwidth: float = H_BANDWIDTH) -> float:
    return float(np.exp(-gower_dist / bandwidth) * trust_score)


def retrieve_top_k(
    query_features: dict,
    prototypes_df: pd.DataFrame,
    exclude_event_id: str | None = None,
    k: int = K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (top_k_row_indices_in_prototypes_df, similarities, gower_dists)."""
    sims = np.zeros(len(prototypes_df))
    gdists = np.zeros(len(prototypes_df))

    for i, (_, proto) in enumerate(prototypes_df.iterrows()):
        # Skip if prototype's representative is the query itself
        if exclude_event_id and str(proto.get('representative_event_id', '')) == str(exclude_event_id):
            sims[i]   = -1.0
            gdists[i] = 1.0
            continue
        gd      = gower_distance(query_features, proto.to_dict(), feature_weights,
                                  feature_ranges, CATEGORICAL_COLS, BINARY_COLS, CONTINUOUS_COLS)
        trust   = float(proto.get('trust_mean', 0.7))
        sims[i]   = trust_weighted_similarity(gd, trust)
        gdists[i] = gd

    top_k_idx = np.argsort(sims)[::-1][:k]
    return top_k_idx, sims[top_k_idx], gdists[top_k_idx]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: EFFECTIVE SAMPLE SIZE AND CONFIDENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION 5: EFFECTIVE SAMPLE SIZE ===")


def effective_sample_size(similarities: np.ndarray) -> float:
    sim_pos = similarities[similarities > 0]
    if len(sim_pos) == 0:
        return 0.0
    ss1 = float(np.sum(sim_pos))
    ss2 = float(np.sum(sim_pos ** 2))
    if ss2 < 1e-12:
        return 0.0
    return (ss1 ** 2) / ss2


def confidence_score(similarities: np.ndarray, k0: float = K0) -> tuple[float, float, float]:
    if len(similarities) == 0:
        return 0.0, 0.0, 0.0
    n_eff    = effective_sample_size(similarities)
    mean_sim = float(np.mean(similarities[similarities > 0])) if np.any(similarities > 0) else 0.0
    conf     = min(1.0, n_eff / k0) * mean_sim
    return float(conf), float(n_eff), float(mean_sim)


def should_abstain(similarities: np.ndarray, s_min: float = S_MIN, n_eff_min: float = N_EFF_MIN) -> tuple[bool, str]:
    if len(similarities) == 0:
        return True, 'no_prototypes_found'
    valid = similarities[similarities > 0]
    if len(valid) == 0:
        return True, 'all_similarities_zero'
    max_sim = float(np.max(valid))
    n_eff   = effective_sample_size(valid)
    if max_sim < s_min:
        return True, f'max_similarity={max_sim:.3f}<{s_min}'
    if n_eff < n_eff_min:
        return True, f'n_eff={n_eff:.2f}<{n_eff_min}'
    return False, 'sufficient_evidence'


print(f'  Abstention thresholds: S_MIN={S_MIN}, N_EFF_MIN={N_EFF_MIN}, K0={K0}')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: WEIGHTED OUTCOME PREDICTION + OBI LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION 6: WEIGHTED OUTCOME PREDICTION ===")


def predict_outcomes(top_k_idx: np.ndarray, similarities: np.ndarray, proto_df: pd.DataFrame) -> dict | None:
    protos = proto_df.iloc[top_k_idx]
    W      = float(similarities.sum())
    if W < 1e-9:
        return None
    w = similarities / W

    return {
        'predicted_duration_median': float(np.dot(w, protos['median_duration'].values)),
        'predicted_duration_p80':    float(np.dot(w, protos['p80_duration'].values)),
        'predicted_duration_p95':    float(np.dot(w, protos['p95_duration'].values)),
        'predicted_closure_probability': float(np.dot(w, protos['closure_rate'].values)),
        'predicted_severity':        float(np.dot(w, protos['mean_severity'].values)),
    }


# OBI for delta_OBI: build corridor→mean_OBI via junction lookup
obi_df_raw = safe_load(OUTPUTS / 'layer2_operational_burden_index.csv')
obi_lookup: dict[str, float] = {}

if len(obi_df_raw) > 0 and 'operational_burden_index' in obi_df_raw.columns:
    # OBI is junction-level; bridge to corridor via events_clean
    junc_corr = (
        df_all[df_all['junction'].notna()][['junction', CORRIDOR_COL]]
        .drop_duplicates()
    )
    obi_joined = obi_df_raw[['junction', 'operational_burden_index']].merge(
        junc_corr, on='junction', how='inner'
    )
    obi_lookup = (
        obi_joined.groupby(CORRIDOR_COL)['operational_burden_index']
        .mean()
        .to_dict()
    )
    print(f'  OBI lookup: {len(obi_lookup)} corridors via junction bridge')
else:
    print('  OBI not available; delta_OBI will be NaN')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN RETRIEVAL LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION 7: RETRIEVAL LOOP ===")

retrieval_rows: list[dict] = []

for i, (_, event_row) in enumerate(df_planned.iterrows()):
    if i % 25 == 0:
        print(f'  Processing event {i+1}/{n_planned} ...', flush=True)

    eid  = str(event_row['event_id'])

    # Build query features dict (original string values for categorical, numeric for continuous)
    query_features: dict = {}
    for col in CATEGORICAL_COLS:
        query_features[col] = str(event_row.get(col, '__NA__') or '__NA__')
    for col in BINARY_COLS:
        query_features[col] = int(event_row.get(col, 0) or 0)
    for col in CONTINUOUS_COLS:
        v = pd.to_numeric(event_row.get(col, 0), errors='coerce')
        query_features[col] = float(v) if not pd.isna(v) else 0.0

    top_k_idx, sims, gdists = retrieve_top_k(query_features, prototypes_df, exclude_event_id=eid)
    valid_sims = sims[sims > 0]

    conf, n_eff, mean_sim = confidence_score(sims)
    abstain, abstain_reason = should_abstain(sims)
    outcomes = predict_outcomes(top_k_idx, sims, prototypes_df)

    # Delta OBI
    proto_corridors = prototypes_df.iloc[top_k_idx]['corridor'].tolist()
    obi_vals   = [obi_lookup.get(c) for c in proto_corridors]
    obi_clean  = [v for v in obi_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    w_obi      = [float(sims[j]) for j, c in enumerate(proto_corridors) if obi_lookup.get(c) is not None]
    if obi_clean and w_obi:
        delta_obi = float(np.average(obi_clean, weights=w_obi))
    else:
        delta_obi = float('nan')

    proto_ids_str = ','.join(str(int(prototypes_df.iloc[j]['prototype_id'])) for j in top_k_idx)
    sims_str      = ','.join(f'{s:.4f}' for s in sims)
    gdists_str    = ','.join(f'{g:.4f}' for g in gdists)

    rec = {
        'query_event_id':              eid,
        'cause':                       query_features[CAUSE_COL],
        'corridor':                    query_features[CORRIDOR_COL],
        'hour_of_day':                 int(query_features['hour_of_day']),
        'day_of_week':                 int(query_features['day_of_week']),
        'top_k_prototype_ids':         proto_ids_str,
        'top_k_similarities':          sims_str,
        'top_k_gower_dists':           gdists_str,
        'mean_similarity':             round(float(mean_sim), 4),
        'effective_sample_size':       round(float(n_eff), 3),
        'confidence':                  round(float(conf), 4),
        'abstain_flag':                int(abstain),
        'abstain_reason':              abstain_reason,
        'predicted_duration_median':   round(float(outcomes['predicted_duration_median']), 1) if outcomes else float('nan'),
        'predicted_duration_p80':      round(float(outcomes['predicted_duration_p80']), 1)    if outcomes else float('nan'),
        'predicted_duration_p95':      round(float(outcomes['predicted_duration_p95']), 1)    if outcomes else float('nan'),
        'predicted_severity':          round(float(outcomes['predicted_severity']), 3)         if outcomes else float('nan'),
        'predicted_closure_probability': round(float(outcomes['predicted_closure_probability']), 4) if outcomes else float('nan'),
        'predicted_delta_obi':         round(delta_obi, 4) if not (isinstance(delta_obi, float) and np.isnan(delta_obi)) else float('nan'),
        'actual_duration':             round(float(event_row.get('duration_clean', float('nan'))), 1),
        'trust_score':                 round(float(event_row.get('trust_clean', 0.7)), 4),
    }
    retrieval_rows.append(rec)

retrieval_df = pd.DataFrame(retrieval_rows)
retrieval_df.to_csv(OUTPUTS / 'layer4_planned_event_retrieval.csv', index=False)
print(f'\n  Saved: layer4_planned_event_retrieval.csv ({len(retrieval_df)} rows)')

# Summary
n_abstain = int(retrieval_df['abstain_flag'].sum())
pct_abs   = n_abstain / max(n_planned, 1)
print(f'\n  Events processed: {n_planned}')
print(f'  Abstained: {n_abstain} ({pct_abs:.1%})')
if n_abstain > 0:
    abs_reasons = retrieval_df[retrieval_df['abstain_flag'] == 1]['abstain_reason'].value_counts().to_dict()
    print(f'  Abstain reasons: {abs_reasons}')

confs = retrieval_df['confidence']
print(f'  Confidence: mean={confs.mean():.3f}, min={confs.min():.3f}, max={confs.max():.3f}, '
      f'pct_above_0.5={( confs > 0.5).mean():.1%}')

neffs = retrieval_df['effective_sample_size']
print(f'  n_eff: mean={neffs.mean():.2f}, min={neffs.min():.2f}, pct_>=3={( neffs >= 3.0).mean():.1%}')

# Duration prediction accuracy (non-abstained, non-NaN actual)
pred_rows = retrieval_df[(retrieval_df['abstain_flag'] == 0) &
                          retrieval_df['actual_duration'].notna() &
                          retrieval_df['predicted_duration_median'].notna()].copy()
if len(pred_rows) > 0:
    abs_err  = (pred_rows['predicted_duration_median'] - pred_rows['actual_duration']).abs()
    mae      = float(abs_err.mean())
    within20 = float((abs_err <= 20).mean())
    print(f'  Duration prediction (n={len(pred_rows)}): MAE={mae:.1f} min, within 20 min: {within20:.1%}')

if pct_abs > 0.5:
    print(f'''
  NOTE: {pct_abs:.0%} of planned events triggered abstention due to weak similarity.
  This is the correct behavior given sparse planned event history (n={n_planned}).
  Layer 3 rule-based resources should be used for these events.''')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: RETRIEVAL QUALITY DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION 8: DIAGNOSTICS ===")

# Compute abs_error for diagnostics
retrieval_df['abs_error'] = (
    (retrieval_df['predicted_duration_median'] - retrieval_df['actual_duration']).abs()
)
retrieval_df['rel_error'] = retrieval_df['abs_error'] / retrieval_df['actual_duration'].clip(lower=1.0)

diag_df = retrieval_df[[
    'query_event_id', 'cause', 'corridor', 'confidence', 'effective_sample_size',
    'mean_similarity', 'abstain_flag', 'actual_duration', 'predicted_duration_median',
    'abs_error', 'rel_error',
]].copy()
diag_df.to_csv(OUTPUTS / 'layer4_retrieval_diagnostics.csv', index=False)
print(f'  Saved: layer4_retrieval_diagnostics.csv ({len(diag_df)} rows)')

# Prototype utilization
top1_counts = {k: 0 for k in prototypes_df['prototype_id'].tolist()}
top5_counts = {k: 0 for k in prototypes_df['prototype_id'].tolist()}
sim_when_retrieved: dict[int, list[float]] = {k: [] for k in prototypes_df['prototype_id'].tolist()}

for _, row in retrieval_df.iterrows():
    if not row['top_k_prototype_ids']:
        continue
    ids  = [int(x) for x in str(row['top_k_prototype_ids']).split(',') if x.strip()]
    sims = [float(x) for x in str(row['top_k_similarities']).split(',') if x.strip()]
    for rank, (pid, s) in enumerate(zip(ids, sims)):
        if pid in top5_counts:
            top5_counts[pid] += 1
            sim_when_retrieved.setdefault(pid, []).append(s)
            if rank == 0:
                top1_counts[pid] = top1_counts.get(pid, 0) + 1

util_rows = []
for _, p_row in prototypes_df.iterrows():
    pid = int(p_row['prototype_id'])
    sw  = sim_when_retrieved.get(pid, [])
    util_rows.append({
        'prototype_id':              pid,
        'support_count':             int(p_row['support_count']),
        'times_in_top1':             top1_counts.get(pid, 0),
        'times_in_top5':             top5_counts.get(pid, 0),
        'mean_similarity_retrieved': round(float(np.mean(sw)), 4) if sw else 0.0,
        'representative_cause':      str(p_row['cause']),
        'representative_corridor':   str(p_row['corridor']),
    })

util_df = pd.DataFrame(util_rows)
util_df.to_csv(OUTPUTS / 'layer4_prototype_utilization.csv', index=False)
print(f'  Saved: layer4_prototype_utilization.csv ({len(util_df)} rows)')

used_in_top1 = int((util_df['times_in_top1'] > 0).sum())
used_in_top5 = int((util_df['times_in_top5'] > 0).sum())
print(f'  Prototype coverage: {used_in_top5}/{len(util_df)} used in any top-5 retrieval')
print(f'  Prototype coverage: {used_in_top1}/{len(util_df)} used as top-1')

if len(util_df) > 0 and n_planned > 0:
    top1_proto = util_df.nlargest(1, 'times_in_top1').iloc[0]
    pct_top1   = float(top1_proto['times_in_top1']) / n_planned
    print(f'  Most frequent top-1 prototype: id={int(top1_proto["prototype_id"])}, '
          f'cause={top1_proto["representative_cause"]}, '
          f'times_top1={int(top1_proto["times_in_top1"])} ({pct_top1:.1%} of queries)')
    if pct_top1 > 0.30:
        print(f'  WARNING: Prototype {int(top1_proto["prototype_id"])} accounts for '
              f'{pct_top1:.0%} of top-1 retrievals — potential degenerate retrieval. '
              f'Consider increasing n_clusters or bandwidth.')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: EXAMPLE RETRIEVALS
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION 9: EXAMPLE RETRIEVALS ===")

examples_to_show = []

# 1. Highest confidence
if len(retrieval_df) > 0:
    examples_to_show.append(('Highest confidence', retrieval_df.nlargest(1, 'confidence').index[0]))
# 2. Lowest confidence (non-abstained)
non_abs = retrieval_df[retrieval_df['abstain_flag'] == 0]
if len(non_abs) > 0:
    examples_to_show.append(('Lowest confidence (non-abstained)', non_abs.nsmallest(1, 'confidence').index[0]))
# 3. Highest predicted duration
valid_pred = retrieval_df.dropna(subset=['predicted_duration_median'])
if len(valid_pred) > 0:
    examples_to_show.append(('Highest predicted duration', valid_pred.nlargest(1, 'predicted_duration_median').index[0]))
# 4. Most diverse cause (rarest planned cause)
cause_counts = retrieval_df['cause'].value_counts()
if len(cause_counts) > 1:
    rarest_cause = cause_counts.index[-1]
    rare_rows = retrieval_df[retrieval_df['cause'] == rarest_cause]
    if len(rare_rows) > 0:
        examples_to_show.append(('Rarest cause type', rare_rows.index[0]))
# 5. Best prediction accuracy
good_pred = retrieval_df[
    (retrieval_df['abstain_flag'] == 0) &
    retrieval_df['abs_error'].notna()
]
if len(good_pred) > 0:
    examples_to_show.append(('Best prediction accuracy', good_pred.nsmallest(1, 'abs_error').index[0]))

example_outputs = []
for label, idx in examples_to_show:
    row = retrieval_df.loc[idx]
    ids   = [int(x) for x in str(row['top_k_prototype_ids']).split(',') if x.strip()]
    sims  = [float(x) for x in str(row['top_k_similarities']).split(',') if x.strip()]
    gdsts = [float(x) for x in str(row['top_k_gower_dists']).split(',') if x.strip()]

    conf   = float(row['confidence'])
    n_eff  = float(row['effective_sample_size'])
    msim   = float(row['mean_similarity'])
    abst   = bool(row['abstain_flag'])
    ar     = str(row['abstain_reason'])
    p50    = float(row['predicted_duration_median']) if pd.notna(row['predicted_duration_median']) else float('nan')
    p80    = float(row['predicted_duration_p80'])    if pd.notna(row['predicted_duration_p80']) else float('nan')
    p95    = float(row['predicted_duration_p95'])    if pd.notna(row['predicted_duration_p95']) else float('nan')
    cl_p   = float(row['predicted_closure_probability']) if pd.notna(row['predicted_closure_probability']) else float('nan')
    sev    = float(row['predicted_severity'])        if pd.notna(row['predicted_severity']) else float('nan')
    dobi   = float(row['predicted_delta_obi'])       if pd.notna(row.get('predicted_delta_obi', float('nan'))) else float('nan')
    actual = float(row['actual_duration'])            if pd.notna(row['actual_duration']) else float('nan')
    err    = float(row['abs_error'])                  if pd.notna(row['abs_error']) else float('nan')

    text = (
        f"\nRETRIEVAL EXAMPLE [{label}] -- {row['cause']} at {row['corridor']}\n"
        f"  Confidence: {conf:.3f} | n_eff: {n_eff:.2f} | Mean similarity: {msim:.3f}\n"
        f"  Abstain: {abst} ({ar})\n"
        f"\n  Top prototypes retrieved:\n"
    )
    for rank, (pid, s, gd) in enumerate(zip(ids, sims, gdsts)):
        if pid < len(prototypes_df):
            p = prototypes_df[prototypes_df['prototype_id'] == pid].iloc[0]
            text += (f"    {rank+1}. {p['cause']} @ {p['corridor']} | sim={s:.3f} | "
                     f"gower={gd:.3f} | dur={p['median_duration']:.0f}min | "
                     f"closure={p['closure_rate']:.0%}\n")
    text += (
        f"\n  Prediction:\n"
        f"    Duration: {p50:.0f} min (P50) / {p80:.0f} min (P80) / {p95:.0f} min (P95)\n"
        f"    Closure probability: {cl_p:.0%}\n"
        f"    Severity: {sev:.2f}\n"
        f"    Delta OBI: {dobi:.3f}\n"
        f"\n  Actual duration: {actual:.0f} min  |  Abs error: {err:.0f} min\n"
        f"  {'-'*60}"
    )
    print(text)

    proto_details = []
    for pid, s, gd in zip(ids, sims, gdsts):
        if pid < len(prototypes_df):
            p = prototypes_df[prototypes_df['prototype_id'] == pid].iloc[0]
            proto_details.append({
                'prototype_id': int(pid),
                'cause': str(p['cause']),
                'corridor': str(p['corridor']),
                'similarity': round(s, 4),
                'gower_dist': round(gd, 4),
                'median_duration': float(p['median_duration']),
                'closure_rate': float(p['closure_rate']),
            })

    example_outputs.append({
        'label':              label,
        'query_event_id':     str(row['query_event_id']),
        'cause':              str(row['cause']),
        'corridor':           str(row['corridor']),
        'confidence':         round(conf, 4),
        'n_eff':              round(n_eff, 3),
        'mean_similarity':    round(msim, 4),
        'abstain':            abst,
        'abstain_reason':     ar,
        'prototypes_retrieved': proto_details,
        'predicted_duration_p50': round(p50, 1) if not np.isnan(p50) else None,
        'predicted_duration_p80': round(p80, 1) if not np.isnan(p80) else None,
        'predicted_duration_p95': round(p95, 1) if not np.isnan(p95) else None,
        'predicted_closure_prob': round(cl_p, 4) if not np.isnan(cl_p) else None,
        'predicted_severity': round(sev, 3) if not np.isnan(sev) else None,
        'delta_obi':          round(dobi, 4) if not np.isnan(dobi) else None,
        'actual_duration':    round(actual, 1) if not np.isnan(actual) else None,
        'abs_error':          round(err, 1) if not np.isnan(err) else None,
    })

def _js(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    raise TypeError(type(obj))

with open(OUTPUTS / 'layer4_example_retrievals.json', 'w', encoding='utf-8') as f:
    json.dump(example_outputs, f, indent=2, default=_js)
print(f'\n  Saved: layer4_example_retrievals.json ({len(example_outputs)} examples)')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: SIMULATE NEW QUERY (DEMO FUNCTION)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION 10: DEMO SIMULATIONS ===")


def _find_closest_class(value: str, classes: list[str]) -> str:
    v_low = str(value).lower()
    # exact match
    for c in classes:
        if c.lower() == v_low:
            return c
    # substring match
    for c in classes:
        if v_low in c.lower() or c.lower() in v_low:
            return c
    # keyword match
    for kw in v_low.split('_'):
        for c in classes:
            if kw in c.lower() and len(kw) > 2:
                return c
    # frequency fallback: return first class (most frequent after sort)
    return classes[0]


# Compute most/least common causes from df_planned
cause_order = df_planned[CAUSE_COL].value_counts()
most_common_cause   = str(cause_order.index[0])
rarest_cause_demo   = str(cause_order.index[-1])
most_common_corr    = str(df_planned[CORRIDOR_COL].value_counts().index[0])


def simulate_planned_event(
    event_type:       str,
    corridor:         str,
    hour:             int,
    day_of_week:      int,
    expected_crowd:   int = 0,
    requires_closure: bool | None = None,
) -> dict:
    cause_classes = list(label_encoders[CAUSE_COL].classes_)
    corr_classes  = list(label_encoders[CORRIDOR_COL].classes_)

    matched_cause = _find_closest_class(event_type, [c for c in cause_classes if c != '__NA__'])
    matched_corr  = _find_closest_class(corridor,    [c for c in corr_classes  if c != '__NA__'])

    cl_val = 1 if requires_closure is True else (0 if requires_closure is False else
             int(df_planned['closure_binary'].mode()[0]))
    dur_est = float(df_planned['duration_clean'].median())
    prio_n  = float(df_planned['priority_numeric'].mode()[0])
    month_v = int(df_planned['month'].mode()[0])

    query: dict = {
        CAUSE_COL:          matched_cause,
        CORRIDOR_COL:       matched_corr,
        'closure_binary':   cl_val,
        'hour_of_day':      int(hour),
        'day_of_week':      int(day_of_week),
        'duration_clean':   dur_est,
        'priority_numeric': prio_n,
        'month':            month_v,
    }

    top_k_idx, sims, gdists = retrieve_top_k(query, prototypes_df, k=K_NEIGHBORS)
    conf, n_eff, msim = confidence_score(sims)
    abst, abst_reason = should_abstain(sims)
    outcomes = predict_outcomes(top_k_idx, sims, prototypes_df)

    if outcomes and expected_crowd > 0:
        crowd_factor = min(float(np.log1p(expected_crowd / 5000)), 2.0)
        adj_duration = outcomes['predicted_duration_median'] * max(1.0, crowd_factor)
        outcomes['predicted_duration_median'] = adj_duration
        outcomes['predicted_duration_p80']    *= max(1.0, crowd_factor)
        outcomes['predicted_duration_p95']    *= max(1.0, crowd_factor)
        outcomes['crowd_factor'] = crowd_factor

    if outcomes and requires_closure is not None:
        outcomes['predicted_closure_probability'] = 0.95 if requires_closure else 0.05

    proto_corridors = prototypes_df.iloc[top_k_idx]['corridor'].tolist()
    obi_v = [obi_lookup.get(c) for c in proto_corridors]
    obi_c = [v for v in obi_v if v is not None and not np.isnan(float(v))]
    w_o   = [float(sims[j]) for j, c in enumerate(proto_corridors) if obi_lookup.get(c) is not None]
    delta_obi_sim = float(np.average(obi_c, weights=w_o)) if obi_c else float('nan')

    return {
        'inputs': {
            'event_type': event_type, 'corridor': corridor,
            'hour': hour, 'day_of_week': day_of_week,
            'expected_crowd': expected_crowd, 'requires_closure': requires_closure,
        },
        'matched_cause':      matched_cause,
        'matched_corridor':   matched_corr,
        'confidence':         round(float(conf), 4),
        'n_eff':              round(float(n_eff), 3),
        'mean_similarity':    round(float(msim), 4),
        'abstain':            bool(abst),
        'abstain_reason':     abst_reason,
        'outcomes':           {k: round(float(v), 3) for k, v in (outcomes or {}).items()},
        'delta_obi':          round(delta_obi_sim, 4) if not np.isnan(delta_obi_sim) else None,
        'top_prototypes': [
            {
                'prototype_id': int(prototypes_df.iloc[j]['prototype_id']),
                'cause':        str(prototypes_df.iloc[j]['cause']),
                'corridor':     str(prototypes_df.iloc[j]['corridor']),
                'similarity':   round(float(sims[rank]), 4),
                'gower_dist':   round(float(gdists[rank]), 4),
            }
            for rank, j in enumerate(top_k_idx)
        ],
    }


# Run 3 demo simulations
sim1 = simulate_planned_event(most_common_cause, most_common_corr, hour=18, day_of_week=4, expected_crowd=10000)
sim2 = simulate_planned_event(rarest_cause_demo, most_common_corr, hour=11, day_of_week=1, expected_crowd=500)
sim3 = simulate_planned_event(most_common_cause, most_common_corr, hour=15, day_of_week=3, requires_closure=True)

demos = [
    {'scenario': 'peak_hour_large_crowd',   **sim1},
    {'scenario': 'offpeak_rare_cause',       **sim2},
    {'scenario': 'forced_closure_override',  **sim3},
]

for d in demos:
    print(f'  Scenario [{d["scenario"]}]: matched={d["matched_cause"]} @ {d["matched_corridor"]} | '
          f'conf={d["confidence"]:.3f} | n_eff={d["n_eff"]:.2f} | abstain={d["abstain"]}')
    if d.get('outcomes'):
        print(f'    dur_p50={d["outcomes"].get("predicted_duration_median",0):.0f}min | '
              f'closure={d["outcomes"].get("predicted_closure_probability",0):.0%}')

with open(OUTPUTS / 'layer4_simulation_demos.json', 'w', encoding='utf-8') as f:
    json.dump(demos, f, indent=2, default=_js)
print(f'  Saved: layer4_simulation_demos.json')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== LAYER 4 PLANNED EVENT RETRIEVAL COMPLETE ===")

required_outputs = [
    'layer4_planned_event_prototypes.csv',
    'layer4_planned_event_retrieval.csv',
    'layer4_retrieval_feature_weights.json',
    'layer4_retrieval_encoders.pkl',
    'layer4_retrieval_diagnostics.csv',
    'layer4_prototype_utilization.csv',
    'layer4_example_retrievals.json',
    'layer4_simulation_demos.json',
]

all_ok = True
for fname in required_outputs:
    p = OUTPUTS / fname
    if p.exists():
        sz = p.stat().st_size
        suffix = ''
        if fname.endswith('.csv'):
            rc = len(pd.read_csv(p))
            suffix = f'{rc} rows | '
        print(f'  [OK] {fname}: {suffix}{sz // 1024} KB')
    else:
        print(f'  [MISSING] {fname}')
        all_ok = False

top3_w = sorted(feature_weights.items(), key=lambda x: -x[1])[:3]
print(f'\n  Planned events:    {n_planned}')
print(f'  Prototypes:        {n_clusters}')
print(f'  Abstention rate:   {retrieval_df["abstain_flag"].mean():.1%}')
print(f'  Mean confidence:   {retrieval_df["confidence"].mean():.3f}')
print(f'  Mean n_eff:        {retrieval_df["effective_sample_size"].mean():.2f}')
print(f'  Top 3 feature weights: ' + ', '.join(f'{k}={v:.4f}' for k, v in top3_w))
print(f'  All outputs present: {all_ok}')


# ─────────────────────────────────────────────────────────────────
# GEO-RADIUS ENRICHMENT (additive post-processing — reads existing
# outputs, writes new columns + new CSV, never modifies upstream)
# ─────────────────────────────────────────────────────────────────
print("\n=== GEO-RADIUS ENRICHMENT ===")

try:
    import math
    import shutil
    from math import radians, sin, cos, sqrt, atan2

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    def gaussian_decay(dist_km, sigma):
        return math.exp(-(dist_km ** 2) / (2 * sigma ** 2))

    def _event_features(event_row) -> dict:
        feats: dict = {}
        for col in CATEGORICAL_COLS:
            feats[col] = str(event_row.get(col, '__NA__') or '__NA__')
        for col in BINARY_COLS:
            feats[col] = int(event_row.get(col, 0) or 0)
        for col in CONTINUOUS_COLS:
            v = pd.to_numeric(event_row.get(col, 0), errors='coerce')
            feats[col] = float(v) if not pd.isna(v) else 0.0
        return feats

    def _gower_sim_from_features(query_feats: dict, precedent_feats: dict) -> float:
        gd = gower_distance(
            query_feats,
            precedent_feats,
            feature_weights,
            feature_ranges,
            CATEGORICAL_COLS,
            BINARY_COLS,
            CONTINUOUS_COLS,
        )
        return float(math.exp(-gd / H_BANDWIDTH))

    hotspot_path = OUTPUTS / 'layer2_hotspots.csv'
    if not hotspot_path.exists():
        print('[geo-radius] WARNING: layer2_hotspots.csv not found — skipping geo block')
    else:
        hotspots_df = pd.read_csv(hotspot_path)
        lat_col = 'latitude' if 'latitude' in hotspots_df.columns else 'lat'
        lon_col = 'longitude' if 'longitude' in hotspots_df.columns else 'lon'
        junc_col = 'junction' if 'junction' in hotspots_df.columns else 'junction_name'

        junction_coords: dict[str, tuple[float, float]] = {}
        for _, hr in hotspots_df.iterrows():
            jname = str(hr.get(junc_col, '')).strip()
            if not jname or jname == 'nan':
                continue
            lat_v = pd.to_numeric(hr.get(lat_col), errors='coerce')
            lon_v = pd.to_numeric(hr.get(lon_col), errors='coerce')
            if pd.isna(lat_v) or pd.isna(lon_v):
                continue
            junction_coords[jname] = (float(lat_v), float(lon_v))

        if len(junction_coords) < 2:
            sigma = 2.0
            print('[geo-radius] WARNING: fewer than 2 junction coordinates — using σ = 2.0 km fallback')
        else:
            pairwise_dists: list[float] = []
            for corridor, grp in df_all.groupby(CORRIDOR_COL):
                juncs = [
                    str(j).strip()
                    for j in grp['junction'].dropna().unique()
                    if str(j).strip() in junction_coords
                ]
                coords = [junction_coords[j] for j in juncs]
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        pairwise_dists.append(
                            haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                        )
            sigma = float(np.median(pairwise_dists)) if pairwise_dists else 2.0

        print(f'[geo-radius] Derived bandwidth σ = {sigma:.3f} km')

        retrieval_path = OUTPUTS / 'layer4_planned_event_retrieval.csv'
        diag_path = OUTPUTS / 'layer4_retrieval_diagnostics.csv'
        if not retrieval_path.exists() or not diag_path.exists():
            print('[geo-radius] WARNING: retrieval outputs missing — skipping geo block')
        else:
            retrieval_geo_df = pd.read_csv(retrieval_path)
            diag_geo_df = pd.read_csv(diag_path)

            # Build per-query retrieved gower similarity lookup from top-k prototypes
            retrieved_sim_lookup: dict[str, dict[str, float]] = {}
            proto_by_id = {int(r['prototype_id']): r for _, r in prototypes_df.iterrows()}
            for _, rrow in retrieval_geo_df.iterrows():
                qid = str(rrow['query_event_id'])
                ids = [int(x) for x in str(rrow.get('top_k_prototype_ids', '')).split(',') if x.strip()]
                gdists = [float(x) for x in str(rrow.get('top_k_gower_dists', '')).split(',') if x.strip()]
                lookup: dict[str, float] = {}
                for pid, gd in zip(ids, gdists):
                    if pid in proto_by_id:
                        rep_id = str(proto_by_id[pid]['representative_event_id'])
                        lookup[rep_id] = max(lookup.get(rep_id, 0.0), float(math.exp(-gd / H_BANDWIDTH)))
                retrieved_sim_lookup[qid] = lookup

            events_pool = df_all.copy()
            event_feats_cache: dict[str, dict] = {}
            for _, er in events_pool.iterrows():
                event_feats_cache[str(er['event_id'])] = _event_features(er)

            geo_diag_rows: list[dict] = []
            geo_match_rows: list[dict] = []

            for _, qrow in df_planned.iterrows():
                qid = str(qrow['event_id'])
                q_junction = str(qrow.get('junction', '') or '').strip()
                q_feats = _event_features(qrow)

                if not q_junction or q_junction not in junction_coords:
                    print(f'[geo-radius] WARNING: no coordinates for query junction "{q_junction}" (event {qid})')
                    geo_diag_rows.append({
                        'query_event_id': qid,
                        'geo_radius_2km_count': 0,
                        'geo_radius_nearest_km': float('nan'),
                        'geo_sigma_km': sigma,
                    })
                    continue

                lat_q, lon_q = junction_coords[q_junction]
                retrieved_for_q = retrieved_sim_lookup.get(qid, {})
                scored: list[dict] = []

                for _, prow in events_pool.iterrows():
                    pid = str(prow['event_id'])
                    if pid == qid:
                        continue
                    p_junction = str(prow.get('junction', '') or '').strip()
                    if not p_junction or p_junction not in junction_coords:
                        continue

                    lat_p, lon_p = junction_coords[p_junction]
                    dist_km = haversine_km(lat_q, lon_q, lat_p, lon_p)
                    p_feats = event_feats_cache[pid]
                    gower_sim = _gower_sim_from_features(q_feats, p_feats)
                    if pid in retrieved_for_q:
                        gower_sim = max(gower_sim, retrieved_for_q[pid])
                    elif gower_sim < 0.5:
                        gower_sim = 0.0

                    phi = gaussian_decay(dist_km, sigma)
                    trust_p = float(pd.to_numeric(prow.get('trust_clean', prow.get('trust_score', 0.7)), errors='coerce') or 0.7)
                    trust_p = float(np.clip(trust_p, 0.1, 1.0))
                    s_final = gower_sim * phi * trust_p

                    scored.append({
                        'match_event_id': pid,
                        'match_junction': p_junction,
                        'match_lat': lat_p,
                        'match_lon': lon_p,
                        'match_cause': str(prow.get(CAUSE_COL, '')),
                        'match_duration_min': float(pd.to_numeric(prow.get('duration_clean', prow.get(DURATION_COL)), errors='coerce') or 0),
                        'dist_km': dist_km,
                        'gower_sim': gower_sim,
                        'phi_weight': phi,
                        's_final': s_final,
                        'within_2km': dist_km <= 2.0,
                        'is_relevant': dist_km <= 2.0 and gower_sim >= 0.5,
                    })

                nearby_relevant = [s for s in scored if s['is_relevant']]
                nearest_km = min((s['dist_km'] for s in nearby_relevant), default=float('nan'))

                geo_diag_rows.append({
                    'query_event_id': qid,
                    'geo_radius_2km_count': len(nearby_relevant),
                    'geo_radius_nearest_km': nearest_km,
                    'geo_sigma_km': sigma,
                })

                top5 = sorted(scored, key=lambda x: -x['s_final'])[:5]
                for rank, m in enumerate(top5, start=1):
                    geo_match_rows.append({
                        'query_event_id': qid,
                        'query_junction': q_junction,
                        'query_lat': lat_q,
                        'query_lon': lon_q,
                        'match_rank': rank,
                        'match_event_id': m['match_event_id'],
                        'match_junction': m['match_junction'],
                        'match_lat': m['match_lat'],
                        'match_lon': m['match_lon'],
                        'match_cause': m['match_cause'],
                        'match_duration_min': round(m['match_duration_min'], 1),
                        'dist_km': round(m['dist_km'], 4),
                        'gower_sim': round(m['gower_sim'], 4),
                        'phi_weight': round(m['phi_weight'], 4),
                        's_final': round(m['s_final'], 4),
                        'within_2km': bool(m['within_2km']),
                        'is_relevant': bool(m['is_relevant']),
                    })

            geo_diag_df = pd.DataFrame(geo_diag_rows)
            diag_merged = diag_geo_df.drop(
                columns=['geo_radius_2km_count', 'geo_radius_nearest_km', 'geo_sigma_km'],
                errors='ignore',
            ).merge(
                geo_diag_df,
                on='query_event_id',
                how='left',
            )
            diag_merged['geo_radius_2km_count'] = diag_merged['geo_radius_2km_count'].fillna(0).astype(int)
            diag_merged.to_csv(diag_path, index=False)

            geo_matches_df = pd.DataFrame(geo_match_rows)
            geo_matches_path = OUTPUTS / 'layer4_geo_radius_matches.csv'
            geo_matches_df.to_csv(geo_matches_path, index=False)

            FRONTEND = OUTPUTS / 'frontend'
            FRONTEND.mkdir(parents=True, exist_ok=True)
            shutil.copy2(diag_path, FRONTEND / 'layer4_retrieval_diagnostics.csv')
            geo_matches_df.to_csv(FRONTEND / 'layer4_geo_radius_matches.csv', index=False)

            counts = geo_diag_df['geo_radius_2km_count'].astype(float)
            mean_count = float(counts.mean()) if len(counts) else 0.0
            n_with_geo = int((counts > 0).sum())
            print(f'[geo-radius] Processed {n_planned} planned events')
            print(f'[geo-radius] Mean precedents within 2km: {mean_count:.1f}')
            print(f'[geo-radius] Events with ≥1 nearby relevant precedent: {n_with_geo}')
            print(f'[geo-radius] Derived σ = {sigma:.3f} km')
            print('[geo-radius] Written: layer4_retrieval_diagnostics.csv (3 new cols)')
            print('[geo-radius] Written: layer4_geo_radius_matches.csv')

except Exception as geo_exc:
    print(f'[geo-radius] ERROR (non-fatal): {geo_exc}')


# ─────────────────────────────────────────────────────────────────
# TEMPORAL DECAY ENRICHMENT (additive post-processing)
#
# Motivation: Layer 6 found CRITICAL sustained shift in log-duration
# (PH max=326.8, mean-shift z=3.21) between Nov-Feb and Mar-Apr.
# Recent precedents are therefore more informative than older ones.
#
# Math: φ_time(Δt) = exp(−λ·Δt_days), same exponential decay
# family as Layer 6 forgetting (w_i = exp(−λ·Δt)) and Layer 7
# Hawkes kernel (α·exp(−β(t−tᵢ))). Consistent mathematical
# principle applied across three layers.
#
# λ derived from autocorrelation of incident duration series —
# not hardcoded. Half-life = lag where autocorr drops below 0.5.
# ─────────────────────────────────────────────────────────────────
print("\n=== TEMPORAL DECAY ENRICHMENT ===")

try:
    import math
    import shutil

    LAYER6_HALF_LIFE_DAYS = 30.0
    half_life_days = LAYER6_HALF_LIFE_DAYS
    fallback_used = False

    retrieval_path = OUTPUTS / 'layer4_planned_event_retrieval.csv'
    events_path = DATA / 'events_clean.parquet'

    if not retrieval_path.exists():
        print('[temporal-decay] WARNING: layer4_planned_event_retrieval.csv not found — skipping')
    elif not events_path.exists():
        print('[temporal-decay] WARNING: events_clean.parquet not found — skipping')
    else:
        retrieval_temp_df = pd.read_csv(retrieval_path)

        events_temp = df_all.copy()
        if START_COL not in events_temp.columns:
            events_temp = pd.read_parquet(events_path)
            events_temp[START_COL] = pd.to_datetime(events_temp[START_COL], errors='coerce')

        events_temp['duration_for_ac'] = pd.to_numeric(
            events_temp.get('duration_clean', events_temp.get(DURATION_COL)),
            errors='coerce',
        )

        try:
            ac_df = events_temp[
                events_temp['duration_for_ac'].notna()
                & (events_temp['duration_for_ac'] > 0)
                & (events_temp['duration_for_ac'] < 1440)
            ].copy()
            ac_df['date'] = pd.to_datetime(ac_df[START_COL], errors='coerce').dt.date
            daily_mean = ac_df.groupby('date')['duration_for_ac'].mean().sort_index()
            if len(daily_mean) >= 10:
                autocorr_values = [
                    daily_mean.autocorr(lag=lag) for lag in range(1, 61)
                ]
                for lag, ac in enumerate(autocorr_values, start=1):
                    if ac is not None and not math.isnan(ac) and ac < 0.5:
                        half_life_days = float(lag)
                        break
                else:
                    half_life_days = LAYER6_HALF_LIFE_DAYS
                    fallback_used = True
                print(f'[temporal-decay] Derived half-life: {half_life_days:.1f} days')
                print('  (lag where duration autocorrelation < 0.5)')
                print(f'  (Layer 6 uses {LAYER6_HALF_LIFE_DAYS:.0f}-day half-life for comparison)')
            else:
                fallback_used = True
                print('[temporal-decay] WARNING: insufficient daily duration series — using 30-day default')
        except Exception as ac_exc:
            fallback_used = True
            half_life_days = LAYER6_HALF_LIFE_DAYS
            print(f'[temporal-decay] Autocorr computation failed: {ac_exc}')
            print(f'[temporal-decay] Using default half-life: {half_life_days} days')

        if half_life_days < 1.0 or half_life_days > 180.0:
            print(
                f'[temporal-decay] WARNING: derived half-life {half_life_days:.1f} outside [1, 180] — '
                f'using {LAYER6_HALF_LIFE_DAYS:.0f}-day fallback (Layer 6)'
            )
            half_life_days = LAYER6_HALF_LIFE_DAYS
            fallback_used = True

        lambda_decay = math.log(2) / half_life_days
        print(f'[temporal-decay] λ = {lambda_decay:.6f} per day')

        def phi_time(delta_t_days: float, lam: float = lambda_decay) -> float:
            """φ_time(Δt) = exp(−λ · Δt_days); same form as Layer 6 forgetting."""
            if delta_t_days < 0:
                return 1.0
            return math.exp(-lam * delta_t_days)

        start_by_event = {
            str(r['event_id']): pd.to_datetime(r[START_COL], errors='coerce')
            for _, r in events_temp.iterrows()
            if pd.notna(r.get('event_id'))
        }
        trust_by_event = {
            str(r['event_id']): float(pd.to_numeric(r.get('trust_clean', r.get(TRUST_COL)), errors='coerce') or 0.7)
            for _, r in events_temp.iterrows()
            if pd.notna(r.get('event_id'))
        }
        junction_by_event = {
            str(r['event_id']): str(r.get('junction', '') or '')
            for _, r in events_temp.iterrows()
            if pd.notna(r.get('event_id'))
        }

        geo_path = OUTPUTS / 'layer4_geo_radius_matches.csv'
        geo_temp_df = pd.read_csv(geo_path) if geo_path.exists() else pd.DataFrame()
        proto_lookup = {
            int(r['prototype_id']): r for _, r in prototypes_df.iterrows()
        }

        def _matches_for_query(qrow: pd.Series) -> list[dict]:
            qid = str(qrow['query_event_id'])
            geo_sub = geo_temp_df[geo_temp_df['query_event_id'].astype(str) == qid]
            out: list[dict] = []
            if len(geo_sub) > 0:
                for _, m in geo_sub.iterrows():
                    out.append({
                        'match_event_id': str(m['match_event_id']),
                        'original_rank': int(pd.to_numeric(m.get('match_rank'), errors='coerce') or 0),
                        'gower_sim': float(pd.to_numeric(m.get('gower_sim'), errors='coerce') or 0.0),
                        's_final': float(pd.to_numeric(m.get('s_final'), errors='coerce'))
                        if pd.notna(m.get('s_final')) else None,
                    })
                return out

            ids = [int(x) for x in str(qrow.get('top_k_prototype_ids', '')).split(',') if x.strip()]
            sims = [float(x) for x in str(qrow.get('top_k_similarities', '')).split(',') if x.strip()]
            for rank, (pid, sim) in enumerate(zip(ids, sims), start=1):
                if pid not in proto_lookup:
                    continue
                rep_id = str(proto_lookup[pid]['representative_event_id'])
                out.append({
                    'match_event_id': rep_id,
                    'original_rank': rank,
                    'gower_sim': float(sim),
                    's_final': None,
                })
            return out

        summary_rows: list[dict] = []
        query_enrich: dict[str, dict] = {}
        all_rank_changes: list[int] = []
        phi_top1_vals: list[float] = []
        top1_ages: list[float] = []

        for _, qrow in retrieval_temp_df.iterrows():
            qid = str(qrow['query_event_id'])
            q_start = start_by_event.get(qid)
            matches = _matches_for_query(qrow)
            scored: list[dict] = []

            for m in matches:
                mid = m['match_event_id']
                m_start = start_by_event.get(mid)
                if q_start is None or m_start is None or pd.isna(q_start) or pd.isna(m_start):
                    delta_t = 0.0
                else:
                    delta_t = float((q_start - m_start).total_seconds() / 86400.0)
                    if delta_t < 0:
                        print(f'[temporal-decay] WARNING: negative Δt for query {qid} / match {mid} — using abs()')
                        delta_t = abs(delta_t)

                phi_t = phi_time(delta_t)
                trust_p = float(np.clip(trust_by_event.get(mid, 0.7), 0.1, 1.0))
                if m['s_final'] is not None:
                    s_temporal = float(m['s_final']) * phi_t
                    score_formula = 's_gower · φ_space · φ_time · τ'
                else:
                    s_temporal = float(m['gower_sim']) * phi_t * trust_p
                    score_formula = 's_gower · φ_time · τ'

                scored.append({
                    **m,
                    'delta_t_days': int(round(delta_t)),
                    'phi_time_weight': round(phi_t, 4),
                    's_temporal': round(s_temporal, 6),
                    'score_formula': score_formula,
                })

            if not scored:
                query_enrich[qid] = {
                    'delta_t_days': '',
                    'phi_time_weight': '',
                    's_temporal': '',
                    'temporal_rank': '',
                    'rank_change': '',
                    'half_life_days_used': round(half_life_days, 4),
                    'lambda_used': round(lambda_decay, 6),
                }
                summary_rows.append({
                    'query_event_id': qid,
                    'query_junction': junction_by_event.get(qid, ''),
                    'query_start_date': str(q_start.date()) if q_start is not None and not pd.isna(q_start) else '',
                    'temporal_mean_delta_t_days': '',
                    'temporal_top1_delta_t_days': '',
                    'top1_phi_time': '',
                    'n_rank_changes_in_top5': 0,
                    'half_life_days_used': round(half_life_days, 4),
                })
                continue

            scored.sort(key=lambda x: -x['s_temporal'])
            for t_rank, item in enumerate(scored, start=1):
                item['temporal_rank'] = t_rank
                item['rank_change'] = int(item['original_rank'] - t_rank)
                all_rank_changes.append(abs(item['rank_change']))

            top5 = scored[:5]
            top1 = scored[0]
            n_changes_top5 = int(sum(1 for x in top5 if abs(x['rank_change']) >= 1))
            mean_delta_top5 = float(np.mean([x['delta_t_days'] for x in top5]))

            phi_top1_vals.append(float(top1['phi_time_weight']))
            top1_ages.append(float(top1['delta_t_days']))

            query_enrich[qid] = {
                'delta_t_days': top1['delta_t_days'],
                'phi_time_weight': top1['phi_time_weight'],
                's_temporal': top1['s_temporal'],
                'temporal_rank': top1['temporal_rank'],
                'rank_change': top1['rank_change'],
                'half_life_days_used': round(half_life_days, 4),
                'lambda_used': round(lambda_decay, 6),
            }
            summary_rows.append({
                'query_event_id': qid,
                'query_junction': junction_by_event.get(qid, ''),
                'query_start_date': str(q_start.date()) if q_start is not None and not pd.isna(q_start) else '',
                'temporal_mean_delta_t_days': round(mean_delta_top5, 2),
                'temporal_top1_delta_t_days': top1['delta_t_days'],
                'top1_phi_time': top1['phi_time_weight'],
                'n_rank_changes_in_top5': n_changes_top5,
                'half_life_days_used': round(half_life_days, 4),
            })

        enrich_df = pd.DataFrame.from_dict(query_enrich, orient='index').reset_index(names='query_event_id')
        new_cols = [
            'delta_t_days', 'phi_time_weight', 's_temporal', 'temporal_rank',
            'rank_change', 'half_life_days_used', 'lambda_used',
        ]
        retrieval_out = retrieval_temp_df.drop(columns=new_cols, errors='ignore').merge(
            enrich_df, on='query_event_id', how='left'
        )
        retrieval_out.to_csv(retrieval_path, index=False)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUTPUTS / 'layer4_temporal_decay_summary.csv'
        summary_df.to_csv(summary_path, index=False)

        total_retrievals = int(len(all_rank_changes))
        n_rank_changes = int(sum(1 for x in all_rank_changes if x >= 1))
        mean_rank_change = float(np.mean(all_rank_changes)) if all_rank_changes else 0.0
        pct_recency = (n_rank_changes / total_retrievals * 100.0) if total_retrievals else 0.0
        mean_phi_t_top1 = float(np.mean(phi_top1_vals)) if phi_top1_vals else float('nan')
        oldest_top1 = float(max(top1_ages)) if top1_ages else float('nan')
        newest_top1 = float(min(top1_ages)) if top1_ages else float('nan')

        if abs(half_life_days - LAYER6_HALF_LIFE_DAYS) <= 5:
            comparison_note = (
                f'Layer 6 uses 30-day half-life for Bayesian forgetting. Derived value of '
                f'{half_life_days:.1f} days from duration autocorrelation is consistent with this.'
            )
        else:
            comparison_note = (
                f'Layer 6 uses 30-day half-life for Bayesian forgetting. Derived value of '
                f'{half_life_days:.1f} days from duration autocorrelation differs from this.'
            )

        score_formula = (
            's_gower · φ_space · φ_time · τ'
            if len(geo_temp_df) > 0 and 's_final' in geo_temp_df.columns
            else 's_gower · φ_time · τ'
        )

        metadata = {
            'half_life_days': float(half_life_days),
            'lambda_decay': float(lambda_decay),
            'derivation_method': 'autocorrelation_daily_duration_series',
            'autocorr_threshold': 0.5,
            'fallback_used': bool(fallback_used),
            'layer6_half_life_days': LAYER6_HALF_LIFE_DAYS,
            'comparison_note': comparison_note,
            'n_retrievals_processed': total_retrievals,
            'n_rank_changes': n_rank_changes,
            'pct_recency_matters': round(pct_recency, 4),
            'mean_rank_change': round(mean_rank_change, 4),
            'mean_phi_t_top1': round(mean_phi_t_top1, 4) if phi_top1_vals else None,
            'oldest_top1_days': round(oldest_top1, 2) if top1_ages else None,
            'newest_top1_days': round(newest_top1, 2) if top1_ages else None,
            'score_formula': score_formula,
        }
        meta_path = OUTPUTS / 'layer4_temporal_decay_metadata.json'
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(metadata, mf, indent=2)

        FRONTEND = OUTPUTS / 'frontend'
        FRONTEND.mkdir(parents=True, exist_ok=True)
        shutil.copy2(retrieval_path, FRONTEND / 'layer4_planned_event_retrieval.csv')
        summary_df.to_csv(FRONTEND / 'layer4_temporal_decay_summary.csv', index=False)
        shutil.copy2(meta_path, FRONTEND / 'layer4_temporal_decay_metadata.json')

        pe_front = FRONTEND / 'planned_event_recommendations.csv'
        if pe_front.exists():
            pe_df = pd.read_csv(pe_front)
            merge_cols = [
                'query_event_id', 'temporal_top1_delta_t_days', 'top1_phi_time',
                'n_rank_changes_in_top5', 'half_life_days_used',
            ]
            pe_merge = summary_df.rename(columns={'query_event_id': 'event_id'})
            keep = ['event_id'] + [c for c in merge_cols[1:] if c in pe_merge.columns]
            if 'event_id' in pe_df.columns:
                pe_df = pe_df.drop(columns=[c for c in keep[1:] if c in pe_df.columns], errors='ignore')
                pe_df = pe_df.merge(pe_merge[keep], on='event_id', how='left')
                pe_df.to_csv(pe_front, index=False)

        print('\n[temporal-decay] Summary:')
        print(f'  Half-life: {half_life_days:.1f} days (λ={lambda_decay:.6f})')
        print(f'  Re-rankings (|Δrank| ≥ 1): {n_rank_changes} / {total_retrievals} ({pct_recency:.1f}%)')
        print(f'  Mean |rank change|: {mean_rank_change:.2f} positions')
        print(f'  Mean φ_time of top-1 match: {mean_phi_t_top1:.3f}')
        if top1_ages:
            print(f'  Age range of top-1 matches: {newest_top1:.0f}–{oldest_top1:.0f} days')
        print('  Wrote layer4_planned_event_retrieval.csv (7 new columns)')
        print('  Wrote layer4_temporal_decay_summary.csv')
        print('  Wrote layer4_temporal_decay_metadata.json')

except Exception as temporal_exc:
    print(f'[temporal-decay] ERROR (non-fatal): {temporal_exc}')
