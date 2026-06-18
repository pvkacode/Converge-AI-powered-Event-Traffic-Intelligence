"""
Tail-risk models for Layer 4.5 Duration Quality Gate — ASTraM.

Trains a binary tail-risk classifier (CatBoostClassifier) using cause-specific
thresholds derived from the training window only.  Also provides conservative
tail-quantile proxies for the tail-aware mixture of quantiles.

Additive only — does not modify Layers 1–4 or any existing targets.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

MIN_TAIL_POSITIVES = 10

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

_TAIL_CAT_PARAMS: dict = dict(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=5,
    random_strength=2,
    bagging_temperature=1,
    early_stopping_rounds=30,
    verbose=0,
    random_seed=42,
)


def compute_tail_labels(
    y_dur: pd.Series,
    causes: pd.Series,
    cause_tau: dict[str, float],
    global_tau: float,
) -> np.ndarray:
    """
    Binary tail label: 1 if duration_i > tau_{cause(i)}.

    tau_cause is estimated from the training window only (no leakage).
    Sparse causes fall back to the global training threshold.
    """
    tau_vec = np.array([cause_tau.get(str(c), global_tau) for c in causes], dtype=float)
    return (np.asarray(y_dur, dtype=float) > tau_vec).astype(int)


def train_tail_classifier(
    X_train: pd.DataFrame,
    y_tail_train: np.ndarray,
    X_val: pd.DataFrame,
    y_tail_val: np.ndarray,
    cat_idx: list[int],
):
    """
    Train a CatBoostClassifier for tail-risk probability.

    Returns None if the training set has fewer than MIN_TAIL_POSITIVES
    positive examples; the caller then uses the base-rate fallback.
    """
    n_pos = int(np.sum(y_tail_train))
    if n_pos < MIN_TAIL_POSITIVES:
        logger.warning(
            "Tail classifier skipped: only %d positive examples (< %d). "
            "Base-rate fallback will be used.",
            n_pos, MIN_TAIL_POSITIVES,
        )
        return None

    if _HAS_CATBOOST:
        model = CatBoostClassifier(loss_function="Logloss", **_TAIL_CAT_PARAMS)
        model.fit(
            X_train, y_tail_train,
            eval_set=(X_val, y_tail_val),
            cat_features=cat_idx,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_tail_train)

    return model


def calibrate_tail_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[IsotonicRegression, np.ndarray]:
    """Isotonic calibration of raw tail-risk probabilities on the val set."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return iso, iso.predict(y_prob)


def predict_tail_risk(
    model,
    X: pd.DataFrame,
    base_rate: float = 0.25,
) -> np.ndarray:
    """
    Predict tail-risk probabilities.

    Falls back to the training base rate if the model is None (sparse data).
    """
    if model is None:
        return np.full(len(X), float(base_rate))
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = np.asarray(model.predict(X), dtype=float)
    return np.clip(probs, 0.0, 1.0)


def build_tail_proxy_quantiles(
    y_dur_train: pd.Series,
    y_tail_train: np.ndarray,
    causes_train: pd.Series,
) -> dict:
    """
    Conservative tail quantile proxies from training-window tail events.

    Uses strictly training data — no future information leaks.
    Returns:
        {
            "global": {"p50": ..., "p80": ..., "p95": ...},
            "by_cause": {cause_str: {"p50": ..., "p80": ..., "p95": ...}, ...},
        }
    """
    dur = np.asarray(y_dur_train, dtype=float)
    tail_mask = np.asarray(y_tail_train, dtype=bool)
    tail_dur = dur[tail_mask]

    if len(tail_dur) == 0:
        global_proxy: dict[str, float] = {"p50": 120.0, "p80": 240.0, "p95": 480.0}
    else:
        global_proxy = {
            "p50": float(np.quantile(tail_dur, 0.50)),
            "p80": float(np.quantile(tail_dur, 0.80)),
            "p95": float(np.quantile(tail_dur, 0.95)),
        }

    by_cause: dict[str, dict[str, float]] = {}
    causes_arr = np.asarray(causes_train, dtype=str)
    for cause in np.unique(causes_arr):
        mask = causes_arr == cause
        cause_tail = dur[mask & tail_mask]
        if len(cause_tail) >= 5:
            by_cause[cause] = {
                "p50": float(np.quantile(cause_tail, 0.50)),
                "p80": float(np.quantile(cause_tail, 0.80)),
                "p95": float(np.quantile(cause_tail, 0.95)),
            }

    return {"global": global_proxy, "by_cause": by_cause}


def blend_tail_quantiles(
    typ_p50: np.ndarray,
    typ_p80: np.ndarray,
    typ_p95: np.ndarray,
    tail_probs: np.ndarray,
    causes: pd.Series,
    tail_proxies: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tail-aware mixture applied to p80 and p95 only.

    p50 is intentionally left unchanged.  Mixing tail-regime proxies (which are
    large by definition — they are quantiles of events that exceeded tau_cause)
    into the p50 estimate drags the typical-case prediction far above the true
    value for ordinary events, even when tail_risk_prob is small.  The p50
    should represent the most likely outcome; tail risk is properly expressed in
    p80/p95 and in tail_risk_prob itself.

        mix_p50 = typ_p50                                   (unchanged)
        mix_p80 = (1 - pi) * typ_p80 + pi * tail_p80
        mix_p95 = (1 - pi) * typ_p95 + pi * tail_p95
    """
    pi = np.clip(np.asarray(tail_probs, dtype=float), 0.0, 1.0)
    global_p = tail_proxies.get("global", {"p50": 120.0, "p80": 240.0, "p95": 480.0})
    by_cause = tail_proxies.get("by_cause", {})

    causes_arr = np.asarray(causes, dtype=str)
    n = len(causes_arr)
    tail_p80 = np.empty(n)
    tail_p95 = np.empty(n)

    global_cnt = 0
    for i, c in enumerate(causes_arr):
        proxy = by_cause.get(c)
        if proxy is None:
            proxy = global_p
            global_cnt += 1
        tail_p80[i] = proxy["p80"]
        tail_p95[i] = proxy["p95"]

    if global_cnt > 0:
        logger.info(
            "Tail proxy: %d / %d rows used global fallback (sparse cause).",
            global_cnt, n,
        )

    mix_p50 = np.array(typ_p50, dtype=float)          # p50 is not tail-mixed
    mix_p80 = (1.0 - pi) * typ_p80 + pi * tail_p80
    mix_p95 = (1.0 - pi) * typ_p95 + pi * tail_p95

    return mix_p50, mix_p80, mix_p95
