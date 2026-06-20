"""
Layer 7 — M7B Step 6: short-horizon state forecasting.

Propagates the posterior state forward via the (stable, mean-reverting) dynamics:

    x_(t+h) = A^h x_t + (I - A^h) t*
    P_(t+h) = A^h P A^hᵀ + Σ_{i=0}^{h-1} A^i Q A^iᵀ

Horizons 5 / 15 / 30 min (h = 1 / 3 / 6 filter steps). No I/O.

ADDITIVE ONLY.
"""

from __future__ import annotations

import numpy as np

from layer7_state_space import IDX, STEP_MINUTES

# reference (acceptable) std per quantity -> forecast confidence falls as horizon variance
# grows (consistent with the state-uncertainty confidence metric).
_SCALE = {"speed": 8.0, "queue_length": 45.0, "travel_time": 22.0, "incident_intensity": 0.15}
FORECAST_QUANTITIES = ["speed", "queue_length", "travel_time", "incident_intensity"]
HORIZON_MIN = [5, 15, 30]

# PATCH M7B.1-A: horizon uncertainty-accumulation models.
#   "decayed"  (original): Ph = A^h P A^hᵀ + Σ A^i Q A^iᵀ  -> variance saturates (flat conf)
#   "linear_h" (Approach 1): Ph = A^h P A^hᵀ + h·Q
#   "factor"   (Approach 2): Ph = A^h P A^hᵀ + factor(h)·Σ A^i Q A^iᵀ
HORIZON_FACTORS = {5: 1.0, 15: 1.5, 30: 2.0}
DEFAULT_HORIZON_NOISE = "factor"   # selected by layer7_forecast_audit (see Part A2)


def _Ah(A, h):
    return np.linalg.matrix_power(A, h)


def _accumulated_Q(A, Q, h):
    acc = np.zeros_like(Q)
    for i in range(h):
        Ai = _Ah(A, i)
        acc = acc + Ai @ Q @ Ai.T
    return acc


def forecast(A, x, P, Q, tstar, mode: str = DEFAULT_HORIZON_NOISE):
    """Return list of (horizon_min, quantity, mean, variance, confidence)."""
    rows = []
    for hmin in HORIZON_MIN:
        h = max(1, int(round(hmin / STEP_MINUTES)))
        Ah = _Ah(A, h)
        x_h = Ah @ x + (np.eye(A.shape[0]) - Ah) @ tstar
        AhPAh = Ah @ P @ Ah.T
        if mode == "linear_h":
            proc = h * Q
        elif mode == "factor":
            proc = HORIZON_FACTORS.get(hmin, 1.0) * _accumulated_Q(A, Q, h)
        else:  # "decayed"
            proc = _accumulated_Q(A, Q, h)
        Ph = 0.5 * (AhPAh + AhPAh.T) + proc
        var = np.clip(np.diag(Ph), 0.0, None)
        for q in FORECAST_QUANTITIES:
            j = IDX[q]
            v = float(var[j])
            conf = float(1.0 / (1.0 + v / (_SCALE[q] ** 2)))
            mean = float(x_h[j])
            if q in ("incident_intensity",):
                mean = float(np.clip(mean, 0.0, 1.0))
            elif q in ("speed", "queue_length", "travel_time"):
                mean = float(max(0.0, mean))
            rows.append((hmin, q, round(mean, 4), round(v, 6), round(conf, 4)))
    return rows
