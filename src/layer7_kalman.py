"""
Layer 7 — M7B: standard linear Kalman filter (pure, numerically guarded).

Predict:  x = A x + f ;            P = A P Aᵀ + Q
Update:   S = H P Hᵀ + R ;          K = P Hᵀ S⁻¹
          x = x + K (y - H x) ;     P = (I-KH) P (I-KH)ᵀ + K R Kᵀ   (Joseph form)

Joseph-form covariance update + symmetrization keep P symmetric positive-semidefinite
for numerical stability. Uses np.linalg.solve (not an explicit inverse). No I/O.

ADDITIVE ONLY. This module writes nothing.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    def __init__(self, A, H, Q, R, x0, P0):
        self.A = np.asarray(A, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R0 = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float).reshape(-1)
        self.P = np.asarray(P0, dtype=float)
        self.n = self.x.size
        self.update_count = 0
        self.max_var = float(np.max(np.diag(self.P)))
        self.min_eig = float(np.min(np.linalg.eigvalsh(self._sym(self.P))))
        self.max_cond = 0.0

    @staticmethod
    def _sym(P):
        return 0.5 * (P + P.T)

    def predict(self, forcing=None):
        f = np.zeros(self.n) if forcing is None else np.asarray(forcing, dtype=float).reshape(-1)
        self.x = self.A @ self.x + f
        self.P = self._sym(self.A @ self.P @ self.A.T + self.Q)
        return self.x

    def update(self, y, R=None):
        y = np.asarray(y, dtype=float).reshape(-1)
        R = self.R0 if R is None else np.asarray(R, dtype=float)
        S = self._sym(self.H @ self.P @ self.H.T + R)
        cond = float(np.linalg.cond(S))
        self.max_cond = max(self.max_cond, cond)
        # K = P Hᵀ S⁻¹  via solve:  (S Kᵀ = H P)  ->  Kᵀ = solve(S, H P)
        Kt = np.linalg.solve(S, self.H @ self.P)
        K = Kt.T
        innovation = y - self.H @ self.x
        self.x = self.x + K @ innovation
        ImKH = np.eye(self.n) - K @ self.H
        self.P = self._sym(ImKH @ self.P @ ImKH.T + K @ R @ K.T)  # Joseph form
        self.update_count += 1
        self.max_var = max(self.max_var, float(np.max(np.diag(self.P))))
        self.min_eig = min(self.min_eig, float(np.min(np.linalg.eigvalsh(self.P))))
        return self.x

    @property
    def variances(self) -> np.ndarray:
        return np.clip(np.diag(self.P), 0.0, None)

    def is_stable(self) -> bool:
        finite = bool(np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.P)))
        psd = self.min_eig >= -1e-6
        bounded = self.max_var < 1e9
        return finite and psd and bounded

    def stability_report(self) -> dict:
        return {
            "spectral_radius_A": float(np.max(np.abs(np.linalg.eigvals(self.A)))),
            "max_posterior_variance": self.max_var,
            "min_P_eigenvalue": self.min_eig,
            "max_innovation_cov_cond": self.max_cond,
            "update_count": self.update_count,
            "stable": self.is_stable(),
        }
