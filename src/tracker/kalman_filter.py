"""src/tracker/kalman_filter.py — Constant-velocity Kalman filter in (cx, cy, aspect, h).

v2 used fixed pixel noise (Q = 1px, R = 1px/10px), which is too loose for an
8px pedestrian and far too tight for a 200px bus. Here every noise term scales
with box height (ByteTrack / DeepSORT reference weights 1/20 and 1/160), so the
filter behaves the same at every altitude and object size.

State (8): [cx, cy, a, h, vcx, vcy, va, vh]   a = w / h
Measurement (4): [cx, cy, a, h]
"""
from __future__ import annotations

import numpy as np
import scipy.linalg


def tlbr_to_xyah(tlbr) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(tlbr, dtype=float)[:4]
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / max(h, 1e-6), h], dtype=float)


def xyah_to_tlbr(xyah) -> np.ndarray:
    cx, cy, a, h = np.asarray(xyah, dtype=float)[:4]
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=float)


class KalmanFilterXYAH:
    """Stateless filter: tracks own their (mean, covariance); this object holds the model."""

    def __init__(self) -> None:
        ndim = 4
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = 1.0
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement) -> tuple[np.ndarray, np.ndarray]:
        measurement = np.asarray(measurement, dtype=float)
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h, 2 * self._std_weight_position * h, 1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h, 10 * self._std_weight_velocity * h, 1e-5,
            10 * self._std_weight_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std_pos = [self._std_weight_position * h, self._std_weight_position * h, 1e-2,
                   self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * h, self._std_weight_velocity * h, 1e-5,
                   self._std_weight_velocity * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._std_weight_position * h, self._std_weight_position * h, 1e-1,
               self._std_weight_position * h]
        innovation_cov = np.diag(np.square(std))
        return (self._update_mat @ mean,
                self._update_mat @ covariance @ self._update_mat.T + innovation_cov)

    def update(self, mean, covariance, measurement) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), (covariance @ self._update_mat.T).T, check_finite=False).T
        innovation = np.asarray(measurement, dtype=float) - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov
