"""src/tracker/matching.py — Association geometry and assignment for ByteTrack.

Cost convention: cost = 1 - IoU (lower is better). A pair is accepted when
cost <= thresh, i.e. IoU >= 1 - thresh (the v2 cost-threshold fix is kept).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

GATED_COST = 1e6  # finite (scipy needs a feasible matrix) but far above any threshold


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)


def _area(x: np.ndarray) -> np.ndarray:
    return np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)


def iou_matrix(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter = _intersection(a, b)
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def ioa_matrix(a, b) -> np.ndarray:
    """Intersection over the area of each box in `a` (used for ignore-region filtering)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area = _area(a)[:, None]
    return np.where(area > 0, _intersection(a, b) / np.maximum(area, 1e-12), 0.0)


def iou_cost(a, b) -> np.ndarray:
    return 1.0 - iou_matrix(a, b)


def fuse_score(cost: np.ndarray, det_scores) -> np.ndarray:
    """Weight IoU similarity by detection confidence (ByteTrack fuse_score)."""
    if cost.size == 0:
        return cost
    return 1.0 - (1.0 - cost) * np.asarray(det_scores, dtype=float)[None, :]


def gate_groups(cost: np.ndarray, track_groups, det_groups) -> np.ndarray:
    """Forbid matches across class groups (person never takes a car's ID)."""
    if cost.size == 0:
        return cost
    mismatch = np.asarray(track_groups)[:, None] != np.asarray(det_groups)[None, :]
    return np.where(mismatch, GATED_COST, cost)


def linear_assignment(cost: np.ndarray, thresh: float):
    rows, cols = cost.shape
    if cost.size == 0:
        return np.empty((0, 2), dtype=int), list(range(rows)), list(range(cols))
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] <= thresh
    matches = np.stack([r[keep], c[keep]], axis=1).astype(int)
    unmatched_rows = sorted(set(range(rows)) - set(r[keep].tolist()))
    unmatched_cols = sorted(set(range(cols)) - set(c[keep].tolist()))
    return matches, unmatched_rows, unmatched_cols
