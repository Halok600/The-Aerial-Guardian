"""src/eval/det_metrics.py — Simple recall/precision matching for old-vs-new comparisons.

The old model is single-class (person) and the new one has 10 classes, so their
Ultralytics mAPs aren't comparable. Instead both are scored the same way: person
predictions at a fixed confidence vs. pedestrian+people ground truth at IoU 0.5,
ignoring predictions that fall inside VisDrone ignored regions.
"""
from __future__ import annotations

import numpy as np

from src.tracker.matching import ioa_matrix, iou_matrix


def greedy_tp(pred_tlbr, pred_scores, gt_tlbr, iou_thr: float = 0.5) -> int:
    """True positives from greedy score-ordered one-to-one matching at IoU >= iou_thr."""
    pred_tlbr = np.asarray(pred_tlbr, dtype=float).reshape(-1, 4)
    gt_tlbr = np.asarray(gt_tlbr, dtype=float).reshape(-1, 4)
    if len(pred_tlbr) == 0 or len(gt_tlbr) == 0:
        return 0
    order = np.argsort(-np.asarray(pred_scores, dtype=float))
    iou = iou_matrix(pred_tlbr[order], gt_tlbr)
    used = np.zeros(len(gt_tlbr), dtype=bool)
    tp = 0
    for row in iou:
        row = np.where(used, -1.0, row)
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            used[j] = True
            tp += 1
    return tp


def keep_outside_ignore(pred_tlbr, ignore_tlbr, ioa_thr: float = 0.5) -> np.ndarray:
    """Mask of predictions NOT mostly (> ioa_thr of their area) inside an ignored region."""
    pred_tlbr = np.asarray(pred_tlbr, dtype=float).reshape(-1, 4)
    if len(np.asarray(ignore_tlbr).reshape(-1, 4)) == 0:
        return np.ones(len(pred_tlbr), dtype=bool)
    return ioa_matrix(pred_tlbr, ignore_tlbr).max(axis=1) <= ioa_thr
