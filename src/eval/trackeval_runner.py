"""src/eval/trackeval_runner.py — Run TrackEval (HOTA, CLEAR, Identity) on MOTChallenge-format files.

Returns {tracker: {HOTA, MOTA, IDF1, IDSW, gt_count, "per_seq": {seq: {HOTA, MOTA, IDF1, IDSW, gt_count}}}}.
A per-sequence entry with gt_count == 0 (no ground truth of the evaluated group) has None metrics.
The top-level numbers are TrackEval's COMBINED_SEQ: counts (TP/FN/FP/IDSW, ...) are
pooled over all sequences, so long and crowded sequences weigh more than short ones.
Use per_seq for the unweighted per-sequence view.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _patch_numpy_aliases() -> None:
    # TrackEval still references np.float / np.int / np.bool (removed in NumPy 1.24).
    for name, typ in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, typ)


def _summary(r: dict, null_without_gt: bool = False) -> dict:
    """HOTA/MOTA/IDF1 (percent) + IDSW + gt_count. With `null_without_gt`, a sequence that has no
    ground-truth objects of this group gets None metrics (TrackEval would report a meaningless 0.0)."""
    gt_count = int(r["Count"]["GT_Dets"])
    if null_without_gt and gt_count == 0:
        return {"HOTA": None, "MOTA": None, "IDF1": None, "IDSW": None, "gt_count": 0}
    return {"HOTA": float(np.mean(r["HOTA"]["HOTA"])) * 100,
            "MOTA": float(r["CLEAR"]["MOTA"]) * 100,
            "IDF1": float(r["Identity"]["IDF1"]) * 100,
            "IDSW": int(r["CLEAR"]["IDSW"]),
            "gt_count": gt_count}


def run_trackeval(gt_root: Path, trackers_root: Path, benchmark: str, trackers: list[str]) -> dict:
    _patch_numpy_aliases()
    import trackeval

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
                     "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                     "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True})
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trackers_root),
                   "BENCHMARK": benchmark, "SPLIT_TO_EVAL": "test", "TRACKERS_TO_EVAL": trackers,
                   "DO_PREPROC": False, "TRACKER_SUB_FOLDER": "data",
                   "SEQMAP_FILE": str(Path(gt_root) / "seqmaps" / f"{benchmark}-test.txt")})
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = trackeval.Evaluator(eval_cfg).evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    out = {}
    for name in trackers:
        per = res["MotChallenge2DBox"][name]
        out[name] = {**_summary(per["COMBINED_SEQ"]["pedestrian"]),
                     "per_seq": {seq: _summary(r["pedestrian"], null_without_gt=True) for seq, r in per.items() if seq != "COMBINED_SEQ"}}
    return out
