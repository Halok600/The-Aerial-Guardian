"""src/eval/mot_format.py — VisDrone MOT <-> MOTChallenge files for TrackEval.

Evaluation groups (ours, documented in README):
  person  = VisDrone pedestrian(1) + people(2)
  vehicle = car(4) + van(5) + truck(6) + bus(9)
Each group is written as its own TrackEval "benchmark", so every group is
scored on its own GT and its own predictions (TrackEval's fixed class name
"pedestrian" is only a label for "the one class being evaluated").

GT rows use conf=1 (TrackEval drops conf=0 rows), class=1, visibility=1.
VisDrone's per-box truncation/occlusion flags are deliberately NOT used to
filter GT: the official toolkit scores occluded objects too.
Predictions mostly inside VisDrone ignored regions (IoA > 0.5) are dropped,
mirroring how the official toolkit treats those regions. TrackEval's own
distractor preprocessing is disabled (DO_PREPROC=False) because it would
need MOT-style class ids; we do the equivalent filtering here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.visdrone import Box
from src.tracker.matching import ioa_matrix

EVAL_GROUPS: dict[str, tuple[int, ...]] = {"person": (1, 2), "vehicle": (4, 5, 6, 9)}


def limit_frames(frames: dict[int, list[Box]], max_frames: int | None) -> dict[int, list[Box]]:
    """Keep only frames 1..max_frames (None keeps everything)."""
    if max_frames is None:
        return frames
    return {f: boxes for f, boxes in frames.items() if f <= max_frames}


def gt_lines(frames: dict[int, list[Box]], categories) -> list[str]:
    cats = set(categories)
    return [f"{f},{b.track_id},{b.x:.2f},{b.y:.2f},{b.w:.2f},{b.h:.2f},1,1,1"
            for f in sorted(frames) for b in frames[f] if not b.is_ignore and b.category in cats]


def ignore_regions(frames: dict[int, list[Box]]) -> dict[int, np.ndarray]:
    return {f: np.array([[b.x, b.y, b.x + b.w, b.y + b.h] for b in boxes if b.is_ignore]).reshape(-1, 4)
            for f, boxes in frames.items()}


def filter_ignored(rows, ignore: dict[int, np.ndarray], ioa_thr: float = 0.5):
    kept = []
    for r in rows:
        regions = ignore.get(r.frame)
        if regions is not None and len(regions) and ioa_matrix([[r.x1, r.y1, r.x2, r.y2]], regions).max() > ioa_thr:
            continue
        kept.append(r)
    return kept


def pred_lines(rows) -> list[str]:
    return [f"{r.frame},{r.track_id},{r.x1:.2f},{r.y1:.2f},{r.x2 - r.x1:.2f},{r.y2 - r.y1:.2f},"
            f"{r.score:.4f},-1,-1,-1" for r in rows]


def write_gt(root: Path, benchmark: str, seq: str, lines: list[str], length: int) -> None:
    seq_dir = Path(root) / f"{benchmark}-test" / seq
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt" / "gt.txt").write_text("".join(f"{l}\n" for l in lines))
    (seq_dir / "seqinfo.ini").write_text(f"[Sequence]\nname={seq}\nseqLength={length}\n")


def write_seqmap(root: Path, benchmark: str, seqs: list[str]) -> None:
    path = Path(root) / "seqmaps" / f"{benchmark}-test.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name\n" + "".join(f"{s}\n" for s in seqs))


def write_tracker(root: Path, benchmark: str, tracker: str, seq: str, lines: list[str]) -> None:
    path = Path(root) / f"{benchmark}-test" / tracker / "data" / f"{seq}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{l}\n" for l in lines))
