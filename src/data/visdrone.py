"""src/data/visdrone.py — VisDrone DET/MOT annotation parsing and YOLO conversion.

Fixes vs the original 01_preprocess.py:
  * all 10 object classes are kept (the old pipeline dropped "people", which
    taught the detector that seated/riding persons are background);
  * ignored regions (category 0, or score 0) and "others" (category 11) are
    returned as ignore boxes so the builder can grey them out instead of
    leaving unlabeled objects in training images.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IGNORE_CATEGORIES = frozenset({0, 11})
MASK_VALUE = (114, 114, 114)
DEFAULT_CLASS_MAP: dict[int, int] = {c: c - 1 for c in range(1, 11)}
PERSON_ONLY_CLASS_MAP: dict[int, int] = {1: 0, 2: 0}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    category: int
    score: int = 1
    track_id: int = -1

    @property
    def is_ignore(self) -> bool:
        return self.score == 0 or self.category in IGNORE_CATEGORIES


def _fields(line: str) -> list[str]:
    return [p for p in line.strip().strip(",").split(",") if p != ""]


def parse_det_file(path: Path) -> list[Box]:
    """DET rows: x,y,w,h,score,category,truncation,occlusion."""
    boxes: list[Box] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = _fields(line)
        if len(p) < 6:
            continue
        boxes.append(Box(float(p[0]), float(p[1]), float(p[2]), float(p[3]),
                         category=int(p[5]), score=int(p[4])))
    return boxes


def parse_mot_file(path: Path) -> dict[int, list[Box]]:
    """MOT rows: frame,id,x,y,w,h,score,category,truncation,occlusion."""
    frames: dict[int, list[Box]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = _fields(line)
        if len(p) < 8:
            continue
        frames[int(p[0])].append(Box(float(p[2]), float(p[3]), float(p[4]), float(p[5]),
                                     category=int(p[7]), score=int(p[6]), track_id=int(p[1])))
    return dict(frames)


def to_yolo_line(box: Box, img_w: int, img_h: int,
                 class_map: dict[int, int] = DEFAULT_CLASS_MAP,
                 min_size: float = 2.0) -> str | None:
    """YOLO label line, or None for ignore boxes, unmapped classes and boxes < min_size px after clipping."""
    if box.is_ignore or box.category not in class_map:
        return None
    x1, y1 = max(0.0, box.x), max(0.0, box.y)
    x2, y2 = min(float(img_w), box.x + box.w), min(float(img_h), box.y + box.h)
    w, h = x2 - x1, y2 - y1
    if w < min_size or h < min_size:
        return None
    cls = class_map[box.category]
    return f"{cls} {(x1 + w / 2) / img_w:.6f} {(y1 + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def mask_ignored(image: np.ndarray, boxes) -> np.ndarray:
    """Return a copy of `image` with every ignore box filled with MASK_VALUE."""
    out = image.copy()
    img_h, img_w = out.shape[:2]
    for b in boxes:
        if not b.is_ignore:
            continue
        x1, y1 = max(0, math.floor(b.x)), max(0, math.floor(b.y))
        x2, y2 = min(img_w, math.ceil(b.x + b.w)), min(img_h, math.ceil(b.y + b.h))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = MASK_VALUE
    return out
