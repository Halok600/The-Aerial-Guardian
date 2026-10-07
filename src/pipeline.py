"""src/pipeline.py — Detector + camera-motion compensation + tracker loop.

Shared by scripts/03_track.py (rendering) and scripts/05_eval_mot.py
(evaluation and tuning) so every number in the README comes from the exact
code path that produced the demo videos.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np

from src.tracker.byte_tracker import ByteTracker, STrack
from src.tracker.ecc_compensator import ECCCompensator

DetectFn = Callable[[np.ndarray], np.ndarray]
TrackRow = namedtuple("TrackRow", "frame track_id x1 y1 x2 y2 score group")


def _np(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def parse_detections(result) -> np.ndarray:
    """Ultralytics Results -> (N, 6) [x1, y1, x2, y2, score, cls] in original-image pixels."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(
        [_np(boxes.xyxy).reshape(-1, 4), _np(boxes.conf).reshape(-1, 1), _np(boxes.cls).reshape(-1, 1)],
        axis=1).astype(np.float32)


@dataclass
class DetectorSettings:
    imgsz: int = 1280
    conf: float = 0.05      # low on purpose: ByteTrack's stage 2 needs the weak boxes
    iou: float = 0.6
    max_det: int = 1000     # dense VisDrone frames exceed the default 300
    half: bool | None = None
    device: str | int | None = None


class Detector:
    def __init__(self, weights, settings: DetectorSettings | None = None):
        import torch
        from ultralytics import YOLO

        self.s = settings or DetectorSettings()
        cuda = torch.cuda.is_available()
        self.device = self.s.device if self.s.device is not None else (0 if cuda else "cpu")
        self.half = cuda if self.s.half is None else self.s.half
        self.model = YOLO(str(weights))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # `half=` is deprecated in Ultralytics 8.4 (warns on every call); it maps to quantize=16.
        precision = {"quantize": 16} if self.half else {}
        results = self.model.predict(frame, imgsz=self.s.imgsz, conf=self.s.conf, iou=self.s.iou,
                                     max_det=self.s.max_det, device=self.device, verbose=False,
                                     **precision)
        return parse_detections(results[0])


def run_tracking(frames: Iterable[tuple[int, np.ndarray]], detect: DetectFn, tracker: ByteTracker,
                 cmc: ECCCompensator | None) -> Iterator[tuple[int, np.ndarray, list[STrack], np.ndarray]]:
    for frame_id, frame in frames:
        dets = detect(frame)
        warp = cmc.update(frame) if cmc is not None else None
        yield frame_id, frame, tracker.update(dets, warp), dets


def cache_sequence(frames: Iterable[tuple[int, np.ndarray]], detect: DetectFn,
                   cmc: ECCCompensator | None) -> tuple[np.ndarray, np.ndarray]:
    """Run the detector (+ECC) once; tracker sweeps then replay the cache in milliseconds.

    Assumes ``frame_id`` runs 1..F in order (true for ``FramePrefetcher``):
    ``track_cached`` recovers frame ``i`` from ``warps[i - 1]``.
    """
    rows, warps = [], []
    for frame_id, frame in frames:
        dets = detect(frame)
        rows.append(np.c_[np.full((len(dets), 1), frame_id, dtype=float), dets])
        warps.append(cmc.update(frame) if cmc is not None else np.eye(2, 3))
    dets_all = np.concatenate(rows) if rows else np.zeros((0, 7))
    return dets_all, np.stack(warps) if warps else np.zeros((0, 2, 3))


def track_cached(dets_all: np.ndarray, warps: np.ndarray, tracker: ByteTracker,
                 use_cmc: bool = True) -> list[TrackRow]:
    out: list[TrackRow] = []
    for idx in range(len(warps)):
        frame_id = idx + 1
        dets = dets_all[dets_all[:, 0] == frame_id, 1:]
        for t in tracker.update(dets, warps[idx] if use_cmc else None):
            x1, y1, x2, y2 = t.tlbr
            out.append(TrackRow(frame_id, t.track_id, x1, y1, x2, y2, t.score, t.group))
    return out
