"""src/tracker/byte_tracker.py — ByteTrack MOT, written from scratch (v3).

What changed vs v2 (and why):
  * Tentative tracks: a new track gets an ID only when it is matched again on
    the next frame, so one-frame false positives no longer burn IDs.
  * Stage 1 matches high-score detections against tracked AND lost tracks
    together (reference ByteTrack), instead of offering lost tracks only the
    leftovers in a later stage.
  * Class-group gating: association cost is infinite across groups
    (person / vehicle / two-wheeler / tricycle). Inside a group the track's
    displayed class is a majority vote, so pedestrian<->people flicker keeps
    the same ID.
  * Duplicate removal between tracked and lost pools.
  * Unique per-group counting (a track counts after `min_hits_for_count`).
  * The per-instance ID counter (not class-level) keeps trackers independent.
  * Degenerate detections (zero or negative width/height) are dropped on
    input: the Kalman filter works in (cx, cy, aspect, height) space and a
    non-positive height makes its innovation covariance singular.

Per frame:
  1. predict all tracks (+ camera-motion warp)
  2. stage 1: high dets  <-> confirmed tracked + lost   (score-fused IoU, gated)
  3. stage 2: low dets   <-> still-unmatched tracked     (plain IoU >= 0.5, gated)
  4. stage 3: leftover high dets <-> tentative tracks    (confirm or remove)
  5. start tentative tracks from leftover dets with score >= new_track
  6. expire lost tracks, remove duplicates, update unique counts
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, fields
from enum import IntEnum
from pathlib import Path

import numpy as np
import yaml

from src.utils.classes import GROUP_ORDER, group_of
from .kalman_filter import KalmanFilterXYAH, tlbr_to_xyah, xyah_to_tlbr
from .matching import fuse_score, gate_groups, iou_cost, iou_matrix, linear_assignment


class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class TrackerConfig:
    track_high: float = 0.4
    track_low: float = 0.1
    new_track: float = 0.5
    match_thresh: float = 0.8
    second_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    max_time_lost: int = 30
    fuse_score: bool = True
    class_vote_window: int = 30
    min_hits_for_count: int = 3

    @classmethod
    def from_yaml(cls, path) -> "TrackerConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown tracker config keys: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True, eq=False)
class Detection:
    tlbr: np.ndarray
    score: float
    cls: int

    @property
    def group(self) -> str:
        return group_of(self.cls)


class STrack:
    """One tracked object: Kalman state, class votes and a centre-point history for trails."""

    def __init__(self, det: Detection, vote_window: int = 30, history_len: int = 60):
        self._init_tlbr = np.asarray(det.tlbr, dtype=float).copy()
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.track_id = 0
        self.state = TrackState.NEW
        self.is_activated = False
        self.score = float(det.score)
        self.group = det.group
        self._votes: deque[int] = deque([int(det.cls)], maxlen=vote_window)
        self.history: deque[tuple[float, float]] = deque(maxlen=history_len)
        self.frame_id = 0
        self.start_frame = 0
        self.hits = 0

    # --- geometry -----------------------------------------------------------
    @property
    def tlbr(self) -> np.ndarray:
        return self._init_tlbr.copy() if self.mean is None else xyah_to_tlbr(self.mean)

    @property
    def cls(self) -> int:
        counts = Counter(self._votes)
        best = max(counts.values())
        for c in reversed(self._votes):        # ties -> most recent class
            if counts[c] == best:
                return c
        return self._votes[-1]

    def _record(self) -> None:
        x1, y1, x2, y2 = self.tlbr
        self.history.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    # --- lifecycle ------------------------------------------------------------
    def activate(self, kf: KalmanFilterXYAH, frame_id: int) -> None:
        self.mean, self.covariance = kf.initiate(tlbr_to_xyah(self._init_tlbr))
        self.state = TrackState.TRACKED
        self.frame_id = self.start_frame = frame_id
        self.hits = 1
        self._record()

    def confirm(self, track_id: int) -> None:
        self.track_id = track_id
        self.is_activated = True

    def predict(self, kf: KalmanFilterXYAH) -> None:
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0.0                      # freeze height velocity while lost
        self.mean, self.covariance = kf.predict(mean, self.covariance)

    def update(self, kf: KalmanFilterXYAH, det: Detection, frame_id: int) -> None:
        self.mean, self.covariance = kf.update(self.mean, self.covariance, tlbr_to_xyah(det.tlbr))
        self.score = float(det.score)
        self._votes.append(int(det.cls))
        self.frame_id = frame_id
        self.hits += 1
        self.state = TrackState.TRACKED
        self._record()

    def apply_warp(self, warp: np.ndarray) -> None:
        """Move the Kalman state with the camera: rotate+translate position, rotate velocity."""
        rot, trans = warp[:2, :2], warp[:2, 2]
        T = np.eye(8)
        T[0:2, 0:2] = rot
        T[4:6, 4:6] = rot
        self.mean = T @ self.mean
        self.mean[0:2] += trans
        self.covariance = T @ self.covariance @ T.T

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


def _joint(a: list[STrack], b: list[STrack]) -> list[STrack]:
    seen = {id(t) for t in a}
    return a + [t for t in b if id(t) not in seen]


def _sub(a: list[STrack], b: list[STrack]) -> list[STrack]:
    drop = {id(t) for t in b}
    return [t for t in a if id(t) not in drop]


def _remove_duplicates(tracked: list[STrack], lost: list[STrack]):
    if not tracked or not lost:
        return tracked, lost
    iou = iou_matrix(np.array([t.tlbr for t in tracked]), np.array([t.tlbr for t in lost]))
    dup_t, dup_l = set(), set()
    for p, q in zip(*np.where(iou > 0.85)):
        if tracked[p].group != lost[q].group:
            continue
        age_t = tracked[p].frame_id - tracked[p].start_frame
        age_l = lost[q].frame_id - lost[q].start_frame
        if age_t > age_l:
            dup_l.add(q)
        else:
            dup_t.add(p)
    return ([t for i, t in enumerate(tracked) if i not in dup_t],
            [t for i, t in enumerate(lost) if i not in dup_l])


class ByteTracker:
    def __init__(self, config: TrackerConfig | None = None, frame_rate: float = 30.0):
        self.cfg = config or TrackerConfig()
        self.max_time_lost = int(round(self.cfg.max_time_lost * frame_rate / 30.0))
        self.kf = KalmanFilterXYAH()
        self.reset()

    def reset(self) -> None:
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.frame_id = 0
        self._last_id = 0
        self._counted: dict[int, str] = {}

    @property
    def total_ids(self) -> int:
        return self._last_id

    def unique_counts(self) -> dict[str, int]:
        counts = {g: 0 for g in GROUP_ORDER}
        for g in self._counted.values():
            counts[g] += 1
        return counts

    def _next_id(self) -> int:
        self._last_id += 1
        return self._last_id

    def _to_detections(self, dets) -> list[Detection]:
        arr = np.asarray(dets, dtype=float).reshape(-1, 6)
        return [Detection(row[:4].copy(), float(row[4]), int(row[5]))
                for row in arr
                if row[2] > row[0] and row[3] > row[1] and row[4] >= self.cfg.track_low]

    def _cost(self, tracks: list[STrack], dets: list[Detection], fuse: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        cost = iou_cost(np.array([t.tlbr for t in tracks]), np.array([d.tlbr for d in dets]))
        if fuse:
            cost = fuse_score(cost, [d.score for d in dets])
        return gate_groups(cost, [t.group for t in tracks], [d.group for d in dets])

    def update(self, dets, warp: np.ndarray | None = None) -> list[STrack]:
        self.frame_id += 1
        cfg = self.cfg
        detections = self._to_detections(dets)
        high = [d for d in detections if d.score >= cfg.track_high]
        low = [d for d in detections if d.score < cfg.track_high]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        pool = _joint([t for t in self.tracked if t.is_activated], self.lost)
        for t in pool + unconfirmed:
            t.predict(self.kf)
            if warp is not None:
                t.apply_warp(warp)

        activated: list[STrack] = []
        refound: list[STrack] = []
        lost_now: list[STrack] = []
        removed: list[STrack] = []

        # Stage 1: high-score detections vs confirmed tracked + lost tracks.
        matches, u_pool, u_high = linear_assignment(self._cost(pool, high, cfg.fuse_score), cfg.match_thresh)
        for ti, di in matches:
            t = pool[ti]
            was_tracked = t.state == TrackState.TRACKED
            t.update(self.kf, high[di], self.frame_id)
            (activated if was_tracked else refound).append(t)

        # Stage 2: low-score detections keep still-unmatched *tracked* tracks alive.
        remaining = [pool[i] for i in u_pool if pool[i].state == TrackState.TRACKED]
        matches, u_rem, _ = linear_assignment(self._cost(remaining, low, False), cfg.second_match_thresh)
        for ti, di in matches:
            remaining[ti].update(self.kf, low[di], self.frame_id)
            activated.append(remaining[ti])
        for i in u_rem:
            remaining[i].mark_lost()
            lost_now.append(remaining[i])

        # Stage 3: tentative tracks are confirmed by a second match, else removed.
        leftover = [high[i] for i in u_high]
        matches, u_unconf, u_left = linear_assignment(
            self._cost(unconfirmed, leftover, cfg.fuse_score), cfg.unconfirmed_match_thresh)
        for ti, di in matches:
            t = unconfirmed[ti]
            t.update(self.kf, leftover[di], self.frame_id)
            t.confirm(self._next_id())
            activated.append(t)
        for i in u_unconf:
            unconfirmed[i].mark_removed()
            removed.append(unconfirmed[i])

        # New tentative tracks (confirmed immediately on the very first frame).
        for i in u_left:
            d = leftover[i]
            if d.score < cfg.new_track:
                continue
            t = STrack(d, cfg.class_vote_window)
            t.activate(self.kf, self.frame_id)
            if self.frame_id == 1:
                t.confirm(self._next_id())
            activated.append(t)

        for t in self.lost:
            if self.frame_id - t.frame_id > self.max_time_lost:
                t.mark_removed()
                removed.append(t)

        self.tracked = [t for t in self.tracked if t.state == TrackState.TRACKED]
        self.tracked = _joint(_joint(self.tracked, activated), refound)
        self.lost = _sub(self.lost, self.tracked) + lost_now
        self.lost = _sub(self.lost, removed)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)

        for t in self.tracked:
            if t.is_activated and t.hits >= cfg.min_hits_for_count and t.track_id not in self._counted:
                self._counted[t.track_id] = t.group
        return [t for t in self.tracked if t.is_activated]
