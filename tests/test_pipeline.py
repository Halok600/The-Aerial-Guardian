from types import SimpleNamespace

import numpy as np

from src.pipeline import cache_sequence, parse_detections, run_tracking, track_cached
from src.tracker.byte_tracker import ByteTracker


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = (np.array(v, float) for v in (xyxy, conf, cls))

    def __len__(self):
        return len(self.conf)


def test_parse_detections_rows():
    res = SimpleNamespace(boxes=_Boxes([[1, 2, 3, 4]], [0.5], [3]))
    assert parse_detections(res).tolist() == [[1, 2, 3, 4, 0.5, 3]]
    assert parse_detections(SimpleNamespace(boxes=_Boxes(np.zeros((0, 4)), [], []))).shape == (0, 6)


def _frames(n=10):
    return [(i + 1, np.zeros((100, 200, 3), np.uint8)) for i in range(n)]


def _moving_detector():
    state = {"i": 0}

    def detect(_frame):
        i = state["i"]
        state["i"] += 1
        return np.array([[10 + 3 * i, 20, 40 + 3 * i, 80, 0.9, 0]], float)
    return detect


def test_run_tracking_yields_consistent_ids():
    outs = list(run_tracking(_frames(), _moving_detector(), ByteTracker(), cmc=None))
    assert [fid for fid, *_ in outs] == list(range(1, 11))
    assert {t.track_id for _, _, tracks, _ in outs for t in tracks} == {1}


class _DriftCmc:
    """Deterministic non-identity stand-in for ECCCompensator: a slow camera drift."""

    def __init__(self):
        self.n = 0

    def update(self, _frame):
        self.n += 1
        return np.array([[1.0, 0.0, 1.5], [0.0, 1.0, -0.5]], np.float32) if self.n > 1 else np.eye(2, 3, dtype=np.float32)


def _scripted_detector():
    """Two objects (person cls 0 and vehicle cls 3), weak scores on some frames, an empty frame at i=6."""
    state = {"i": 0}

    def detect(_frame):
        i = state["i"]
        state["i"] += 1
        if i == 6:
            return np.zeros((0, 6), np.float32)
        person_score = 0.9 if i % 4 else 0.15      # drops into ByteTrack's low-score stage
        return np.array([[10 + 3 * i, 20, 40 + 3 * i, 80, person_score, 0],
                         [150 - 2 * i, 10, 190 - 2 * i, 60, 0.8, 3]], np.float32)
    return detect


def _live_rows(cmc):
    rows = []
    for fid, _f, tracks, _d in run_tracking(_frames(14), _scripted_detector(), ByteTracker(), cmc):
        for t in tracks:
            rows.append((fid, t.track_id, *[round(float(v), 3) for v in t.tlbr], round(float(t.score), 3), t.group))
    return rows


def _rounded(rows):
    return [(r.frame, r.track_id, *[round(float(v), 3) for v in (r.x1, r.y1, r.x2, r.y2, r.score)], r.group)
            for r in rows]


def test_cache_then_track_matches_live_tracking():
    live = _live_rows(_DriftCmc())
    dets_all, warps = cache_sequence(_frames(14), _scripted_detector(), _DriftCmc())
    assert warps.shape == (14, 2, 3) and not np.allclose(warps[3], np.eye(2, 3))
    cached = _rounded(track_cached(dets_all, warps, ByteTracker()))
    assert live and live == cached
    assert {g for *_, g in live} == {"person", "vehicle"}


def test_cache_without_cmc_matches_live_without_cmc():
    live = _live_rows(None)
    dets_all, warps = cache_sequence(_frames(14), _scripted_detector(), None)
    assert np.allclose(warps, np.eye(2, 3))
    cached = _rounded(track_cached(dets_all, warps, ByteTracker(), use_cmc=False))
    assert live and live == cached
    # warps cached from a drifting camera are ignored when use_cmc=False
    _, drift = cache_sequence(_frames(14), _scripted_detector(), _DriftCmc())
    assert _rounded(track_cached(dets_all, drift, ByteTracker(), use_cmc=False)) == live


def test_empty_input():
    dets_all, warps = cache_sequence([], _scripted_detector(), None)
    assert dets_all.shape == (0, 7) and warps.shape == (0, 2, 3)
    assert track_cached(dets_all, warps, ByteTracker()) == []
