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


def test_cache_then_track_matches_live_tracking():
    dets_all, warps = cache_sequence(_frames(), _moving_detector(), cmc=None)
    assert dets_all.shape == (10, 7) and warps.shape == (10, 2, 3)
    rows = track_cached(dets_all, warps, ByteTracker())
    assert len(rows) == 10 and {r.track_id for r in rows} == {1} and rows[0].group == "person"
