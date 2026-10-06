from collections import deque
from types import SimpleNamespace

import numpy as np

from src.utils.classes import GROUP_COLORS_BGR
from src.utils.visualizer import TrackVisualizer, hud_lines, label_for


def _track(tid=12, cls=3, group="vehicle", tlbr=(100, 100, 160, 140)):
    hist = deque([(60 + 5 * i, 120.0) for i in range(10)])
    return SimpleNamespace(track_id=tid, cls=cls, group=group, tlbr=np.array(tlbr, float), history=hist)


def test_label_text():
    assert label_for(_track()) == "car ID:12"
    assert label_for(_track(tid=5, cls=0, group="person")) == "pedestrian ID:5"


def test_hud_lines():
    counts = {"person": 23, "vehicle": 41, "two-wheeler": 2, "tricycle": 0}
    assert hud_lines(counts, 17.25, 88, 30) == [
        "Unique  Persons: 23  Vehicles: 41  Two-wheelers: 2  Tricycles: 0",
        "Frame: 88  |  Active: 30  |  FPS: 17.2",
    ]


def test_draw_returns_new_annotated_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    out = TrackVisualizer().draw(frame, [_track()], {"vehicle": 1}, 20.0, 1)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert frame.max() == 0, "input must not be mutated"
    assert tuple(out[140, 130]) == GROUP_COLORS_BGR["vehicle"]   # bottom box edge
    assert out[:40, :200].max() > 0                             # HUD drawn
