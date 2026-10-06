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


def test_scale_params_are_legible_and_monotonic():
    from src.utils.visualizer import _scale_params

    sizes = [(480, 640), (720, 1280), (1080, 1920), (1500, 2000), (1280, 720)]  # last: portrait
    params = [_scale_params(h, w) for h, w in sizes]
    for box_t, label_fs, label_t, hud_fs, hud_t in params:
        assert box_t >= 2 and label_t >= 1 and hud_t >= 1
        assert label_fs >= 0.55 and hud_fs >= 0.55
    # non-decreasing with frame size (landscape sizes in ascending order)
    for small, big in zip(params[:4], params[1:4]):
        assert all(b >= s for s, b in zip(small, big))
    # sizing follows min(h, w): the portrait frame matches its 720p landscape twin
    assert _scale_params(1280, 720) == _scale_params(720, 1280)


def test_box_and_label_are_bold_at_720p():
    frame = np.zeros((720, 1280, 3), np.uint8)
    t = _track(tlbr=(400, 300, 600, 450))
    t.history = deque()                                          # no trail to interfere
    out = TrackVisualizer().draw(frame, [t], {}, 20.0, 1)
    color = GROUP_COLORS_BGR["vehicle"]
    is_color = lambda px: tuple(px) == color
    # bottom box edge at column 500: at least 2 rows carry the group colour
    assert sum(is_color(out[y, 500]) for y in range(440, 460)) >= 2
    # right box edge at row 375: at least 2 columns carry the group colour
    assert sum(is_color(out[375, x]) for x in range(590, 610)) >= 2
    # label chip sits above the box top edge (y=300); count its filled rows at its left padding column
    chip_rows = sum(is_color(out[y, 401]) for y in range(150, 298))
    assert chip_rows >= 14, f"label chip only {chip_rows}px tall"


def test_hud_present_and_confined_without_tracks():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = TrackVisualizer().draw(frame, [], {"person": 3}, 20.0, 1)
    assert out[:60, :600].max() > 0, "HUD text must be drawn"
    assert out[80:].max() == 0, "nothing may be drawn below the HUD"
    assert out[:, 800:].max() == 0, "HUD must not spill far to the right"
