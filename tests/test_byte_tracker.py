import numpy as np
import pytest

from src.tracker.byte_tracker import ByteTracker, Detection, STrack, TrackerConfig
from src.tracker.kalman_filter import KalmanFilterXYAH
from tests.helpers import ROOT


def box(cx, cy, w, h, score=0.9, cls=0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, cls]


def run(tracker, frames):
    return [tracker.update(np.array(f, dtype=float).reshape(-1, 6)) for f in frames]


def all_ids(outs):
    return {t.track_id for out in outs for t in out}


def test_constant_motion_keeps_one_id_from_first_frame():
    outs = run(ByteTracker(), [[box(100 + 3 * i, 200, 30, 60)] for i in range(30)])
    assert all(len(o) == 1 for o in outs)
    assert all_ids(outs) == {1}


def test_new_track_is_tentative_until_second_match():
    frames = [[] for _ in range(4)] + [[box(100 + 3 * i, 200, 30, 60)] for i in range(5)]
    outs = run(ByteTracker(), frames)
    assert outs[4] == []
    assert [t.track_id for t in outs[5]] == [1]


def test_single_frame_false_positive_never_gets_an_id():
    frames = [[box(100 + 3 * i, 200, 30, 60)] for i in range(30)]
    frames[10].append(box(600, 600, 20, 40))
    tracker = ByteTracker()
    outs = run(tracker, frames)
    assert all_ids(outs) == {1} and tracker.total_ids == 1


def test_occlusion_recovers_same_id():
    frames = []
    for i in range(45):
        frames.append([] if 20 <= i < 30 else [box(100 + 2 * i, 200, 30, 60)])
    outs = run(ByteTracker(), frames)
    assert all(o == [] for o in outs[20:30])
    assert {t.track_id for t in outs[44]} == {1}


def test_low_score_boxes_continue_existing_track():
    frames = [[box(100 + 3 * i, 200, 30, 60, score=0.9 if i < 10 else 0.2)] for i in range(20)]
    outs = run(ByteTracker(), frames)
    assert all(len(o) == 1 for o in outs) and all_ids(outs) == {1}


def test_pedestrian_people_flicker_keeps_id_and_majority_class():
    pattern = [0, 0, 1]
    frames = [[box(100 + 3 * i, 200, 30, 60, cls=pattern[i % 3])] for i in range(30)]
    outs = run(ByteTracker(), frames)
    assert all_ids(outs) == {1}
    assert outs[-1][0].cls == 0 and outs[-1][0].group == "person"


def test_crossing_person_and_car_do_not_swap():
    frames = [[box(100 + 5 * i, 200, 20, 40, cls=0), box(300 - 5 * i, 205, 40, 30, cls=3)]
              for i in range(41)]
    tracker = ByteTracker()
    outs = run(tracker, frames)
    final = {t.group: (t.tlbr[0] + t.tlbr[2]) / 2 for t in outs[-1]}
    assert final["person"] == pytest.approx(300, abs=10)
    assert final["vehicle"] == pytest.approx(100, abs=10)
    assert tracker.total_ids == 2


def test_unique_counts_need_min_hits():
    frames = [[box(100, 100, 20, 40), box(200, 100, 20, 40), box(400, 300, 60, 40, cls=3)]] * 5
    tracker = ByteTracker()
    run(tracker, frames[:2])
    assert tracker.unique_counts() == {"person": 0, "vehicle": 0, "two-wheeler": 0, "tricycle": 0}
    run(tracker, frames[2:])
    assert tracker.unique_counts() == {"person": 2, "vehicle": 1, "two-wheeler": 0, "tricycle": 0}


def test_apply_warp_translates_track():
    kf = KalmanFilterXYAH()
    t = STrack(Detection(np.array([0.0, 0.0, 20.0, 40.0]), 0.9, 0))
    t.activate(kf, 1)
    t.apply_warp(np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0]]))
    assert np.allclose(t.tlbr, [10.0, -5.0, 30.0, 35.0])


def test_panning_camera_with_warp_keeps_id():
    frames, warps = [], []
    for i in range(20):
        frames.append([box(100 + 8 * i, 200, 10, 20)])   # static object, camera pans
        warps.append(np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 0.0]]))
    tracker = ByteTracker()
    outs = [tracker.update(np.array(f, float), w) for f, w in zip(frames, warps)]
    assert all_ids(outs) == {1}


def test_empty_frames_and_frame_rate_scaling():
    assert ByteTracker().update(np.zeros((0, 6))) == []
    assert ByteTracker(TrackerConfig(max_time_lost=30), frame_rate=60).max_time_lost == 60


def test_config_from_yaml(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("track_high: 0.3\n")
    cfg = TrackerConfig.from_yaml(p)
    assert cfg.track_high == 0.3 and cfg.new_track == TrackerConfig().new_track
    p.write_text("bogus: 1\n")
    with pytest.raises(ValueError):
        TrackerConfig.from_yaml(p)
    assert TrackerConfig.from_yaml(ROOT / "configs" / "tracker.yaml").track_low == 0.1


def test_dataclass_defaults_equal_tracker_yaml():
    assert TrackerConfig() == TrackerConfig.from_yaml(ROOT / "configs" / "tracker.yaml")


def test_degenerate_boxes_are_dropped_before_tracking():
    good = box(100, 200, 30, 60)
    zero_height = [300.0, 400.0, 340.0, 400.0, 0.9, 0]    # y2 == y1
    inverted = [500.0, 600.0, 480.0, 640.0, 0.9, 0]       # x2 < x1
    tracker = ByteTracker()
    outs = run(tracker, [[good, zero_height, inverted],
                         [box(103, 200, 30, 60), zero_height, inverted]])
    assert [t.track_id for t in outs[-1]] == [1]
    assert tracker.total_ids == 1
    assert len(tracker.tracked) == 1


def test_class_group_gating_blocks_cross_group_takeover():
    # A car appears exactly where the vanished person would be predicted: without
    # group gating the car would be matched to the person's track.
    frames = ([[box(100 + 2 * i, 100, 30, 30, cls=0)] for i in range(5)]
              + [[box(110 + 2 * i, 100, 30, 30, cls=3)] for i in range(5)])
    tracker = ByteTracker()
    outs = run(tracker, frames)
    assert [(t.group, t.track_id) for t in outs[-1]] == [("vehicle", 2)]
    assert tracker.total_ids == 2
