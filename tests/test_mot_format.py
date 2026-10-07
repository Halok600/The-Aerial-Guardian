from pathlib import Path

import numpy as np
import pytest

from src.data.visdrone import Box, parse_mot_file
from src.eval.mot_format import (
    EVAL_GROUPS, filter_ignored, gt_lines, ignore_regions, limit_frames, pred_lines,
    write_gt, write_seqmap, write_tracker,
)
from src.pipeline import TrackRow
from src.utils.classes import group_of

FIX = Path(__file__).parent / "fixtures" / "visdrone" / "mot_sample.txt"


def test_gt_lines_per_group():
    frames = parse_mot_file(FIX)
    assert gt_lines(frames, EVAL_GROUPS["person"]) == [
        "1,1,10.00,10.00,20.00,40.00,1,1,1", "2,1,12.00,10.00,20.00,40.00,1,1,1"]
    assert len(gt_lines(frames, EVAL_GROUPS["vehicle"])) == 2


def test_gt_lines_ignore_visdrone_occlusion_flags():
    # The fixture's last row has occlusion=1; it must still be a normal GT row (conf=1).
    frames = parse_mot_file(FIX)
    assert "2,2,104.00,50.00,60.00,30.00,1,1,1" in gt_lines(frames, EVAL_GROUPS["vehicle"])


def test_ignore_filter_and_pred_lines():
    ignore = ignore_regions(parse_mot_file(FIX))
    rows = [TrackRow(1, 7, 510, 10, 530, 50, 0.9, "person"),   # inside ignored region
            TrackRow(1, 8, 10, 10, 30, 50, 0.8, "person")]
    kept = filter_ignored(rows, ignore)
    assert [r.track_id for r in kept] == [8]
    assert pred_lines(kept) == ["1,8,10.00,10.00,20.00,40.00,0.8000,-1,-1,-1"]


def test_filter_ignored_uses_ioa_threshold_and_frame():
    ignore = {1: np.array([[500.0, 0.0, 600.0, 100.0]])}
    at_40pct = TrackRow(1, 1, 470, 10, 520, 50, 0.9, "person")   # x 500..520 of 470..520 = 40% inside
    at_60pct = TrackRow(1, 1, 480, 10, 530, 50, 0.9, "person")   # x 500..530 of 480..530 = 60% inside
    other_frame = at_60pct._replace(frame=2)                      # frame 2 has no ignored regions
    assert filter_ignored([at_40pct, other_frame], ignore) == [at_40pct, other_frame]
    assert filter_ignored([at_60pct], ignore) == []


def test_limit_frames_drops_later_frames():
    frames = parse_mot_file(FIX)
    assert sorted(limit_frames(frames, 1)) == [1]
    assert limit_frames(frames, None) is frames
    assert sorted(limit_frames(frames, 5)) == [1, 2]


def test_write_gt_seqinfo_has_requested_length(tmp_path):
    write_gt(tmp_path, "T", "s", ["1,1,1.00,1.00,2.00,2.00,1,1,1"], 7)
    ini = (tmp_path / "T-test" / "s" / "seqinfo.ini").read_text()
    assert "seqLength=7" in ini


def _run(tmp_path, gt, preds, length):
    pytest.importorskip("trackeval")
    from src.eval.trackeval_runner import run_trackeval

    write_gt(tmp_path / "gt", "T", "seq1", gt, length)
    write_seqmap(tmp_path / "gt", "T", ["seq1"])
    write_tracker(tmp_path / "trk", "T", "trk", "seq1", preds)
    return run_trackeval(tmp_path / "gt", tmp_path / "trk", "T", ["trk"])["trk"]


def test_trackeval_perfect_tracker_scores_100(tmp_path):
    pytest.importorskip("trackeval")
    from src.eval.trackeval_runner import run_trackeval

    lines = [f"{f},{i},{10 + 50 * i + f},20.00,20.00,40.00,1,1,1" for f in range(1, 6) for i in (1, 2)]
    preds = [f"{f},{i},{10 + 50 * i + f},20.00,20.00,40.00,0.9,-1,-1,-1" for f in range(1, 6) for i in (1, 2)]
    write_gt(tmp_path / "gt", "T", "seq1", lines, 5)
    write_seqmap(tmp_path / "gt", "T", ["seq1"])
    write_tracker(tmp_path / "trk", "T", "perfect", "seq1", preds)
    res = run_trackeval(tmp_path / "gt", tmp_path / "trk", "T", ["perfect"])
    assert res["perfect"]["MOTA"] == pytest.approx(100.0)
    assert res["perfect"]["IDF1"] == pytest.approx(100.0)
    assert res["perfect"]["HOTA"] > 99.0


def test_trackeval_one_id_switch_and_one_miss_hand_computed(tmp_path):
    # Object A (5 frames): predicted as id 1 on frames 1-2, id 9 on frames 3-5  -> 1 ID switch
    # Object B (5 frames): predicted as id 2 on frames 1-4, missed on frame 5   -> 1 FN
    box = lambda f, i: f"{10 + 50 * i + f},20.00,20.00,40.00"
    gt = [f"{f},{i},{box(f, i)},1,1,1" for f in range(1, 6) for i in (1, 2)]
    pred_id = {(1, 1): 1, (2, 1): 1, (3, 1): 9, (4, 1): 9, (5, 1): 9,
               (1, 2): 2, (2, 2): 2, (3, 2): 2, (4, 2): 2}
    preds = [f"{f},{pid},{box(f, i)},0.9,-1,-1,-1" for (f, i), pid in sorted(pred_id.items())]
    r = _run(tmp_path, gt, preds, 5)
    # GT=10, TP=9, FN=1, FP=0, IDSW=1 -> MOTA = (10 - 1 - 0 - 1) / 10 = 80
    assert r["MOTA"] == pytest.approx(80.0)
    assert r["IDSW"] == 1
    # best identity matching: A<->9 (3 frames) + B<->2 (4 frames): IDTP=7, IDFN=3, IDFP=2
    # IDF1 = 2*7 / (2*7 + 3 + 2) = 14/19
    assert r["IDF1"] == pytest.approx(100 * 14 / 19)


def test_trackeval_zero_confidence_gt_rows_are_ignored(tmp_path):
    # GT row with conf 0 must not count as a missed detection.
    gt = ["1,1,10.00,20.00,20.00,40.00,1,1,1", "1,2,200.00,20.00,20.00,40.00,0,1,1"]
    preds = ["1,1,10.00,20.00,20.00,40.00,0.9,-1,-1,-1"]
    r = _run(tmp_path, gt, preds, 1)
    assert r["MOTA"] == pytest.approx(100.0)


def test_gt_lines_drop_boxes_inside_ignore_regions():
    # Same IoA > 0.5 rule as for predictions: GT mostly inside an ignored region is discarded too.
    frames = parse_mot_file(FIX)
    frames[1] = frames[1] + [Box(500, 10, 20, 40, category=1, score=1, track_id=9)]   # fully inside (500,0,100,100)
    frames[1] = frames[1] + [Box(480, 10, 50, 40, category=1, score=1, track_id=8)]   # 60% inside
    assert not any(l.split(",")[1] in ("9", "8") for l in gt_lines(frames, EVAL_GROUPS["person"]))
    # ignore regions of the *same frame* only: frame 2 has no region, so the frame-1 box position is kept there
    frames[2] = frames[2] + [Box(500, 10, 20, 40, category=1, score=1, track_id=7)]
    assert any(l.startswith("2,7,") for l in gt_lines(frames, EVAL_GROUPS["person"]))


def _synthetic_split(root, n=5):
    seq = root / "sequences" / "s1"
    seq.mkdir(parents=True)
    for i in range(1, n + 1):
        (seq / f"{i:07d}.jpg").write_bytes(b"")
    (root / "annotations").mkdir()
    lines = []
    for f in range(1, n + 1):
        lines += [f"{f},1,{10 + f},10,20,40,1,1,0,0",          # person, clear of the ignored region
                  f"{f},2,500,20,20,40,1,1,0,0",              # person inside the ignored region
                  f"{f},3,100,200,60,30,1,4,0,0",             # car
                  f"{f},-1,480,0,100,100,0,0,0,0"]            # ignored region
    (root / "annotations" / "s1.txt").write_text("\n".join(lines) + "\n")


def test_gt_as_prediction_scores_100_with_ignore_region_over_a_gt_box(tmp_path):
    pytest.importorskip("trackeval")
    from tests.helpers import load_script
    from src.eval.trackeval_runner import run_trackeval

    mod = load_script("05_eval_mot.py")
    _synthetic_split(tmp_path / "split")
    ignores = mod.prepare_gt(tmp_path / "split", ["s1"], tmp_path / "gt")
    frames = parse_mot_file(tmp_path / "split" / "annotations" / "s1.txt")
    rows = [TrackRow(f, b.track_id, b.x, b.y, b.x + b.w, b.y + b.h, 0.9, group_of(b.category - 1))
            for f in sorted(frames) for b in frames[f] if not b.is_ignore]
    mod.write_rows(tmp_path / "trk", "gt_as_pred", "s1", rows, ignores["s1"])
    for group in EVAL_GROUPS:
        r = run_trackeval(tmp_path / "gt", tmp_path / "trk", group, ["gt_as_pred"])["gt_as_pred"]
        assert r["MOTA"] == pytest.approx(100.0), group
        assert r["IDF1"] == pytest.approx(100.0), group
        assert r["HOTA"] == pytest.approx(100.0), group
        assert r["IDSW"] == 0


def test_run_trackeval_returns_per_sequence_results(tmp_path):
    pytest.importorskip("trackeval")
    from src.eval.trackeval_runner import run_trackeval

    box = lambda f, i: f"{10 + 50 * i + f},20.00,20.00,40.00"
    gt1 = [f"{f},{i},{box(f, i)},1,1,1" for f in range(1, 6) for i in (1, 2)]
    gt2 = [f"{f},1,{box(f, 1)},1,1,1" for f in range(1, 6)]
    write_gt(tmp_path / "gt", "T", "s1", gt1, 5)
    write_gt(tmp_path / "gt", "T", "s2", gt2, 5)
    write_seqmap(tmp_path / "gt", "T", ["s1", "s2"])
    write_tracker(tmp_path / "trk", "T", "t", "s1", [f"{f},{i},{box(f, i)},0.9,-1,-1,-1" for f in range(1, 6) for i in (1, 2)])
    write_tracker(tmp_path / "trk", "T", "t", "s2", [f"{f},1,{box(f, 1)},0.9,-1,-1,-1" for f in range(1, 5)])   # last frame missed
    r = run_trackeval(tmp_path / "gt", tmp_path / "trk", "T", ["t"])["t"]
    assert set(r["per_seq"]) == {"s1", "s2"}
    assert r["per_seq"]["s1"]["MOTA"] == pytest.approx(100.0)
    assert r["per_seq"]["s2"]["MOTA"] == pytest.approx(80.0)          # 1 FN of 5 GT
    assert set(r["per_seq"]["s2"]) == {"HOTA", "MOTA", "IDF1", "IDSW"}
    assert r["MOTA"] == pytest.approx(100 * (1 - 1 / 15))             # combined pools counts: 1 FN of 15 GT
