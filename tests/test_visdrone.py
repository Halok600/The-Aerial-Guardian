from pathlib import Path

import numpy as np

from src.data.visdrone import (
    MASK_VALUE, PERSON_ONLY_CLASS_MAP, Box, mask_ignored, parse_det_file,
    parse_mot_file, to_yolo_line,
)

FIX = Path(__file__).parent / "fixtures" / "visdrone"


def test_parse_det_marks_ignore_rows():
    boxes = parse_det_file(FIX / "det_sample.txt")
    assert len(boxes) == 8
    assert [b.is_ignore for b in boxes] == [True, True, True, False, False, False, False, True]
    assert boxes[4] == Box(50, 60, 30, 15, category=4, score=1)


def test_yolo_lines_for_det_sample():
    boxes = parse_det_file(FIX / "det_sample.txt")
    lines = [line for b in boxes if (line := to_yolo_line(b, 1000, 600)) is not None]
    assert lines == [
        "0 0.006000 0.018333 0.010000 0.033333",  # pedestrian
        "3 0.065000 0.112500 0.030000 0.025000",  # car
        "1 0.995000 0.866667 0.010000 0.066667",  # people, clipped at right edge
    ]


def test_person_only_map_drops_vehicles():
    boxes = parse_det_file(FIX / "det_sample.txt")
    lines = [line for b in boxes if (line := to_yolo_line(b, 1000, 600, PERSON_ONLY_CLASS_MAP))]
    assert [line.split()[0] for line in lines] == ["0", "0"]


def test_parse_mot_groups_by_frame():
    frames = parse_mot_file(FIX / "mot_sample.txt")
    assert sorted(frames) == [1, 2]
    assert len(frames[1]) == 3 and len(frames[2]) == 2
    assert frames[1][0].track_id == 1 and frames[1][0].category == 1
    assert frames[1][2].is_ignore


def test_mask_ignored_paints_only_ignore_boxes_and_copies():
    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    boxes = parse_det_file(FIX / "det_sample.txt")
    out = mask_ignored(img, boxes)
    assert img.max() == 0, "input must not be mutated"
    assert (out[8:124, 684:957] == MASK_VALUE).all()
    assert (out[300:320, 300:320] == MASK_VALUE).all()   # category 11 "others"
    assert (out[1:21, 1:11] == 0).all()                  # pedestrian untouched
