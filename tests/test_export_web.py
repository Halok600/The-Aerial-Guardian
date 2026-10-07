"""Tests for the pure helpers in scripts/06_export_web.py (no weights, no export)."""
import numpy as np
import pytest

from src.utils.classes import VISDRONE_NAMES
from tests.helpers import load_script

export_web = load_script("06_export_web.py")


@pytest.mark.parametrize("shape", [
    [1, 300, 6], [1, 100, 6], (1, 300, 6), ["batch", 300, 6], [None, 300, 6], ["N", 300, 6],
])
def test_detect_output_format_end2end(shape):
    assert export_web.detect_output_format(shape) == "end2end"


@pytest.mark.parametrize("shape", [
    [1, 14, 18900], [1, 5, 8400], ["batch", 14, "anchors"], [None, 14, 18900], [1, 84, 8400],
])
def test_detect_output_format_raw(shape):
    assert export_web.detect_output_format(shape) == "raw"


def test_detect_output_format_rejects_unknown_rank():
    with pytest.raises(ValueError):
        export_web.detect_output_format([1, 300])


def test_build_meta_has_contract_fields():
    meta = export_web.build_meta(960, "end2end", "models/weights/yolo26n-p2-visdrone.pt", "8.4.174")
    assert meta == {
        "input_size": 960,
        "output_format": "end2end",
        "names": list(VISDRONE_NAMES),
        "source_weights": "yolo26n-p2-visdrone.pt",
        "exported_with": "ultralytics 8.4.174",
    }
    assert len(meta["names"]) == 10


def test_build_meta_rejects_bad_format():
    with pytest.raises(ValueError):
        export_web.build_meta(960, "nms", "w.pt", "8.4.174")


def _e2e(rows):
    return np.array([rows], np.float32)


def test_compare_outputs_end2end_ignores_order_among_tied_scores():
    a = _e2e([[0, 0, 10, 10, 0.5, 1], [20, 20, 30, 30, 0.5, 1], [5, 5, 9, 9, 0.9, 2]])
    b = _e2e([[20, 20, 30, 30, 0.5, 1], [5, 5, 9, 9, 0.9, 2], [0, 0, 10, 10, 0.5, 1]])
    assert export_web.compare_outputs(a, b, "end2end") == (0.0, 0.0, 1.0)


def test_compare_outputs_end2end_flags_missing_and_shifted_boxes():
    a = _e2e([[0, 0, 10, 10, 0.9, 1], [40, 40, 50, 50, 0.7, 3], [100, 100, 110, 110, 0.8, 2], [1, 1, 2, 2, 0.1, 0]])
    b = _e2e([[3, 0, 10, 10, 0.9, 1], [100, 100, 110, 110, 0.8, 2], [0, 0, 1, 1, 0.1, 0], [2, 2, 3, 3, 0.1, 0]])
    box, score, frac = export_web.compare_outputs(a, b, "end2end")
    assert box == pytest.approx(0.0) and score == pytest.approx(0.0)
    # a->b: the 0.9 box is 3px off and the class-3 row is missing, the 0.8 row matches;
    # b->a: the 0.9 row fails again, the 0.8 row matches; the 0.1 rows sit at the cut-off and are exempt
    assert frac == pytest.approx(2 / 5)


def test_compare_outputs_raw_is_elementwise():
    a = np.zeros((1, 14, 6), np.float32)
    b = a.copy()
    b[0, 2, 4] += 0.25   # a box coordinate (px)
    b[0, 9, 1] += 0.01   # a class score
    box, score, frac = export_web.compare_outputs(a, b, "raw")
    assert box == pytest.approx(0.25) and score == pytest.approx(0.01) and frac == 1.0
