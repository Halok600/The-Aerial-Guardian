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
    [1, 6, 8400],      # raw with nc=2
    [1, 6, 300],       # non-end2end: 6 channels but anchors last, never rows of 6
    [1, 14, 76500],
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
    assert meta["names"][0] == "pedestrian" and meta["names"][-1] == "motor"
    assert meta["names"][3] == "car" and meta["names"][8] == "bus"


def test_build_meta_rejects_bad_format():
    with pytest.raises(ValueError):
        export_web.build_meta(960, "nms", "w.pt", "8.4.174")


def _e2e(rows):
    return np.array([rows], np.float32)


def test_compare_outputs_end2end_ignores_order_among_tied_scores():
    a = _e2e([[0, 0, 10, 10, 0.5, 1], [20, 20, 30, 30, 0.5, 1], [5, 5, 9, 9, 0.9, 2]])
    b = _e2e([[20, 20, 30, 30, 0.5, 1], [5, 5, 9, 9, 0.9, 2], [0, 0, 10, 10, 0.5, 1]])
    assert export_web.compare_outputs(a, b, "end2end") == (0.0, 0.0, 1.0, 0, 0.0)


def test_compare_outputs_end2end_flags_missing_and_shifted_boxes():
    a = _e2e([[0, 0, 10, 10, 0.9, 1], [40, 40, 50, 50, 0.7, 3], [100, 100, 110, 110, 0.8, 2], [1, 1, 2, 2, 0.1, 0]])
    b = _e2e([[3, 0, 10, 10, 0.9, 1], [100, 100, 110, 110, 0.8, 2], [0, 0, 1, 1, 0.1, 0], [2, 2, 3, 3, 0.1, 0]])
    box, score, frac, n_bad, worst = export_web.compare_outputs(a, b, "end2end")
    assert n_bad == 3 and worst == float("inf")  # the 0.9 row is 3px off (twice); class 3 has no counterpart
    assert box == pytest.approx(0.0) and score == pytest.approx(0.0)
    # a->b: the 0.9 box is 3px off and the class-3 row is missing, the 0.8 row matches;
    # b->a: the 0.9 row fails again, the 0.8 row matches; the 0.1 rows sit at the cut-off and are exempt
    assert frac == pytest.approx(2 / 5)


def test_compare_outputs_raw_is_elementwise():
    a = np.zeros((1, 14, 6), np.float32)
    b = a.copy()
    b[0, 2, 4] += 0.25   # a box coordinate (px)
    b[0, 9, 1] += 0.01   # a class score
    box, score, frac, n_bad, worst = export_web.compare_outputs(a, b, "raw")
    assert box == pytest.approx(0.25) and score == pytest.approx(0.01) and frac == 1.0 and n_bad == 0


def test_compare_outputs_end2end_cannot_pass_vacuously():
    """Equal scores everywhere => every row is exempt; garbage boxes must not pass."""
    a = np.zeros((1, 300, 6), np.float32)
    a[0, :, 4] = 0.5
    b = a.copy()
    b[0, :, :4] = 999.0
    assert export_web.compare_outputs(a, b, "end2end").matched_frac == 0.0


@pytest.mark.parametrize("simplify, available, expected", [
    (True, True, "ok"), (True, False, "fail"), (False, False, "ok"), (False, True, "ok"),
])
def test_simplify_status(simplify, available, expected):
    assert export_web.simplify_status(simplify, available) == expected


def test_layout_problems_clean():
    assert export_web.layout_problems("end2end", "end2end", "True") == []
    assert export_web.layout_problems("raw", "raw", "False") == []


def test_layout_problems_requested_vs_detected():
    probs = export_web.layout_problems("end2end", "raw", "False")
    assert len(probs) == 1 and "requested end2end" in probs[0]


def test_layout_problems_metadata_disagrees_with_shape():
    assert export_web.layout_problems("end2end", "end2end", "False")   # shape says end2end, metadata says not
    assert export_web.layout_problems("raw", "raw", "True")
    assert export_web.layout_problems("raw", "raw", "true") != []      # case-insensitive, still disagrees


def test_layout_problems_missing_metadata_is_a_problem():
    assert export_web.layout_problems("end2end", "end2end", None)


def test_out_dir_guard(tmp_path):
    web_model = export_web.DEFAULT_OUT_DIR
    assert export_web.out_dir_problem("models/weights/yolo26n-p2-smoke.pt", web_model, False)
    assert export_web.out_dir_problem("models/weights/Dev-run.PT", web_model, False)
    assert export_web.out_dir_problem("models/weights/yolo26n-p2-smoke.pt", web_model / ".." / "model", False)
    assert export_web.out_dir_problem("models/weights/yolo26n-p2-smoke.pt", web_model, True) is None  # --force
    assert export_web.out_dir_problem("models/weights/yolo26n-p2-smoke.pt", tmp_path, False) is None
    assert export_web.out_dir_problem("models/weights/yolo26n-p2-visdrone.pt", web_model, False) is None
