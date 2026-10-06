import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.eval.det_metrics import greedy_tp, keep_outside_ignore
from tests.helpers import load_script


def test_greedy_tp_one_gt_matched_once():
    gt = np.array([[0, 0, 10, 10]], float)
    preds = np.array([[0, 0, 10, 10], [1, 1, 10, 10]], float)
    assert greedy_tp(preds, np.array([0.9, 0.8]), gt) == 1


def test_greedy_tp_respects_iou_threshold():
    gt = np.array([[0, 0, 10, 10]], float)
    assert greedy_tp(np.array([[8, 8, 18, 18]], float), np.array([0.9]), gt) == 0
    assert greedy_tp(np.zeros((0, 4)), np.zeros(0), gt) == 0


def test_keep_outside_ignore():
    preds = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
    ignore = np.array([[0, 0, 20, 20]], float)
    assert keep_outside_ignore(preds, ignore).tolist() == [False, True]
    assert keep_outside_ignore(preds, np.zeros((0, 4))).tolist() == [True, True]


# --- scripts/04_eval_det.py orchestration, driven by fake models -----------------------

def _ann(path, rows):
    path.write_text("\n".join(",".join(str(v) for v in r) for r in rows) + "\n")


@pytest.fixture
def det_val(tmp_path):
    root = tmp_path / "det"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    for stem in ("a", "b"):
        (root / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8")  # never decoded by the fakes
    # x,y,w,h,score,category,truncation,occlusion
    _ann(root / "annotations" / "a.txt", [
        (0, 0, 10, 10, 1, 1, 0, 0),         # pedestrian GT
        (100, 100, 10, 10, 1, 2, 0, 0),     # people GT
        (200, 200, 10, 10, 1, 4, 0, 0),     # car: not a person GT
        (300, 300, 100, 100, 0, 0, 0, 0),   # ignored region (category 0)
        (500, 500, 20, 20, 1, 11, 0, 0),    # "others": also ignored
    ])
    _ann(root / "annotations" / "b.txt", [(50, 50, 20, 20, 1, 1, 0, 0)])
    return root


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    def __len__(self):
        return len(self.conf)


class FakeModel:
    """Duck-typed Ultralytics model: .predict(path, ...) and .val(...)."""

    ckpt_path = None

    def __init__(self, per_image, as_torch=False):
        self.per_image, self.as_torch, self.calls = per_image, as_torch, []

    def predict(self, path, imgsz, conf, max_det, verbose):
        from pathlib import Path
        self.calls.append((Path(path).name, imgsz, conf, max_det, verbose))
        rows = np.array(self.per_image.get(Path(path).stem, []), dtype=np.float32).reshape(-1, 6)
        conv = torch.from_numpy if self.as_torch else np.asarray
        boxes = _Boxes(conv(rows[:, :4]), conv(rows[:, 4]), conv(rows[:, 5]))
        return [SimpleNamespace(boxes=boxes)]

    def val(self, **kwargs):
        self.val_kwargs = kwargs
        return SimpleNamespace(
            names={0: "pedestrian", 1: "people", 3: "car"},
            box=SimpleNamespace(map50=0.5, map=0.3, ap_class_index=np.array([0, 3]),
                                ap50=np.array([0.6, 0.8])))


PREDS = {
    "a": [
        [0, 0, 10, 10, 0.9, 0],          # TP for the pedestrian
        [100, 100, 110, 110, 0.8, 1],    # TP for the people GT (class 1)
        [200, 200, 210, 210, 0.95, 3],   # car prediction: not a person class
        [310, 310, 330, 330, 0.7, 0],    # inside the ignored region: dropped
        [700, 700, 710, 710, 0.6, 0],    # false positive
        [505, 505, 515, 515, 0.6, 0],    # inside the "others" region: dropped
    ],
    # image b: no predictions at all
}


@pytest.mark.parametrize("as_torch", [False, True])
def test_person_scores_counts_tp_gt_pred(det_val, as_torch):
    mod = load_script("04_eval_det.py")
    model = FakeModel(PREDS, as_torch=as_torch)
    s = mod.person_scores(model, det_val, imgsz=640, conf=0.25, person_classes={0, 1})
    # GT: ped + people (a) + ped (b) = 3, car/ignore/others excluded; preds kept: 3 (2 TP + 1 FP)
    assert (s["tp"], s["gt"], s["pred"]) == (2, 3, 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["precision"] == pytest.approx(2 / 3)
    assert sorted(c[0] for c in model.calls) == ["a.jpg", "b.jpg"]
    assert all(c[1:] == (640, 0.25, 1000, False) for c in model.calls)


def test_person_scores_restricts_to_person_classes(det_val):
    mod = load_script("04_eval_det.py")
    s = mod.person_scores(FakeModel(PREDS), det_val, 960, 0.25, {0})
    # class-1 prediction no longer counts: only the pedestrian TP and the FP remain
    assert (s["tp"], s["gt"], s["pred"]) == (1, 3, 2)
    assert s["recall"] == pytest.approx(1 / 3)
    assert s["precision"] == pytest.approx(1 / 2)


def test_person_scores_handles_no_ground_truth(tmp_path):
    mod = load_script("04_eval_det.py")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images").mkdir()
    s = mod.person_scores(FakeModel({}), tmp_path, 960, 0.25, {0})
    assert s == {"recall": 0.0, "precision": 0.0, "tp": 0, "gt": 0, "pred": 0}


def _result(det_val, with_old):
    mod = load_script("04_eval_det.py")
    new, old = FakeModel(PREDS), FakeModel(PREDS)
    result = mod.build_result(
        new, old if with_old else None, weights="w.pt", imgsz=1280, data="d.yaml", det_val=det_val,
        conf=0.25, old_onnx="old.onnx", old_imgsz=960, label="v2 YOLO26s-P2")
    return mod, new, result


def test_build_result_json_structure(det_val):
    mod, new, result = _result(det_val, with_old=True)
    assert set(result) == {"new", "old"}
    n = result["new"]
    assert n["weights"] == "w.pt" and n["imgsz"] == 1280
    assert n["mAP50"] == 0.5 and n["mAP50_95"] == 0.3
    assert n["per_class_AP50"] == {"pedestrian": 0.6, "car": 0.8}
    assert n["person"]["tp"] == 2 and n["person"]["gt"] == 3
    assert result["old"]["weights"] == "old.onnx" and result["old"]["imgsz"] == 960
    assert result["old"]["person"]["tp"] == 1  # old model scored on class {0} only
    assert new.val_kwargs["data"] == "d.yaml" and new.val_kwargs["imgsz"] == 1280
    json.dumps(result)  # must be plain-JSON serialisable (no numpy scalars)


def test_table_has_both_rows_when_old_exists(det_val):
    mod, _, result = _result(det_val, with_old=True)
    table = mod.render_table(result)
    lines = table.splitlines()
    assert lines[0].startswith("| Model |")
    assert any(l.startswith("| v1 YOLOv8n-P2 (person only) | 33.3% | 50.0% | n/a | n/a |") for l in lines)
    assert any(l.startswith("| v2 YOLO26s-P2 @1280 | 66.7% | 66.7% | 0.500 | 0.300 |") for l in lines)
    assert "| pedestrian | 0.600 |" in lines and "| car | 0.800 |" in lines


def test_table_only_new_row_without_old(det_val):
    mod, _, result = _result(det_val, with_old=False)
    assert "old" not in result
    table = mod.render_table(result)
    assert "v1" not in table
    assert table.count("| v2 YOLO26s-P2 @1280 |") == 1


def test_write_outputs(det_val, tmp_path):
    mod, _, result = _result(det_val, with_old=True)
    out = tmp_path / "out" / "det"
    mod.write_outputs(result, out)
    assert json.loads((out / "det_metrics.json").read_text())["new"]["mAP50"] == 0.5
    assert (out / "det_table.md").read_text() == mod.render_table(result) + "\n"
