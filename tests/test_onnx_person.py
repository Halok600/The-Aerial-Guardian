from pathlib import Path

import cv2
import numpy as np
import pytest

from src.eval.onnx_person import OnnxPersonModel, decode_person_output, letterbox


def test_letterbox_wide_image_is_padded_top_and_bottom():
    img = np.full((100, 200, 3), 255, np.uint8)  # H=100, W=200
    chw, scale, pad_x, pad_y = letterbox(img, 960)
    assert chw.shape == (1, 3, 960, 960) and chw.dtype == np.float32
    assert scale == pytest.approx(4.8) and pad_x == 0 and pad_y == 240
    assert chw[0, 0, 0, 0] == pytest.approx(114 / 255)      # padding
    assert chw[0, 0, 480, 480] == pytest.approx(1.0)        # image content
    assert chw[0, 0, 239, 480] == pytest.approx(114 / 255)  # last padded row above the image


def test_letterbox_is_rgb_chw():
    bgr = np.zeros((10, 10, 3), np.uint8)
    bgr[..., 2] = 255  # pure red in BGR
    chw, *_ = letterbox(bgr, 20)
    assert chw[0, 0, 10, 10] == pytest.approx(1.0) and chw[0, 2, 10, 10] == pytest.approx(0.0)


def _output(rows):
    """rows of (cx, cy, w, h, score) -> [1, 5, N] like the old single-class export."""
    return np.array(rows, np.float32).T[None]


def test_decode_thresholds_nms_and_unletterboxes():
    # 200x100 source letterboxed to 960: scale 4.8, pad (0, 240)
    out = _output([
        [480, 480, 96, 96, 0.9],    # keeper -> source box (90..110, 40..60)
        [484, 480, 96, 96, 0.8],    # near-duplicate: suppressed by NMS
        [100, 300, 20, 20, 0.1],    # below the confidence threshold
        [800, 700, 48, 48, 0.6],    # separate object
    ])
    dets = decode_person_output(out, conf=0.25, iou=0.45, scale=4.8, pad_x=0, pad_y=240)
    assert dets.shape == (2, 6)
    assert dets[:, 4].tolist() == pytest.approx([0.9, 0.6])
    assert dets[:, 5].tolist() == [0, 0]
    # (480-48, 480-48, 480+48, 480+48) -> ((x-0)/4.8, (y-240)/4.8)
    assert dets[0, :4] == pytest.approx([90.0, 40.0, 110.0, 60.0])
    assert dets[1, :4] == pytest.approx([(800 - 24) / 4.8, (700 - 24 - 240) / 4.8,
                                         (800 + 24) / 4.8, (700 + 24 - 240) / 4.8])


def test_decode_empty():
    out = _output([[10, 10, 5, 5, 0.01]])
    assert decode_person_output(out, 0.25, 0.45, 1.0, 0, 0).shape == (0, 6)


class _StubSession:
    def __init__(self, output):
        self.output, self.feeds = output, None

    def get_inputs(self):
        return [type("I", (), {"name": "images", "shape": [1, 3, 960, 960]})()]

    def run(self, _names, feeds):
        self.feeds = feeds
        return [self.output]


def test_predict_adapter_matches_ultralytics_result_contract(tmp_path: Path):
    img_path = tmp_path / "x.jpg"
    cv2.imwrite(str(img_path), np.full((100, 200, 3), 200, np.uint8))
    sess = _StubSession(_output([[480, 480, 96, 96, 0.9]]))
    model = OnnxPersonModel(session=sess)
    res = model.predict(str(img_path), imgsz=960, conf=0.25, max_det=1000, verbose=False)
    assert len(res) == 1 and len(res[0].boxes) == 1
    assert res[0].boxes.xyxy[0].tolist() == pytest.approx([90, 40, 110, 60])
    assert res[0].boxes.conf.tolist() == pytest.approx([0.9]) and res[0].boxes.cls.tolist() == [0]
    assert sess.feeds["images"].shape == (1, 3, 960, 960)
