"""src/eval/onnx_person.py — Run the v1 person-only ONNX the way the old web client did.

The v1 model was exported from Ultralytics 8.0.236 as a single-class, NMS-free
[1, 5, N] head (cx, cy, w, h, score). This decodes it exactly like the old
web/js/inference.js (letterbox to a square with 114 padding, RGB, /255, CHW;
threshold; NMS at IoU 0.45) and exposes an Ultralytics-like `.predict(...)` so
the same evaluation code can score it. It is an independent cross-check of
Ultralytics' own ONNX path.
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torchvision.ops import nms

PAD_VALUE = 114


class _Boxes:
    def __init__(self, dets: np.ndarray):
        self.xyxy, self.conf, self.cls = dets[:, :4], dets[:, 4], dets[:, 5]

    def __len__(self) -> int:
        return len(self.conf)


def letterbox(img_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """BGR image -> (1, 3, size, size) float32 RGB in [0, 1], plus (scale, pad_x, pad_y)."""
    h, w = img_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(
        img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    chw = canvas[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw)[None], scale, pad_x, pad_y


def decode_person_output(output: np.ndarray, conf: float, iou: float,
                         scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    """[1, 5, N] (cx, cy, w, h, score) in letterbox space -> (K, 6) [x1, y1, x2, y2, score, 0] in source pixels."""
    out = np.asarray(output, dtype=np.float32).reshape(5, -1)
    keep = out[4] >= conf
    cx, cy, w, h, score = out[:, keep]
    if len(score) == 0:
        return np.zeros((0, 6), np.float32)
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    kept = nms(torch.from_numpy(xyxy), torch.from_numpy(score), iou).numpy()
    xyxy, score = xyxy[kept], score[kept]
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
    return np.concatenate([xyxy, score[:, None], np.zeros((len(score), 1), np.float32)], axis=1)


class OnnxPersonModel:
    """Ultralytics-compatible `.predict` over an onnxruntime session for the v1 person model."""

    def __init__(self, path: str | None = None, session=None, nms_iou: float = 0.45):
        if session is None:
            import onnxruntime as ort
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.session, self.nms_iou = session, nms_iou
        inp = session.get_inputs()[0]
        self.input_name = inp.name
        side = inp.shape[2] if isinstance(inp.shape[2], int) else None
        self.fixed_size = side  # static exports must be fed exactly this size
        self.ckpt_path = str(path) if path else None

    def predict(self, path: str, imgsz: int, conf: float, max_det: int = 1000, verbose: bool = False):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        chw, scale, pad_x, pad_y = letterbox(img, self.fixed_size or imgsz)
        (out,) = self.session.run(None, {self.input_name: chw})[:1]
        dets = decode_person_output(out, conf, self.nms_iou, scale, pad_x, pad_y)
        dets = dets[np.argsort(-dets[:, 4])][:max_det]
        return [SimpleNamespace(boxes=_Boxes(dets))]
