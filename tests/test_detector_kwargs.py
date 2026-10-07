import sys
import types
from types import SimpleNamespace

import numpy as np

from src.pipeline import Detector, DetectorSettings


def test_detector_passes_quantize_not_deprecated_half(monkeypatch):
    captured = []

    class FakeYOLO:
        def __init__(self, *_a, **_k):
            pass

        def predict(self, _frame, **kwargs):
            captured.append(kwargs)
            return [SimpleNamespace(boxes=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    frame = np.zeros((8, 8, 3), np.uint8)
    for half, expect in ((True, 16), (False, None)):
        Detector("w.pt", DetectorSettings(half=half, device="cpu"))(frame)
        kwargs = captured[-1]
        assert "half" not in kwargs
        assert kwargs.get("quantize") == expect
        assert (expect is None) == ("quantize" not in kwargs)
