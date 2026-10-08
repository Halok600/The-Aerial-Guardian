"""src/tracker/ecc_compensator.py — OpenCV ECC camera-motion compensation (v2).

Estimates the drone's ego-motion between consecutive frames and returns a 2x3
warp (previous-frame pixels -> current-frame pixels) that ByteTrack applies to
its Kalman predictions before association.

v2 fixes:
  * Downscale preserves aspect ratio (fixed width, derived height). The old
    fixed 320x180 squashed non-16:9 frames, which breaks a Euclidean model.
  * Warp rescale uses the exact similarity transform S * M * S^-1 instead of
    scaling only the translation column.
  * Coarse-to-fine pyramid. A single-scale ECC on the 320px image only
    converges when the motion is within about one texture wavelength; on
    fine-grained texture (aerial grass, gravel) it diverges or locks onto a
    wrong optimum. Solving at 1/4 scale first and refining keeps it stable.
"""
from __future__ import annotations

import cv2
import numpy as np


class ECCCompensator:
    CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)

    def __init__(self, downscale_width: int = 320, motion: int = cv2.MOTION_EUCLIDEAN,
                 levels: int = 3):
        self.dw = downscale_width
        self.motion = motion
        self.levels = max(1, levels)
        self.prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None

    @staticmethod
    def rescale_warp(warp_small: np.ndarray, sx: float, sy: float) -> np.ndarray:
        s = np.diag([sx, sy, 1.0])
        m = np.vstack([np.asarray(warp_small, dtype=float), [0.0, 0.0, 1.0]])
        return (s @ m @ np.linalg.inv(s))[:2].astype(np.float32)

    def update(self, frame: np.ndarray) -> np.ndarray:
        fh, fw = frame.shape[:2]
        dh = max(1, int(round(fh * self.dw / fw)))
        small = cv2.resize(frame, (self.dw, dh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

        warp = np.eye(2, 3, dtype=np.float32)
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            warp_small = self._estimate(self.prev_gray, gray)
            if warp_small is not None:
                warp = self.rescale_warp(warp_small, fw / self.dw, fh / dh)
        self.prev_gray = gray
        return warp

    def _estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray | None:
        """Coarse-to-fine ECC; returns the prev->curr warp at ``prev``'s scale.

        cv2.findTransformECC(template, input, ...) returns the warp that maps
        template coords to input coords, so passing (prev, curr) yields
        prev -> curr directly. A failure at a finer level keeps the coarser
        estimate; a failure at the coarsest level (flat sky, heavy blur)
        returns None, i.e. "no camera motion".
        """
        pyr_prev, pyr_curr = [prev], [curr]
        for _ in range(self.levels - 1):
            if min(pyr_prev[-1].shape[:2]) < 32:
                break
            pyr_prev.append(cv2.pyrDown(pyr_prev[-1]))
            pyr_curr.append(cv2.pyrDown(pyr_curr[-1]))

        warp: np.ndarray | None = None
        for lvl in range(len(pyr_prev) - 1, -1, -1):
            init = np.eye(2, 3, dtype=np.float32) if warp is None else warp
            try:
                _, init = cv2.findTransformECC(
                    pyr_prev[lvl], pyr_curr[lvl], init.copy(), self.motion, self.CRITERIA, None, 5)
            except cv2.error:
                if warp is None:
                    return None
                # keep the coarser estimate, brought up to this level's scale
                init = warp
            warp = init if lvl == 0 else self.rescale_warp(init, 2.0, 2.0)
        return warp
