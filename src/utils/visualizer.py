"""src/utils/visualizer.py — Box, label, trail and HUD renderer for tracked objects.

Style: class-group coloured boxes with "<class> ID:<n>" chips, fading per-ID
trails, and a HUD with per-group UNIQUE counts (objects seen so far, not just
on screen now).

Performance: all trail segments are drawn onto ONE overlay and added to the
frame once, so blending cost is O(1) per frame regardless of track count or
trail length (the v2 optimisation is kept).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.classes import GROUP_COLORS_BGR, GROUP_LABELS, GROUP_ORDER, class_name

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scale_params(height: int, width: int) -> tuple[int, float, int, float, int]:
    """Sizing for boxes, label chips and HUD from ONE scale factor.

    s = max(1, min(h, w) / 720): never smaller than the 720p look (so 480p
    clips stay legible), growing with resolution so the output survives
    downscaling (README GIF). Using min(h, w) keeps the HUD inside portrait
    frames. Returns (box_thickness, label_fs, label_thickness, hud_fs, hud_thickness).
    """
    s = max(1.0, min(height, width) / 720.0)
    return (max(2, round(2 * s)), 0.6 * s, max(1, round(s)), 0.7 * s, max(1, round(s)))


def id_to_bgr(track_id: int) -> tuple[int, int, int]:
    """Golden-angle hue per ID so neighbouring trails stay distinguishable."""
    hue = int((track_id * 137.508) % 180)
    bgr = cv2.cvtColor(np.array([[[hue, 220, 230]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def label_for(track) -> str:
    return f"{class_name(track.cls)} ID:{track.track_id}"


def hud_lines(counts: dict, fps: float, frame_id: int, n_active: int) -> list[str]:
    unique = "  ".join(f"{GROUP_LABELS[g]}: {counts.get(g, 0)}" for g in GROUP_ORDER)
    return [f"Unique  {unique}", f"Frame: {frame_id}  |  Active: {n_active}  |  FPS: {fps:.1f}"]


class TrackVisualizer:
    def __init__(self, trail_len: int = 40):
        self.trail_len = trail_len

    def draw(self, frame: np.ndarray, tracks, counts: dict, fps: float, frame_id: int) -> np.ndarray:
        out = frame.copy()
        box_t, label_fs, label_t, hud_fs, hud_t = _scale_params(*out.shape[:2])

        if self.trail_len > 1:
            overlay = np.zeros_like(out)
            for t in tracks:
                pts = [(int(x), int(y)) for x, y in list(t.history)[-self.trail_len:]]
                if len(pts) < 2:
                    continue
                b, g, r = id_to_bgr(t.track_id)
                n = len(pts) - 1
                for i in range(1, len(pts)):
                    a = i / n                                   # old -> dim, new -> bright
                    cv2.line(overlay, pts[i - 1], pts[i], (int(b * a), int(g * a), int(r * a)),
                             max(1, int(round(3 * a))), cv2.LINE_AA)
            out = cv2.add(out, overlay)

        for t in tracks:
            x1, y1, x2, y2 = (int(round(v)) for v in t.tlbr)
            color = GROUP_COLORS_BGR[t.group]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, box_t)    # non-AA: crisp, full-coverage edges
            label = label_for(t)
            (lw, lh), base = cv2.getTextSize(label, FONT, label_fs, label_t)
            ly = max(0, y1 - lh - base - 2)
            cv2.rectangle(out, (x1, ly), (x1 + lw + 4, ly + lh + base + 2), color, -1)
            cv2.putText(out, label, (x1 + 2, ly + lh + 1), FONT, label_fs, (255, 255, 255), label_t,
                        cv2.LINE_AA)

        self._draw_hud(out, hud_lines(counts, fps, frame_id, len(tracks)), hud_fs, hud_t)
        return out

    @staticmethod
    def _draw_hud(img: np.ndarray, lines: list[str], fs: float, thickness: int) -> None:
        pad = 8
        sizes = [cv2.getTextSize(line, FONT, fs, thickness)[0] for line in lines]
        line_h = max(s[1] for s in sizes) + pad
        w = min(img.shape[1], max(s[0] for s in sizes) + 2 * pad)
        h = min(img.shape[0], line_h * len(lines) + pad)
        img[:h, :w] = (img[:h, :w].astype(np.float32) * 0.35).astype(np.uint8)
        for i, line in enumerate(lines):
            color = (80, 255, 80) if i == 0 else (235, 235, 235)
            cv2.putText(img, line, (pad, pad + line_h * i + sizes[i][1]), FONT, fs, color,
                        thickness, cv2.LINE_AA)
