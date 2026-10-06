#!/usr/bin/env python3
"""scripts/dev/make_tracker_golden.py — deterministic tracker scenario -> tests/fixtures/tracker_golden.json.

The JS port (web/js/tracker.js) must reproduce these IDs and boxes exactly
(web/tests/tracker.test.mjs). Re-run after any tracker or tracker.yaml change.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.tracker.byte_tracker import ByteTracker, TrackerConfig  # noqa: E402


def scenario(n_frames: int = 50) -> list[list[list[float]]]:
    rng = np.random.default_rng(7)
    objects = [  # (x0, y0, vx, vy, w, h, cls)
        (100, 100, 3.0, 0.5, 20, 40, 0), (400, 120, -2.0, 1.0, 18, 36, 0), (200, 300, 1.5, -1.0, 22, 44, 0),
        (50, 400, 6.0, 0.0, 60, 36, 3), (600, 380, -5.0, 0.3, 64, 40, 3), (300, 500, 2.5, -2.0, 24, 30, 9),
    ]
    frames = []
    for f in range(n_frames):
        dets = []
        for k, (x0, y0, vx, vy, w, h, c) in enumerate(objects):
            if k == 2 and 25 <= f < 32:
                continue                                         # occlusion
            cx, cy = x0 + vx * f, y0 + vy * f
            score = 0.25 if (k == 3 and 15 <= f < 23) else 0.85 + 0.1 * rng.random()   # low-score run
            cls = (1 if f % 3 == 2 else 0) if k == 1 else c                           # class flicker
            jitter = rng.normal(0.0, 0.5, 4)
            dets.append([cx - w / 2 + jitter[0], cy - h / 2 + jitter[1],
                         cx + w / 2 + jitter[2], cy + h / 2 + jitter[3], round(score, 4), cls])
        if f == 18:
            dets.append([700.0, 50.0, 720.0, 90.0, 0.9, 0])     # one-frame false positive
        frames.append(dets)
    return frames


def main() -> None:
    cfg = TrackerConfig.from_yaml(ROOT / "configs" / "tracker.yaml")
    tracker = ByteTracker(cfg)
    out = {"config": asdict(cfg), "frames": []}
    for dets in scenario():
        tracks = tracker.update(np.array(dets, dtype=float))
        out["frames"].append({
            "dets": dets,
            "tracks": sorted([[t.track_id, *map(float, t.tlbr), t.cls] for t in tracks]),
        })
    out["unique_counts"] = tracker.unique_counts()
    path = ROOT / "tests" / "fixtures" / "tracker_golden.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({len(out['frames'])} frames, ids={tracker.total_ids})")


if __name__ == "__main__":
    main()
