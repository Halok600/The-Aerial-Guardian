#!/usr/bin/env python3
"""scripts/dev/make_tracker_golden.py — deterministic tracker scenarios -> tests/fixtures/*.json.

Two fixtures, both replayed frame by frame by the JS port (web/js/tracker.js):
  * tracker_golden.json  one 50-frame scenario (web/tests/tracker.test.mjs)
  * tracker_parity.json  compact scenarios built to exercise every tracker branch the first one
                         misses: tentative confirm/removal, the new_track skip, lost-track expiry
                         at the frame-rate-scaled boundary, duplicate removal, rotating camera
                         warps, class-vote window trimming (web/tests/parity.test.mjs)

Regenerate after ANY change to the Python tracker or configs/tracker.yaml; the pytest
tests/test_tracker_golden.py fails with a "stale" message until you do.
The parity generator counts tracker events while it runs and raises if a branch never fired,
so the scenarios cannot silently go stale.
"""
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.tracker.byte_tracker as bt  # noqa: E402
from src.tracker.byte_tracker import ByteTracker, TrackerConfig  # noqa: E402

GOLDEN_PATH = ROOT / "tests" / "fixtures" / "tracker_golden.json"
PARITY_PATH = ROOT / "tests" / "fixtures" / "tracker_parity.json"
CONFIG_PATH = ROOT / "configs" / "tracker.yaml"

# Events every scenario set must fire at least once (checked at generation time).
REQUIRED_EVENTS = ("tent_confirm", "rm_tentative", "rm_expiry", "new_track_skip", "dup_removed",
                   "refound", "stage2", "warp_rot", "warp_none", "vote_trim")


# --------------------------------------------------------------------------- original scenario
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


def build_golden() -> dict:
    cfg = TrackerConfig.from_yaml(CONFIG_PATH)
    tracker = ByteTracker(cfg)
    out = {"config": asdict(cfg), "frames": []}
    for dets in scenario():
        tracks = tracker.update(np.array(dets, dtype=float))
        out["frames"].append({
            "dets": dets,
            "tracks": sorted([[t.track_id, *map(float, t.tlbr), t.cls] for t in tracks]),
        })
    out["unique_counts"] = tracker.unique_counts()
    out["total_ids"] = tracker.total_ids
    return out


# --------------------------------------------------------------------------- event counting
class Probe:
    """Counts tracker events by wrapping a few Python-tracker hooks; restores them on exit."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.assignments: list = []
        self.track_high = 0.4

    def __enter__(self) -> "Probe":
        self._orig = {"update": bt.STrack.update, "removed": bt.STrack.mark_removed,
                      "dups": bt._remove_duplicates, "la": bt.linear_assignment}
        probe, o = self, self._orig

        def update(track, kf, det, frame_id):
            if track.state == bt.TrackState.LOST:
                probe.counts["refound"] += 1
            elif not track.is_activated:
                probe.counts["tent_confirm"] += 1
            elif det.score < probe.track_high:
                probe.counts["stage2"] += 1
            if len(track._votes) == track._votes.maxlen:
                probe.counts["vote_trim"] += 1
            o["update"](track, kf, det, frame_id)

        def mark_removed(track):
            probe.counts["rm_tentative" if not track.is_activated else "rm_expiry"] += 1
            o["removed"](track)

        def dups(tracked, lost):
            a, b = o["dups"](tracked, lost)
            probe.counts["dup_removed"] += (len(tracked) - len(a)) + (len(lost) - len(b))
            return a, b

        def linear_assignment(cost, thresh):
            res = o["la"](cost, thresh)
            probe.assignments.append(res)
            return res

        bt.STrack.update, bt.STrack.mark_removed = update, mark_removed
        bt._remove_duplicates, bt.linear_assignment = dups, linear_assignment
        return self

    def __exit__(self, *exc) -> None:
        bt.STrack.update, bt.STrack.mark_removed = self._orig["update"], self._orig["removed"]
        bt._remove_duplicates, bt.linear_assignment = self._orig["dups"], self._orig["la"]

    def count_new_track_skips(self, dets, cfg: TrackerConfig) -> None:
        """High detections left over after stage 1 and 3 whose score is below new_track."""
        stage1, stage3 = self.assignments[-3], self.assignments[-1]
        high = [d for d in dets if d[2] > d[0] and d[3] > d[1] and d[4] >= cfg.track_low and d[4] >= cfg.track_high]
        leftover = [high[i] for i in stage1[2]]
        self.counts["new_track_skip"] += sum(1 for i in stage3[2] if leftover[i][4] < cfg.new_track)


# --------------------------------------------------------------------------- parity scenarios
def _r6(v: float) -> float:
    return round(float(v), 6)


def make_obj(x, y, vx, vy, w, h, cls=0, spans=((0, 10 ** 9),), low=(), jitter=0.5):
    """Moving box. spans: visible frame ranges [a, b); low: (a, b, lo, hi) score-range overrides;
    cls: an int, or a per-frame sequence/callable."""
    return dict(x=x, y=y, vx=vx, vy=vy, w=w, h=h, cls=cls, spans=spans, low=low, jitter=jitter)


def rot_warp(angle: float, tx: float, ty: float):
    c, s = np.cos(angle), np.sin(angle)
    return [[round(float(c), 9), round(float(-s), 9), _r6(tx)], [round(float(s), 9), round(float(c), 9), _r6(ty)]]


def render(objs, n_frames, rng, warps=None, extras=None):
    """Detections per frame. Object centres follow the tracker's own order of operations
    (advance by velocity, then apply the camera warp, which also rotates the velocity), so
    tracks survive the camera motion."""
    pos = [[o["x"], o["y"]] for o in objs]
    vel = [[o["vx"], o["vy"]] for o in objs]
    frames = []
    for f in range(n_frames):
        w = warps[f] if warps else None
        for p, v in zip(pos, vel):
            if f > 0:
                p[0] += v[0]
                p[1] += v[1]
            if w is not None:
                p[0], p[1] = w[0][0] * p[0] + w[0][1] * p[1] + w[0][2], w[1][0] * p[0] + w[1][1] * p[1] + w[1][2]
                v[0], v[1] = w[0][0] * v[0] + w[0][1] * v[1], w[1][0] * v[0] + w[1][1] * v[1]
        dets = []
        for o, p in zip(objs, pos):
            if not any(a <= f < b for a, b in o["spans"]):
                continue
            score = rng.uniform(0.75, 0.98)
            for a, b, lo, hi in o["low"]:
                if a <= f < b:
                    score = rng.uniform(lo, hi)
            cls = o["cls"][f] if isinstance(o["cls"], (list, tuple)) else o["cls"]
            j = rng.normal(0.0, o["jitter"], 4)
            dets.append([_r6(p[0] - o["w"] / 2 + j[0]), _r6(p[1] - o["h"] / 2 + j[1]),
                         _r6(p[0] + o["w"] / 2 + j[2]), _r6(p[1] + o["h"] / 2 + j[3]), round(float(score), 4), int(cls)])
        for ef, det in (extras or []):
            if ef == f:
                dets.append(det)
        frames.append(dets)
    return frames


def births_scenario():
    rng = np.random.default_rng(101)
    objs = [
        make_obj(100, 300, 2.0, 0.0, 60, 36, 3),                                    # steady car
        make_obj(300, 100, 1.0, 1.0, 20, 44, 0, spans=((10, 60),), low=((10, 13, 0.41, 0.49),)),
        make_obj(450, 150, -1.5, 0.5, 20, 44, 0, spans=((20, 23),)),                # 3 matches: counted, then lost+expired
        make_obj(500, 400, 0.5, -1.0, 22, 44, 1, spans=((30, 32),)),                # 2 matches: confirmed, not counted
        make_obj(650, 80, 0.0, 0.0, 30, 30, 0, spans=((40, 41),)),                  # one-frame false positive
        make_obj(80, 450, 1.0, 0.0, 30, 22, 9, spans=((2, 16),), low=((2, 16, 0.41, 0.49),)),   # never reaches new_track
        make_obj(250, 450, 2.0, -0.5, 64, 40, 4, spans=((25, 60),), low=((35, 42, 0.2, 0.35),)),  # late birth, stage-2 run
    ]
    return render(objs, 60, rng), None, {}, 30.0, ("tent_confirm", "rm_tentative", "new_track_skip", "stage2", "rm_expiry")


def expiry_scenario(seed, n_frames, gaps, overrides, frame_rate):
    rng = np.random.default_rng(seed)
    objs = [make_obj(80 + 120 * k, 100 + 40 * k, 0.4, 0.2, 24, 48, 0 if k % 2 == 0 else 3,
                     spans=((0, 20), (20 + g, n_frames)) if g else ((0, n_frames),))
            for k, g in enumerate(gaps)]
    return render(objs, n_frames, rng), None, overrides, frame_rate, ("rm_expiry", "refound")


def warp_scenario():
    rng = np.random.default_rng(303)
    objs = [make_obj(100 + 90 * k, 120 + 50 * k, 1.5 - 0.7 * k, 0.5, 26 + 4 * k, 50, (0, 3, 2, 6, 9)[k],
                     spans=((0, 24), (30, 60)) if k == 2 else ((0, 60),)) for k in range(5)]
    warps = [None if f % 7 == 3 else rot_warp(0.04 * np.sin(f / 3.0) + 0.01, 3.0 * np.cos(f / 4.0), -2.0 * np.sin(f / 5.0))
             for f in range(60)]
    return render(objs, 60, rng, warps), warps, {}, 30.0, ("warp_rot", "warp_none", "refound")


def vote_scenario():
    rng = np.random.default_rng(404)
    flicker = [int(c) for c in rng.choice([3, 4, 8], 100)]
    objs = [
        make_obj(100, 100, 1.0, 0.5, 20, 44, [0] * 45 + [1] * 55),                  # class change: majority flips only if the window trims
        make_obj(300, 200, -1.0, 0.5, 60, 38, flicker),
        make_obj(500, 300, 0.5, -0.5, 30, 26, [9] * 50 + [2] * 50),
    ]
    return render(objs, 100, rng), None, {}, 30.0, ("vote_trim",)


def threshold_scenario():
    """Association-threshold edges: a tentative track whose second match has fused cost in
    (unconfirmed_match_thresh, match_thresh), and a low-score box whose plain cost lies in
    (second_match_thresh, match_thresh). Both must be rejected."""
    rng = np.random.default_rng(606)
    objs = [
        make_obj(200, 200, 0, 0, 40, 60, 0, spans=((10, 11),), jitter=0.3),                  # tentative ...
        make_obj(220.7, 200, 0.3, 0, 40, 60, 0, spans=((11, 45),), jitter=0.3),              # ... jumps 24 px at frame 11: IoU ~0.25
        make_obj(420, 300, 0, 0, 40, 60, 3, spans=((0, 20), (23, 45)), jitter=0.3),          # tracked car ...
        make_obj(435, 300, 0, 0, 40, 60, 3, spans=((20, 23),), low=((20, 23, 0.25, 0.35),), jitter=0.3),  # ... low-score box, IoU ~0.45
    ]
    return render(objs, 45, rng), None, {}, 30.0, ("tent_confirm", "rm_tentative", "refound")


def cross_group_scenario():
    """Near-identical boxes of different class groups: only duplicate removal's group check
    keeps a lost person from deleting the car born on top of it (and vice versa)."""
    rng = np.random.default_rng(707)
    objs = [
        make_obj(150, 150, 0.1, 0, 40, 60, 0, spans=((0, 15),), jitter=0.3),                 # person vanishes ...
        make_obj(150, 150, 0.1, 0, 40, 60, 3, spans=((20, 50),), jitter=0.3),                # ... car appears on top of it
        make_obj(450, 300, 0, 0, 40, 60, 3, spans=((0, 50),), jitter=0.3),                   # old car ...
        make_obj(450, 300, 0, 0, 40, 60, 0, spans=((10, 14),), jitter=0.3),                  # ... young person briefly on top of it
    ]
    return render(objs, 50, rng), None, {}, 30.0, ("tent_confirm", "rm_expiry")


def random_scene(seed, n_frames, n_objs, ang, overrides, frame_rate, n_clones=2):
    """Crowd of random objects, some with near-identical same-group clones born later
    (tracked-vs-lost duplicates), gaps, low-score runs, junk detections and a camera warp."""
    rng = np.random.default_rng(seed)
    groups = [[0, 1], [3, 4, 5, 8], [2, 9], [6, 7]]
    objs = []
    for _ in range(n_objs):
        g = groups[int(rng.integers(4))]
        w = float(rng.uniform(18, 60))
        birth = int(rng.integers(0, 15))
        gap = int(rng.integers(3, 14))
        s = birth + int(rng.integers(8, 25))
        objs.append(make_obj(float(rng.uniform(80, 560)), float(rng.uniform(80, 380)), float(rng.normal(0, 2)),
                             float(rng.normal(0, 1.5)), w, w * float(rng.uniform(0.7, 1.8)), g[int(rng.integers(len(g)))],
                             spans=((birth, s), (s + gap, n_frames)),
                             low=((s - 6, s, 0.15, 0.4),) if rng.random() < 0.5 else (), jitter=0.8))
    for k in range(n_clones):
        o = dict(objs[k])
        o.update(x=o["x"] + float(rng.normal(0, 2)), y=o["y"] + float(rng.normal(0, 2)),
                 spans=((int(rng.integers(8, 20)), n_frames),))
        objs.append(o)
    extras = [(int(rng.integers(n_frames)), [round(float(v), 6) for v in (10, 10, 10, 40)] + [0.9, 0]) for _ in range(2)]
    warps = [None if rng.random() < 0.25 else rot_warp(float(rng.normal(0, ang)), float(rng.normal(0, 2)), float(rng.normal(0, 2)))
             for _ in range(n_frames)]
    return render(objs, n_frames, rng, warps, extras), warps, overrides, frame_rate, ()


def scenarios():
    """name -> (frames, warps, config overrides, frame_rate, events this scenario must fire)."""
    return {
        "births": births_scenario(),
        "expiry_fps25": expiry_scenario(202, 75, (0, 11, 12, 13, 14), {"max_time_lost": 15}, 25.0),
        "expiry_fps30": expiry_scenario(203, 100, (0, 29, 30, 31), {}, 30.0),
        "warp_rotation": warp_scenario(),
        "vote_trim": vote_scenario(),
        "thresholds": threshold_scenario(),
        "cross_group_overlap": cross_group_scenario(),
        "crowd_a": random_scene(505, 70, 8, 0.04, {}, 30.0),
        "crowd_b": random_scene(506, 70, 8, 0.04, {"max_time_lost": 15}, 25.0, n_clones=3),
        "crowd_c": random_scene(507, 70, 6, 0.01, {"match_thresh": 0.7, "new_track": 0.45}, 30.0, n_clones=3),
    }


def run_scenario(name, frames, warps, overrides, frame_rate, base_cfg, probe):
    cfg = TrackerConfig(**{**asdict(base_cfg), **overrides})
    tracker = ByteTracker(cfg, frame_rate)
    probe.track_high = cfg.track_high
    probe.counts.clear()
    rec = {"name": name, "frame_rate": frame_rate, "overrides": overrides, "config": asdict(cfg), "frames": []}
    for i, dets in enumerate(frames):
        warp = warps[i] if warps else None
        probe.counts["warp_rot" if warp and abs(warp[1][0]) > 1e-3 else "warp_none" if warp is None else "warp_flat"] += 1
        probe.assignments.clear()
        tracks = tracker.update(np.array(dets, dtype=float).reshape(-1, 6), None if warp is None else np.array(warp))
        probe.count_new_track_skips(dets, cfg)
        rec["frames"].append({
            "dets": dets, "warp": warp,
            "tracks": sorted([[t.track_id, *map(float, t.tlbr), t.cls] for t in tracks]),
            "n_tracked": len(tracker.tracked), "n_lost": len(tracker.lost),
        })
    rec["unique_counts"] = tracker.unique_counts()
    rec["total_ids"] = tracker.total_ids
    return rec, Counter(probe.counts)


def build_parity() -> tuple[dict, Counter]:
    base_cfg = TrackerConfig.from_yaml(CONFIG_PATH)
    out, total = {"scenarios": []}, Counter()
    with Probe() as probe:
        for name, (frames, warps, overrides, frame_rate, needs) in scenarios().items():
            rec, counts = run_scenario(name, frames, warps, overrides, frame_rate, base_cfg, probe)
            missing = [e for e in needs if not counts[e]]
            if missing:
                raise RuntimeError(f"scenario {name!r} no longer fires {missing}; counts={dict(counts)}")
            out["scenarios"].append(rec)
            total.update(counts)
    missing = [e for e in REQUIRED_EVENTS if not total[e]]
    if missing:
        raise RuntimeError(f"parity scenarios never fire {missing}; counts={dict(total)}")
    return out, total


# --------------------------------------------------------------------------- output
def dumps_fixture(obj: dict, frames_per_line: bool = True) -> str:
    """JSON with one frame per line: compact, but diffs stay readable."""
    def compact(v):
        return json.dumps(v, separators=(",", ":"))

    def scenario_text(sc):
        head = {k: v for k, v in sc.items() if k != "frames"}
        frames = ",\n".join("  " + compact(fr) for fr in sc["frames"])
        return compact(head)[:-1] + ',"frames":[\n' + frames + "\n]}"
    if "scenarios" in obj:
        return '{"scenarios":[\n' + ",\n".join(scenario_text(sc) for sc in obj["scenarios"]) + "\n]}\n"
    return json.dumps(obj, indent=1) + "\n"


def main() -> None:
    golden = build_golden()
    GOLDEN_PATH.write_text(dumps_fixture(golden), encoding="utf-8")
    print(f"wrote {GOLDEN_PATH} ({len(golden['frames'])} frames, ids={golden['total_ids']})")
    parity, counts = build_parity()
    PARITY_PATH.write_text(dumps_fixture(parity), encoding="utf-8")
    n_frames = sum(len(s["frames"]) for s in parity["scenarios"])
    print(f"wrote {PARITY_PATH} ({len(parity['scenarios'])} scenarios, {n_frames} frames, "
          f"{PARITY_PATH.stat().st_size // 1024} KiB)")
    print("events:", ", ".join(f"{k}={counts[k]}" for k in sorted(counts)))


if __name__ == "__main__":
    main()
