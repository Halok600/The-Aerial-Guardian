import re
from pathlib import Path

import numpy as np
import pytest

from src.tracker.byte_tracker import TrackerConfig
from tests.helpers import load_script


def _make_split(root, n_frames=6):
    seq = root / "sequences" / "s1"
    seq.mkdir(parents=True)
    for i in range(1, n_frames + 1):
        (seq / f"{i:07d}.jpg").write_bytes(b"")
    (root / "annotations").mkdir()
    lines = [f"{f},1,10,10,20,40,1,1,0,0" for f in range(1, n_frames + 1)]
    lines += [f"{f},2,100,50,60,30,1,4,0,0" for f in range(1, n_frames + 1)]
    lines.append("1,-1,500,0,100,100,0,0,0,0")
    (root / "annotations" / "s1.txt").write_text("\n".join(lines) + "\n")


def test_prepare_gt_truncates_gt_and_seqlength(tmp_path):
    mod = load_script("05_eval_mot.py")
    _make_split(tmp_path / "split")
    ignores = mod.prepare_gt(tmp_path / "split", ["s1"], tmp_path / "gt", max_frames=3)
    person = tmp_path / "gt" / "person-test" / "s1"
    assert "seqLength=3" in (person / "seqinfo.ini").read_text()
    frames = {int(l.split(",")[0]) for l in (person / "gt" / "gt.txt").read_text().splitlines()}
    assert frames == {1, 2, 3}
    assert max(ignores["s1"]) == 3 and len(ignores["s1"][1]) == 1


def test_prepare_gt_without_limit_uses_all_frames(tmp_path):
    mod = load_script("05_eval_mot.py")
    _make_split(tmp_path / "split")
    mod.prepare_gt(tmp_path / "split", ["s1"], tmp_path / "gt")
    assert "seqLength=6" in (tmp_path / "gt" / "vehicle-test" / "s1" / "seqinfo.ini").read_text()


# ---- argument resolution / safety guards ---------------------------------------------------

def test_tune_refuses_test_splits(capsys):
    mod = load_script("05_eval_mot.py")
    for split in ("data/visdrone/VisDrone2019-MOT-test-dev", "data/X/MOT-TEST-challenge/"):
        with pytest.raises(SystemExit) as e:
            mod.parse_args(["--tune", "--split", split])
        assert e.value.code != 0
    assert "test" in capsys.readouterr().err.lower()
    # report mode on a test split and tune mode on val are fine
    assert mod.parse_args(["--split", "data/visdrone/VisDrone2019-MOT-test-dev"]).tune is False
    assert mod.parse_args(["--tune", "--split", "data/visdrone/VisDrone2019-MOT-val"]).tune is True


def test_default_out_depends_on_mode():
    mod = load_script("05_eval_mot.py")
    assert mod.parse_args([]).out == Path("outputs/eval/mot")
    assert mod.parse_args(["--tune", "--split", "x/val"]).out == Path("outputs/eval/mot_tune")
    assert mod.parse_args(["--out", "foo"]).out == Path("foo")


def test_safe_rmtree_only_inside_outputs(tmp_path):
    mod = load_script("05_eval_mot.py")
    outputs = tmp_path / "outputs"
    inside = outputs / "eval" / "x" / "gt"
    inside.mkdir(parents=True)
    (inside / "f.txt").write_text("1")
    outside = tmp_path / "data"
    outside.mkdir()
    (outside / "keep.txt").write_text("1")
    for bad in (outside, tmp_path, outputs, tmp_path / "outputs" / ".." / "data"):
        with pytest.raises(ValueError):
            mod.safe_rmtree(bad, outputs_root=outputs)
    assert (outside / "keep.txt").exists()
    mod.safe_rmtree(inside, outputs_root=outputs)
    assert not inside.exists()
    mod.safe_rmtree(outputs / "does-not-exist", outputs_root=outputs)       # missing dir is a no-op
    mod.safe_rmtree(outside, outputs_root=outputs, force=True)              # explicit --force-out
    assert not outside.exists()


def test_safe_rmtree_resolves_symlinks(tmp_path):
    mod = load_script("05_eval_mot.py")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    target = tmp_path / "data"
    target.mkdir()
    (target / "keep.txt").write_text("1")
    link = outputs / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        mod.safe_rmtree(link, outputs_root=outputs)
    assert (target / "keep.txt").exists()


def test_check_out_dir_accepts_default_outputs_only():
    mod = load_script("05_eval_mot.py")
    for out in (mod.parse_args([]).out, mod.parse_args(["--tune", "--split", "x/val"]).out):
        mod.check_out_dir(out)          # relative paths resolve against the repo root
    for bad in (Path("."), Path("data"), Path("..")):
        with pytest.raises(ValueError):
            mod.check_out_dir(bad)
    mod.check_out_dir(Path("."), force=True)


# ---- variants, grid, Ultralytics config ------------------------------------------------------

def test_grid_is_108_unique_filesystem_safe_names():
    mod = load_script("05_eval_mot.py")
    args = mod.parse_args(["--tune", "--split", "x/val"])
    variants = mod.build_variants(args, TrackerConfig())
    assert len(variants) == 108
    assert len(set(variants)) == 108
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+", n) for n in variants)
    assert all(use_cmc is True for _, use_cmc in variants.values())            # CMC stays ON for tuning
    assert {cfg.match_thresh for cfg, _ in variants.values()} == {0.7, 0.8, 0.9}
    assert {cfg.fuse_score for cfg, _ in variants.values()} == {True, False}
    small = mod.build_variants(mod.parse_args(["--tune", "--grid-small", "--split", "x/val"]), TrackerConfig())
    assert 1 < len(small) < 10


def test_report_variants_are_explicit_about_cmc():
    mod = load_script("05_eval_mot.py")
    variants = mod.build_variants(mod.parse_args([]), TrackerConfig())
    assert {n: use for n, (_, use) in variants.items()} == {"ours_cmc": True, "ours_nocmc": False}


def test_track_variants_uses_mapping_not_the_name(tmp_path, monkeypatch):
    mod = load_script("05_eval_mot.py")
    seen = []

    def fake_track_cached(dets, warps, tracker, use_cmc=True):
        seen.append(use_cmc)
        return []

    monkeypatch.setattr(mod, "track_cached", fake_track_cached)
    caches = {"s1": (np.zeros((0, 7)), np.zeros((0, 2, 3)))}
    variants = {"ours_nocmc_typo": (TrackerConfig(), True), "whatever": (TrackerConfig(), False)}
    mod.track_variants(variants, caches, {"s1": {}}, tmp_path)
    assert seen == [True, False]


def test_ultralytics_tracker_config_key_mapping():
    mod = load_script("05_eval_mot.py")
    cfg = TrackerConfig(track_high=0.35, track_low=0.12, new_track=0.55, match_thresh=0.9,
                        max_time_lost=60, fuse_score=False)
    assert mod.ultralytics_tracker_config(cfg) == {
        "tracker_type": "bytetrack", "track_high_thresh": 0.35, "track_low_thresh": 0.12,
        "new_track_thresh": 0.55, "track_buffer": 60, "match_thresh": 0.9, "fuse_score": False}


def test_ultralytics_tracker_yaml_roundtrip(tmp_path):
    import yaml

    mod = load_script("05_eval_mot.py")
    path = mod.write_ultralytics_yaml(TrackerConfig(), tmp_path / "bt.yaml")
    assert yaml.safe_load(Path(path).read_text()) == mod.ultralytics_tracker_config(TrackerConfig())


def test_ultralytics_config_keys_match_installed_default_yaml():
    ultra = pytest.importorskip("ultralytics")
    import yaml

    mod = load_script("05_eval_mot.py")
    default_yaml = Path(ultra.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    default = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))
    assert set(mod.ultralytics_tracker_config(TrackerConfig())) == set(default)


def test_table_has_both_baseline_rows_in_order():
    mod = load_script("05_eval_mot.py")
    keys = ("ours_cmc", "ours_nocmc", "ultralytics_ours", "ultralytics_default")
    res = {"person": {k: {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0, "IDSW": 4} for k in keys}}
    rows = [l for l in mod.table(res, mod.TABLE_NAMES).splitlines() if l.startswith("| ") and "Tracker" not in l]
    assert [r.split("|")[1].strip() for r in rows] == [
        "Ultralytics ByteTrack (default config)", "Ultralytics ByteTrack (our thresholds)",
        "Ours (no camera-motion comp.)", "Ours + ECC camera-motion comp."]


def test_per_seq_table():
    mod = load_script("05_eval_mot.py")

    def m(h):
        return {"HOTA": h, "MOTA": 0.0, "IDF1": 0.0, "IDSW": 0}

    res = {"person": {"ours_cmc": {**m(50.0), "per_seq": {"a": m(40.0), "b": m(60.0)}},
                      "ultralytics_default": {**m(30.0), "per_seq": {"a": m(20.0), "b": m(41.5)}}}}
    md = mod.per_seq_table(res, mod.TABLE_NAMES)
    assert "| a | 20.0 | 40.0 |" in md and "| b | 41.5 | 60.0 |" in md
    assert "Ultralytics ByteTrack (default config) | Ours + ECC" in md


# ---- detection cache -------------------------------------------------------------------------

def _write_images(seq_dir, n=3):
    import cv2

    seq_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        cv2.imwrite(str(seq_dir / f"{i:07d}.jpg"), rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))


class _CountingDetector:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = 0, fail_on

    def __call__(self, frame):
        self.calls += 1
        if self.fail_on == self.calls:
            raise RuntimeError("boom")
        return np.array([[1, 2, 30, 40, 0.9, 0]], np.float32)


def test_load_cache_roundtrip_and_reuse(tmp_path):
    mod = load_script("05_eval_mot.py")
    seq = tmp_path / "sequences" / "s1"
    _write_images(seq)
    det = _CountingDetector()
    cache_dir = tmp_path / "cache"
    dets, warps = mod.load_cache(seq, lambda: det, cache_dir)
    assert dets.shape == (3, 7) and warps.shape == (3, 2, 3) and det.calls == 3
    assert [p.name for p in cache_dir.iterdir()] == ["s1.npz"]                  # no leftover temp files
    again, _ = mod.load_cache(seq, lambda: pytest.fail("detector must not be built"), cache_dir)
    assert np.array_equal(again, dets)


def test_load_cache_max_frames(tmp_path):
    mod = load_script("05_eval_mot.py")
    seq = tmp_path / "s"
    _write_images(seq, 4)
    dets, warps = mod.load_cache(seq, lambda: _CountingDetector(), tmp_path / "c", max_frames=2)
    assert warps.shape[0] == 2 and set(dets[:, 0]) == {1.0, 2.0}


def test_corrupt_cache_is_ignored_and_recomputed(tmp_path):
    mod = load_script("05_eval_mot.py")
    seq = tmp_path / "s1"
    _write_images(seq)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "s1.npz").write_bytes(b"not an npz")
    det = _CountingDetector()
    dets, _ = mod.load_cache(seq, lambda: det, cache_dir)
    assert det.calls == 3 and dets.shape == (3, 7)
    assert np.load(cache_dir / "s1.npz")["dets"].shape == (3, 7)                  # replaced by a valid file


def test_interrupted_cache_run_leaves_no_cache_file(tmp_path):
    mod = load_script("05_eval_mot.py")
    seq = tmp_path / "s1"
    _write_images(seq)
    cache_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError):
        mod.load_cache(seq, lambda: _CountingDetector(fail_on=2), cache_dir)
    assert not (cache_dir.exists() and any(cache_dir.iterdir()))


def test_cache_dir_key_covers_detector_settings_and_version(tmp_path, monkeypatch):
    from src.pipeline import DetectorSettings

    mod = load_script("05_eval_mot.py")
    w = tmp_path / "w.pt"
    w.write_bytes(b"abc")

    def key(max_frames=60, **kw):
        s = {"imgsz": 960, "conf": 0.05, "iou": 0.6, "max_det": 1000, **kw}
        return mod.cache_dir_for(w, DetectorSettings(**s), Path("data/val"), max_frames)

    base = key()
    for other in (key(conf=0.01), key(iou=0.7), key(max_det=300), key(imgsz=1280), key(max_frames=None)):
        assert other != base
    monkeypatch.setattr(mod, "CACHE_VERSION", mod.CACHE_VERSION + 1)
    assert key() != base
