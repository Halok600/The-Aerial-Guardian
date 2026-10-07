import subprocess

import cv2
import numpy as np
import pytest

from tests.helpers import ROOT, load_script


@pytest.fixture(scope="module")
def mod():
    return load_script("07_make_showcase.py")


def make_sequence(folder, n=6, size=(64, 48), ext=".jpg"):
    folder.mkdir(parents=True, exist_ok=True)
    w, h = size
    for i in range(1, n + 1):
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.rectangle(img, (i * 4, 10), (i * 4 + 12, 30), (0, 200, 255), -1)
        assert cv2.imwrite(str(folder / f"{i:07d}{ext}"), img)
    return folder


def probe(ffmpeg, path):
    """ffmpeg -i prints stream info on stderr and exits 1 with no output file; that is fine here."""
    return subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True).stderr


def test_parser_defaults_and_flags(mod, tmp_path):
    args = mod.build_parser().parse_args(["--sequence", str(tmp_path), "--name", "demo1"])
    assert args.max_frames == 450 and not args.raw and not args.gif
    assert args.demo_dir == mod.DEMO and args.media_dir == mod.MEDIA
    args = mod.build_parser().parse_args(
        ["--sequence", "s", "--name", "x", "--raw", "--gif", "--max-frames", "3", "--demo-dir", "d"])
    assert args.raw and args.gif and args.max_frames == 3 and str(args.demo_dir) == "d"


def test_parser_requires_sequence_and_name(mod):
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--name", "x"])
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--sequence", "s"])


def test_list_frames_sorted_and_mixed_extensions(mod, tmp_path):
    for name in ["0000010.jpg", "0000002.jpg", "0000001.png", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in mod.list_frames(tmp_path)] == ["0000001.png", "0000002.jpg", "0000010.jpg"]


def test_list_frames_errors(mod, tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        mod.list_frames(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="No .jpg"):
        mod.list_frames(tmp_path)


def test_h264_args_are_web_safe(mod, tmp_path):
    args = mod.h264_args(tmp_path / "a.mp4", tmp_path / "b.mp4")
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[args.index("-movflags") + 1] == "+faststart"
    assert "-an" in args


def test_ffmpeg_failure_surfaces_stderr(mod, tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        mod.h264(tmp_path / "does-not-exist.mp4", tmp_path / "out.mp4")


def test_bad_name_rejected(mod, tmp_path):
    seq = make_sequence(tmp_path / "seq", n=1)
    with pytest.raises(SystemExit, match="--name"):
        mod.main(["--sequence", str(seq), "--name", "../evil", "--raw", "--demo-dir", str(tmp_path / "d")])


def test_missing_weights_fails_fast(mod, tmp_path):
    seq = make_sequence(tmp_path / "seq", n=1)
    with pytest.raises(SystemExit, match="Weights not found"):
        mod.main(["--sequence", str(seq), "--name", "t", "--weights", "no/such.pt",
                  "--demo-dir", str(tmp_path / "d")])
    assert not list((tmp_path / "d").glob("*"))


def test_raw_end_to_end_produces_h264_yuv420p(mod, tmp_path):
    seq = make_sequence(tmp_path / "seq", n=6, size=(65, 47))  # odd size must still encode
    demo, media = tmp_path / "demo", tmp_path / "media"
    out = mod.main(["--sequence", str(seq), "--name", "clip", "--raw", "--max-frames", "5", "--gif",
                    "--demo-dir", str(demo), "--media-dir", str(media)])
    assert out == demo / "clip.mp4" and out.stat().st_size > 0
    assert [p.name for p in demo.iterdir()] == ["clip.mp4"]  # temp file removed

    info = probe(mod.FFMPEG, out)
    assert "h264" in info and "yuv420p" in info
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    frames = 0
    while cap.read()[0]:
        frames += 1
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    assert frames == 5 and w % 2 == 0 and h % 2 == 0 and w <= 65

    g = media / "hero.gif"
    assert g.stat().st_size > 0 and g.read_bytes()[:3] == b"GIF"


def test_defaults_point_inside_repo(mod):
    assert mod.DEMO == ROOT / "web" / "assets" / "demo"
    assert mod.MEDIA == ROOT / "docs" / "media"
