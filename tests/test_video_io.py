import threading
import time

import cv2
import numpy as np
import pytest

from src.utils.video_io import AsyncWriter, FramePrefetcher


def _folder(tmp_path, n=3, w=64, h=48):
    for i in range(n):
        cv2.imwrite(str(tmp_path / f"{i + 1:04d}.jpg"), np.full((h, w, 3), (10 * i) % 256, np.uint8))
    return tmp_path


def _consume_in_thread(src, timeout=5.0):
    """Drain ``src`` on a worker thread so a hang fails the test instead of the suite."""
    box = {"frames": [], "error": None}

    def work():
        try:
            for item in src:
                box["frames"].append(item)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "consumer hung"
    return box


def test_prefetcher_yields_all_frames(tmp_path):
    with FramePrefetcher(_folder(tmp_path, 5)) as src:
        box = _consume_in_thread(src)
    assert [i for i, _ in box["frames"]] == [1, 2, 3, 4, 5] and box["error"] is None
    assert (src.width, src.height, src.total_frames) == (64, 48, 5)


def test_reader_exception_is_raised_in_consumer(tmp_path, monkeypatch):
    folder = _folder(tmp_path, 5)
    real, calls = cv2.imread, {"n": 0}

    def flaky(path, *a):
        calls["n"] += 1
        if calls["n"] == 4:  # 1 constructor sample + frames 1..3
            raise RuntimeError("disk exploded")
        return real(path, *a)

    monkeypatch.setattr(cv2, "imread", flaky)
    with FramePrefetcher(folder) as src:
        box = _consume_in_thread(src)
    assert [i for i, _ in box["frames"]] == [1, 2]
    assert isinstance(box["error"], RuntimeError) and "disk exploded" in str(box["error"])


def test_unreadable_image_names_the_file(tmp_path):
    folder = _folder(tmp_path, 3)
    (folder / "0002.jpg").write_bytes(b"not a jpeg")
    with FramePrefetcher(folder) as src:
        box = _consume_in_thread(src)
    assert box["error"] is not None and "0002.jpg" in str(box["error"])


def test_unreadable_first_image_raises_in_constructor(tmp_path):
    (tmp_path / "0001.jpg").write_bytes(b"garbage")
    with pytest.raises((ValueError, OSError), match="0001.jpg"):
        FramePrefetcher(tmp_path)


def test_close_stops_reader_thread_and_frees_capture(tmp_path):
    folder = _folder(tmp_path, 40)
    srcs = []
    for _ in range(4):
        s = FramePrefetcher(folder, buffer_size=2)
        next(iter(s))
        s.close()
        srcs.append(s)
    deadline = time.time() + 2
    while time.time() < deadline and any(s._thread.is_alive() for s in srcs):
        time.sleep(0.05)
    assert not any(s._thread.is_alive() for s in srcs)


def test_close_releases_video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 5, np.uint8))
    w.release()
    s = FramePrefetcher(path, buffer_size=2)
    next(iter(s))
    s.close()
    assert not s._thread.is_alive()
    path.unlink()  # fails on Windows if the capture is still open


def test_writer_roundtrip(tmp_path):
    out = tmp_path / "o.mp4"
    w = AsyncWriter(out, 10, (64, 48))
    for i in range(7):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    w.release()
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (64, 48)
    cap.release()


def test_writer_unopenable_path_raises(tmp_path):
    with pytest.raises(RuntimeError, match="nodir"):
        AsyncWriter(tmp_path / "nodir" / "o.mp4", 10, (64, 48))


def test_writer_rejects_mismatched_frame_size(tmp_path):
    w = AsyncWriter(tmp_path / "o.mp4", 10, (64, 48))
    try:
        with pytest.raises(ValueError):
            w.write(np.zeros((48, 65, 3), np.uint8))
    finally:
        w.release()
