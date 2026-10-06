import zipfile
from pathlib import Path

from tests.helpers import load_script

dl = load_script("00_download_data.py")

PREFIX = "VisDrone_Zips/VisDrone2019-MOT-val/VisDrone2019-MOT-val"


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extract_prefix_strips_prefix_and_counts(tmp_path):
    zp = _make_zip(tmp_path / "d.zip", {
        f"{PREFIX}/annotations/seq1.txt": b"a",
        f"{PREFIX}/sequences/seq1/0000001.jpg": b"b",
        f"{PREFIX}/sequences/seq1/0000002.jpg": b"c",
        "VisDrone_Zips/VisDrone2019-MOT-train/VisDrone2019-MOT-train/annotations/t.txt": b"x",
        "README.md": b"r",
    })
    dest = tmp_path / "out" / "VisDrone2019-MOT-val"
    n = dl.extract_prefix(zp, PREFIX, dest)
    assert n == 3
    assert (dest / "annotations" / "seq1.txt").read_bytes() == b"a"
    assert (dest / "sequences" / "seq1" / "0000002.jpg").read_bytes() == b"c"
    assert sorted(p.name for p in dest.iterdir()) == ["annotations", "sequences"]
    assert not (tmp_path / "out" / "README.md").exists()


def test_extract_prefix_ignores_members_outside_prefix_and_lookalikes(tmp_path):
    zp = _make_zip(tmp_path / "d.zip", {
        f"{PREFIX}/annotations/a.txt": b"a",
        f"{PREFIX}-extra/annotations/b.txt": b"b",   # shares the string prefix but not the directory
        "other/annotations/c.txt": b"c",
    })
    dest = tmp_path / "dest"
    assert dl.extract_prefix(zp, PREFIX, dest) == 1
    assert [p.name for p in dest.rglob("*") if p.is_file()] == ["a.txt"]


def test_extract_prefix_skips_zip_slip_member(tmp_path):
    zp = _make_zip(tmp_path / "d.zip", {
        f"{PREFIX}/annotations/ok.txt": b"ok",
        f"{PREFIX}/../../../evil.txt": b"evil",
        f"{PREFIX}/annotations/../../../evil2.txt": b"evil",
    })
    dest = tmp_path / "nest" / "dest"
    n = dl.extract_prefix(zp, PREFIX, dest)
    assert n == 1
    assert (dest / "annotations" / "ok.txt").exists()
    assert not list(tmp_path.rglob("evil*"))


def test_extract_prefix_handles_directory_entries(tmp_path):
    zp = tmp_path / "d.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr(f"{PREFIX}/", b"")
        zf.writestr(f"{PREFIX}/annotations/", b"")
        zf.writestr(f"{PREFIX}/annotations/a.txt", b"a")
    dest = tmp_path / "dest"
    assert dl.extract_prefix(zp, PREFIX, dest) == 1
    assert (dest / "annotations" / "a.txt").exists()


def test_extract_prefix_accepts_trailing_slash_prefix(tmp_path):
    zp = _make_zip(tmp_path / "d.zip", {f"{PREFIX}/annotations/a.txt": b"a"})
    assert dl.extract_prefix(zp, PREFIX + "/", tmp_path / "dest") == 1


def test_parse_extract_spec():
    assert dl.parse_extract_spec("a/b=C") == ("a/b", "C")
    for bad in ("nodest", "=x", "x=", "x=../up", "x=/abs"):
        try:
            dl.parse_extract_spec(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad spec {bad!r}")


def test_download_kaggle_zip_uses_plain_cli_list_without_unzip(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check=False, **kw):
        calls.append((cmd, check, kw))
        _make_zip(tmp_path / "_downloads" / "visdrone2019-mot.zip", {"a/b.txt": b"x"})

    monkeypatch.setattr(dl.subprocess, "run", fake_run)
    monkeypatch.setattr(dl, "kaggle_command", lambda: ["/venv/Scripts/kaggle.exe"])
    zp = dl.download_kaggle_zip("owner/visdrone2019-mot", tmp_path)
    cmd, check, kw = calls[0]
    assert cmd == ["/venv/Scripts/kaggle.exe", "datasets", "download", "-d", "owner/visdrone2019-mot",
                   "-p", str(tmp_path / "_downloads")]
    assert check is True and not kw.get("shell")
    assert "--unzip" not in cmd and zp.name == "visdrone2019-mot.zip"
