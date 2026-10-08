import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("notebook_helpers", ROOT / "kaggle" / "notebook_helpers.py")
nh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nh)


def _mot(base: Path) -> Path:
    (base / "annotations").mkdir(parents=True)
    (base / "sequences").mkdir()
    (base / "annotations" / "s.txt").write_text("x")
    return base


def test_configured_path_exists(tmp_path):
    good = _mot(tmp_path / "input" / "ds" / "VisDrone2019-MOT-train")
    assert nh.resolve_mot_dir(str(good), tmp_path / "input") == str(good)


def test_configured_none_is_explicit_det_only(tmp_path):
    assert nh.resolve_mot_dir(None, tmp_path) is None


def test_configured_missing_but_found_elsewhere(tmp_path, capsys):
    found = _mot(tmp_path / "input" / "other" / "deep" / "VisDrone2019-MOT-train")
    _mot(tmp_path / "input" / "other" / "VisDrone2019-MOT-val")  # wrong name, must not match
    got = nh.resolve_mot_dir(str(tmp_path / "input" / "nope" / "VisDrone2019-MOT-train"), tmp_path / "input")
    assert got == str(found)
    assert str(found) in capsys.readouterr().out


def test_configured_dir_without_annotations_and_sequences_is_not_valid(tmp_path):
    bad = tmp_path / "input" / "ds" / "VisDrone2019-MOT-train"
    bad.mkdir(parents=True)
    good = _mot(tmp_path / "input" / "zz" / "VisDrone2019-MOT-train")
    assert nh.resolve_mot_dir(str(bad), tmp_path / "input") == str(good)


def test_nothing_found_raises_with_listing(tmp_path):
    (tmp_path / "input" / "aerial-guardian-code" / "src").mkdir(parents=True)
    (tmp_path / "input" / "weird-ds" / "stuff").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as e:
        nh.resolve_mot_dir("/kaggle/input/x/VisDrone2019-MOT-train", tmp_path / "input")
    msg = str(e.value)
    assert "aerial-guardian-code/src" in msg and "weird-ds/stuff" in msg
    assert "--mot-train-dir" in msg and "--no-mot" in msg


def test_nothing_found_missing_input_root_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        nh.resolve_mot_dir("/nope/VisDrone2019-MOT-train", tmp_path / "absent")


def test_compute_time_budget():
    assert nh.compute_time_budget(9.5, 0.2) == 9.5                    # plenty of room: requested wins
    assert nh.compute_time_budget(9.5, 1.0) == pytest.approx(9.0)     # 11 - 1.0 - 1.0 = 9.0 < 9.5
    assert nh.compute_time_budget(9.5, 0.5, session_limit_h=11.0, reserve_h=1.0) == pytest.approx(9.5)
    with pytest.raises(RuntimeError, match="pointless"):
        nh.compute_time_budget(9.5, 9.6)
    with pytest.raises(RuntimeError):
        nh.compute_time_budget(9.5, 9.5)  # budget exactly 0.5 is not > 0.5


def test_push_code_stages_helpers_at_dataset_root():
    src = (ROOT / "kaggle" / "push_code.py").read_text(encoding="utf-8")
    assert 'ROOT / "kaggle" / "notebook_helpers.py", STAGE)' in src


def _zip(path: Path, members: dict[str, bytes]) -> None:
    import zipfile
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_unpack_code_zips_extracts_prefixless_zip_into_its_folder(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    _zip(code / "src.zip", {"pipeline.py": b"p", "utils/classes.py": b"c"})
    _zip(code / "scripts.zip", {"01_build_dataset.py": b"b"})
    (code / "notebook_helpers.py").write_text("h")
    assert nh.unpack_code_zips(code) == ["scripts.zip", "src.zip"]
    assert (code / "src" / "pipeline.py").read_bytes() == b"p"
    assert (code / "src" / "utils" / "classes.py").read_bytes() == b"c"
    assert (code / "scripts" / "01_build_dataset.py").read_bytes() == b"b"
    assert not (code / "01_build_dataset.py").exists() and not (code / "pipeline.py").exists()
    assert not list(code.glob("*.zip")) and (code / "notebook_helpers.py").exists()


def test_unpack_code_zips_leaves_existing_folder_alone(tmp_path):
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    (code / "src" / "pipeline.py").write_text("mounted")
    _zip(code / "src.zip", {"pipeline.py": b"from-zip"})
    assert nh.unpack_code_zips(code) == []
    assert (code / "src" / "pipeline.py").read_text() == "mounted"
    assert not (code / "src.zip").exists()


def test_unpack_code_zips_without_zips_is_noop(tmp_path):
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    assert nh.unpack_code_zips(code) == []
    assert sorted(p.name for p in code.iterdir()) == ["src"]


def test_list_top_levels_depth_three_and_truncation(tmp_path):
    (tmp_path / "a" / "b" / "c" / "d").mkdir(parents=True)
    for i in range(30):
        (tmp_path / "many" / f"d{i:02d}").mkdir(parents=True)
    out = nh.list_top_levels(tmp_path, depth=3, max_entries=5)
    assert "a/b/c" in out and "a/b/c/d" not in out
    assert any("more directories" in line for line in out)


def test_resolve_mot_dir_error_lists_three_levels(tmp_path):
    (tmp_path / "input" / "datasets" / "owner" / "slug").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="datasets/owner/slug"):
        nh.resolve_mot_dir("/x/VisDrone2019-MOT-train", tmp_path / "input")


def test_resolve_mot_dir_finds_dataset_under_datasets_owner_layout(tmp_path):
    found = _mot(tmp_path / "input" / "datasets" / "akhileswarelluru" / "visdrone2019-mot"
                 / "VisDrone_Zips" / "VisDrone2019-MOT-train" / "VisDrone2019-MOT-train")
    assert nh.resolve_mot_dir("/kaggle/input/visdrone2019-mot/nope", tmp_path / "input") == str(found)
