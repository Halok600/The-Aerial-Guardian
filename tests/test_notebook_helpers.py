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
