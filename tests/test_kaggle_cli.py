import sys

import pytest

from src.utils import kaggle_cli


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Empty PATH, cwd away from the repo (Windows `which` also searches cwd), interpreter in tmp_path/venv."""
    venv = tmp_path / "venv"
    venv.mkdir()
    py = venv / "python.exe"
    py.write_text("")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(sys, "executable", str(py))
    return venv


def test_sibling_of_interpreter_is_found_when_not_on_path(isolated):
    exe = isolated / "kaggle.exe"
    exe.write_text("")
    assert kaggle_cli.kaggle_command() == [str(exe.resolve())]


def test_posix_style_sibling_is_found(isolated):
    exe = isolated / "kaggle"
    exe.write_text("")
    assert kaggle_cli.kaggle_command() == [str(exe.resolve())]


def test_clear_error_when_missing(isolated):
    with pytest.raises(FileNotFoundError) as e:
        kaggle_cli.kaggle_command()
    assert "pip install kaggle" in str(e.value) and "venv" in str(e.value)


def test_path_hit_wins_over_sibling(isolated, monkeypatch):
    (isolated / "kaggle.exe").write_text("")
    monkeypatch.setattr(kaggle_cli.shutil, "which", lambda name: "/somewhere/kaggle")
    assert kaggle_cli.kaggle_command() == ["/somewhere/kaggle"]


def test_push_code_uses_resolved_cli_and_stages_files(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("push_code", root / "kaggle" / "push_code.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    calls = []
    monkeypatch.setattr(pc, "STAGE", tmp_path / "stage")
    monkeypatch.setattr(pc, "kaggle_command", lambda: ["/venv/Scripts/kaggle.exe"])
    monkeypatch.setattr(pc.subprocess, "run", lambda cmd, check=False: calls.append((cmd, check)))
    monkeypatch.setattr(sys, "argv", ["push_code.py", "--user", "u", "--new"])
    pc.main()
    cmd, check = calls[0]
    assert cmd[:3] == ["/venv/Scripts/kaggle.exe", "datasets", "create"] and check is True
    assert (tmp_path / "stage" / "notebook_helpers.py").exists()
    assert (tmp_path / "stage" / "scripts" / "01_build_dataset.py").exists()
    assert (tmp_path / "stage" / "src" / "utils" / "kaggle_cli.py").exists()
