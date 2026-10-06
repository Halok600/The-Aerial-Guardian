"""Tests for the inline input-discovery functions of kaggle/train_notebook.py (extracted with ast)."""
import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "kaggle" / "train_notebook.py"


def _load(*names, os_module=os):
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    ns = {"os": os_module, "Path": Path}
    for name in names:
        exec(compile(ast.Module([funcs[name]], type_ignores=[]), str(NOTEBOOK), "exec"), ns)
    return ns


NS = _load("list_tree", "find_input_dir")
find_input_dir, list_tree = NS["find_input_dir"], NS["list_tree"]


def _code_ds(base: Path) -> Path:
    base.mkdir(parents=True)
    (base / "notebook_helpers.py").write_text("x")
    (base / "src").mkdir()
    return base


def test_found_at_top_level(tmp_path):
    ds = _code_ds(tmp_path / "input" / "aerial-guardian-code")
    assert find_input_dir("notebook_helpers.py", tmp_path / "input") == ds


def test_found_under_datasets_owner_layout_with_noise(tmp_path):
    root = tmp_path / "input"
    (root / "visdrone2019-mot" / "VisDrone_Zips" / "VisDrone2019-MOT-train").mkdir(parents=True)
    (root / "readme.txt").write_text("noise")
    (root / "datasets" / "other" / "thing").mkdir(parents=True)
    (root / "datasets" / "other" / "thing" / "data.csv").write_text("1")
    ds = _code_ds(root / "datasets" / "halok600" / "aerial-guardian-code")
    assert find_input_dir("notebook_helpers.py", root, fallback_name="aerial-guardian-code") == ds


def test_fallback_to_directory_name_when_marker_missing(tmp_path):
    root = tmp_path / "input"
    d = root / "datasets" / "halok600" / "aerial-guardian-code"
    d.mkdir(parents=True)
    assert find_input_dir("notebook_helpers.py", root, fallback_name="aerial-guardian-code") == d


def test_marker_wins_over_name_fallback(tmp_path):
    root = tmp_path / "input"
    (root / "aerial-guardian-code").mkdir(parents=True)
    ds = _code_ds(root / "zzz" / "renamed")
    assert find_input_dir("notebook_helpers.py", root, fallback_name="aerial-guardian-code") == ds


def test_directory_named_like_marker_is_not_a_hit(tmp_path):
    root = tmp_path / "input"
    (root / "ds" / "notebook_helpers.py").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        find_input_dir("notebook_helpers.py", root)


def test_deeper_than_max_depth_is_not_found_and_error_lists_tree(tmp_path):
    root = tmp_path / "input"
    _code_ds(root / "a" / "b" / "c" / "d" / "e" / "aerial-guardian-code")   # depth 6
    with pytest.raises(FileNotFoundError) as e:
        find_input_dir("notebook_helpers.py", root, max_depth=4, fallback_name="aerial-guardian-code")
    msg = str(e.value)
    assert "a/" in msg and "b/" in msg and "c/" in msg and "d/" not in msg   # tree printed to 3 levels
    assert "notebook_helpers.py" in msg


def test_max_depth_boundary_is_inclusive(tmp_path):
    root = tmp_path / "input"
    ds = _code_ds(root / "a" / "b" / "c" / "d")                              # depth 4
    assert find_input_dir("notebook_helpers.py", root, max_depth=4) == ds
    with pytest.raises(FileNotFoundError):
        find_input_dir("notebook_helpers.py", root, max_depth=3)


def test_not_present_raises_with_listing(tmp_path):
    root = tmp_path / "input"
    (root / "other-ds" / "stuff").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="other-ds/"):
        find_input_dir("notebook_helpers.py", root, fallback_name="aerial-guardian-code")


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        find_input_dir("notebook_helpers.py", tmp_path / "nope")


def test_never_lists_directories_deeper_than_max_depth(tmp_path):
    root = tmp_path / "input"
    (root / "a" / "b" / "c" / "d" / "e" / "sequences").mkdir(parents=True)
    scanned = []

    class SpyOs:
        @staticmethod
        def scandir(path):
            scanned.append(Path(path))
            return os.scandir(path)

    ns = _load("list_tree", "find_input_dir", os_module=SpyOs)
    with pytest.raises(FileNotFoundError):
        ns["find_input_dir"]("notebook_helpers.py", root, max_depth=3)
    rels = sorted("/".join(p.relative_to(root).parts) for p in scanned)
    assert "a/b/c" in rels            # depth 3 is listed
    assert "a/b/c/d" not in rels      # depth 4 never is


def test_list_tree_truncates_big_dirs_and_limits_depth(tmp_path):
    root = tmp_path / "input"
    (root / "ds").mkdir(parents=True)
    for i in range(40):
        (root / "ds" / f"f{i:02d}.txt").write_text("x")
    (root / "ds" / "x" / "y" / "z").mkdir(parents=True)
    lines = list_tree(root, max_depth=3, max_entries=10)
    assert lines[0] == "ds/"
    assert any("entries" in line and "not shown" in line for line in lines)
    assert not any(line.strip() == "z/" for line in lines)


def test_notebook_wiring_uses_discovery_and_prints_tree_early():
    src = NOTEBOOK.read_text(encoding="utf-8")
    assert "/kaggle/input/aerial-guardian-code" not in src
    assert 'find_input_dir("notebook_helpers.py"' in src and "shutil.copytree(CODE_SRC, CODE" in src
    assert src.index("list_tree(INPUT_ROOT") < src.index("\"pip\", \"install\"")
    assert "resolve_mot_dir(MOT_TRAIN_DIR, INPUT_ROOT)" in src
    assert '__version__ == "8.4.174"' in src and "importlib.invalidate_caches()" in src
    assert '"--limit", "200"' in src and '"--limit", "100"' in src


def test_params_lines_still_strict_key_value():
    src = NOTEBOOK.read_text(encoding="utf-8")
    params = src.split("# %% PARAMS\n")[1].split("\n# %%")[0].strip().splitlines()
    assert [ln.split(" = ")[0] for ln in params] == [
        "SCALE", "SMOKE", "TIME_HOURS", "MOT_TRAIN_DIR", "RESUME_FROM", "USE_BOTH_GPUS"]
