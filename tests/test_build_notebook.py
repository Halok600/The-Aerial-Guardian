import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "kaggle" / "build_notebook.py")
bn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bn)


def test_render_substitutes_params():
    src = bn.render({"SCALE": "n", "SMOKE": True, "TIME_HOURS": 6.5,
                     "MOT_TRAIN_DIR": "/kaggle/input/x/VisDrone2019-MOT-train", "RESUME_FROM": None})
    assert 'SCALE = "n"' in src and "SMOKE = True" in src and "TIME_HOURS = 6.5" in src
    assert 'MOT_TRAIN_DIR = "/kaggle/input/x/VisDrone2019-MOT-train"' in src
    assert "RESUME_FROM = None" in src


def test_metadata():
    meta = bn.kernel_metadata("alice", "bob/visdrone-mot", "s")
    assert meta["id"] == "alice/aerial-guardian-train-s"
    assert meta["dataset_sources"] == ["alice/aerial-guardian-code", "bob/visdrone-mot"]
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]


def test_default_mot_train_dir_matches_chosen_dataset_layout():
    assert bn.default_mot_train_dir("akhileswarelluru/visdrone2019-mot") == (
        "/kaggle/input/visdrone2019-mot/VisDrone_Zips/VisDrone2019-MOT-train/VisDrone2019-MOT-train")


def test_notebook_source_default_matches_builder_default():
    src = bn.render({})
    assert 'MOT_TRAIN_DIR = "/kaggle/input/visdrone-mot/VisDrone_Zips/VisDrone2019-MOT-train/VisDrone2019-MOT-train"' in src


def test_time_hours_default_is_pinned_at_9_5_everywhere():
    import re
    import sys
    assert bn.DEFAULT_TIME_HOURS == 9.5
    assert re.search(r"^TIME_HOURS = 9\.5$", bn.render({}), flags=re.M)
    argv = sys.argv
    sys.argv = ["build_notebook.py", "--user", "u", "--mot-dataset", "o/s"]
    try:
        import argparse
        captured = {}
        orig = argparse.ArgumentParser.parse_args

        def spy(self, *a, **k):
            ns = orig(self, *a, **k)
            captured["ns"] = ns
            raise SystemExit  # stop before writing any files
        argparse.ArgumentParser.parse_args = spy
        try:
            try:
                bn.main()
            except SystemExit:
                pass
        finally:
            argparse.ArgumentParser.parse_args = orig
    finally:
        sys.argv = argv
    assert captured["ns"].time_hours == 9.5


def test_render_writes_python_literals_for_none_and_bool():
    src = bn.render({"MOT_TRAIN_DIR": None, "SMOKE": False, "RESUME_FROM": "/kaggle/input/x/last.pt"})
    assert "MOT_TRAIN_DIR = None" in src and "SMOKE = False" in src
    assert 'RESUME_FROM = "/kaggle/input/x/last.pt"' in src
    assert "null" not in src.split("# %%")[2]


def test_params_cell_is_one_assignment_per_line_after_render():
    src = bn.render({"MOT_TRAIN_DIR": None, "TIME_HOURS": 9.5})
    params = src.split("# %% PARAMS\n")[1].split("\n# %%")[0].strip().splitlines()
    assert all(" = " in line and not line.startswith(" ") for line in params), params
    assert [line.split(" = ")[0] for line in params] == [
        "SCALE", "SMOKE", "TIME_HOURS", "MOT_TRAIN_DIR", "RESUME_FROM", "USE_BOTH_GPUS"]


def test_mot_dir_param_no_mot_is_explicit_none():
    assert bn.mot_dir_param("o/s", None, True) is None
    assert bn.mot_dir_param("o/s", "/custom", False) == "/custom"
    assert bn.mot_dir_param("o/s", None, False) == bn.default_mot_train_dir("o/s")


def test_notebook_uses_helpers_and_budget():
    nb_src = (ROOT / "kaggle" / "train_notebook.py").read_text(encoding="utf-8")
    first_code = nb_src.index("T0 = time.time()")
    assert first_code < nb_src.index("pip")  # clock starts before pip install
    assert "from notebook_helpers import" in nb_src and "compute_time_budget(" in nb_src
    assert "time=budget_h" in nb_src and "SESSION_LIMIT_H = 11.0" in nb_src and "RESERVE_H = 1.0" in nb_src
