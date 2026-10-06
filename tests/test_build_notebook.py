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
