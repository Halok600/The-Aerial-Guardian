# %% [markdown]
# # The Aerial Guardian — YOLO26-P2 training on VisDrone (Kaggle)
#
# Builds the 10-class dataset (VisDrone2019-DET-train + every 5th frame of
# VisDrone2019-MOT-train, ignored regions greyed out), loads **real pretrained
# YOLO26 weights into the P2 architecture** (and asserts that it happened),
# then trains within a fixed time budget so the run always finishes inside a
# Kaggle session. Generated from `kaggle/train_notebook.py`.

# %% PARAMS
SCALE = "s"
SMOKE = False
TIME_HOURS = 9.5
MOT_TRAIN_DIR = "/kaggle/input/visdrone-mot/VisDrone_Zips/VisDrone2019-MOT-train/VisDrone2019-MOT-train"
RESUME_FROM = None
USE_BOTH_GPUS = False

# %%
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()  # session clock for the time budget; keep before pip install
SESSION_LIMIT_H = 11.0  # Kaggle kills sessions at 12 h; keep 1 h headroom
RESERVE_H = 1.0  # post-training: final val, checkpoint save, plots, notebook val + copy
INPUT_ROOT = "/kaggle/input"


# Input discovery is stdlib-only and defined here because it must run BEFORE the code dataset
# (and so notebook_helpers.py) can be imported. Kaggle has mounted datasets at different paths
# over time (/kaggle/input/<slug>, /kaggle/input/datasets/<owner>/<slug>, ...).
# Tests extract these two functions from this file with `ast`; keep them self-contained.
def list_tree(root, max_depth=3, max_entries=25):
    """Names only, `max_depth` levels below root; big directories are truncated with their entry count."""
    root = Path(root)
    if not root.is_dir():
        return [f"(missing: {root})"]
    lines = []

    def walk(d, depth):
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError as exc:
            lines.append("  " * depth + f"(unreadable: {exc})")
            return
        for e in entries[:max_entries]:
            is_dir = e.is_dir()
            lines.append("  " * depth + e.name + ("/" if is_dir else ""))
            if is_dir and depth + 1 < max_depth:
                walk(e.path, depth + 1)
        if len(entries) > max_entries:
            lines.append("  " * depth + f"... ({len(entries)} entries, {len(entries) - max_entries} not shown)")

    walk(root, 0)
    return lines


def find_input_dir(marker, root="/kaggle/input", max_depth=4, fallback_name=None):
    """Breadth-first search for the directory that contains the file `marker`.

    Only directories up to `max_depth` below `root` are listed (never descends into the
    huge file trees of the MOT dataset). If no directory holds the marker, a directory named
    `fallback_name` is returned; if neither exists, raises with the input tree.
    """
    root = Path(root)
    by_name = None
    level = [root] if root.is_dir() else []
    for depth in range(max_depth + 1):
        nxt = []
        for d in level:
            try:
                entries = sorted(os.scandir(d), key=lambda e: e.name)
            except OSError:
                continue
            if any(e.name == marker and not e.is_dir() for e in entries):
                return d
            if depth < max_depth:
                for e in entries:
                    if e.is_dir():
                        if fallback_name and e.name == fallback_name and by_name is None:
                            by_name = Path(e.path)
                        nxt.append(Path(e.path))
        level = nxt
    if by_name is not None:
        return by_name
    tree = "\n".join(list_tree(root, 3))
    raise FileNotFoundError(f"no directory with {marker!r} (or named {fallback_name!r}) within {max_depth} "
                            f"levels of {root}. Tree:\n{tree}")


print(f"{INPUT_ROOT} tree (3 levels):")
print("\n".join(list_tree(INPUT_ROOT, 3)))

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174"], check=True)
importlib.invalidate_caches()
import ultralytics  # first import in this kernel; if the image's own version was already imported the assert fires

assert ultralytics.__version__ == "8.4.174", (
    f"ultralytics {ultralytics.__version__} is loaded, expected 8.4.174: restart the kernel and run again")

WORK = Path("/kaggle/working")
TMP = Path("/tmp/ag")
CODE = WORK / "code"
IMGSZ = 1280 if SCALE == "s" else 960
BATCH = 8 if SCALE == "s" else 16
RUN_NAME = f"yolo26{SCALE}-p2"

CODE_SRC = find_input_dir("notebook_helpers.py", INPUT_ROOT, fallback_name="aerial-guardian-code")
print(f"code dataset found at {CODE_SRC}")
shutil.copytree(CODE_SRC, CODE, dirs_exist_ok=True)
sys.path.insert(0, str(CODE))  # push_code.py stages notebook_helpers.py at the dataset root
from notebook_helpers import compute_time_budget, resolve_mot_dir, unpack_code_zips
unpack_code_zips(CODE)  # `--dir-mode zip` zips have no src/ or scripts/ prefix: unpack each into CODE/<name>/
print(sorted(p.name for p in CODE.iterdir()))

# %% Download VisDrone DET (Ultralytics asset mirror, same URLs as VisDrone.yaml)
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

RAW = TMP / "raw"
download([f"{ASSETS_URL}/VisDrone2019-DET-train.zip", f"{ASSETS_URL}/VisDrone2019-DET-val.zip"],
         dir=RAW, threads=2)
for split in ("train", "val"):
    assert (RAW / f"VisDrone2019-DET-{split}" / "images").is_dir(), (
        f"VisDrone2019-DET-{split} was not extracted under {RAW}: {sorted(p.name for p in RAW.iterdir())}")

# %% Build the YOLO dataset with the repo's own builder
DS = TMP / "visdrone10"
YAML = WORK / "visdrone10.yaml"
builder = str(CODE / "scripts" / "01_build_dataset.py")
# SMOKE builds a tiny dataset (--limit = max samples per source dir) so the whole run takes minutes.
LIMIT_TRAIN = ["--limit", "200"] if SMOKE else []
LIMIT_VAL = ["--limit", "100"] if SMOKE else []
train_cmd = [sys.executable, builder, "--split", "train", "--out", str(DS),
             "--det", str(RAW / "VisDrone2019-DET-train"), *LIMIT_TRAIN]
MOT_DIR = resolve_mot_dir(MOT_TRAIN_DIR, INPUT_ROOT)  # searches the input tree if missing, raises if not found
if MOT_DIR:
    train_cmd += ["--mot", MOT_DIR]
else:
    print("MOT_TRAIN_DIR is None (--no-mot): training on DET-train only")
subprocess.run(train_cmd, check=True)
subprocess.run([sys.executable, builder, "--split", "val", "--out", str(DS),
                "--det", str(RAW / "VisDrone2019-DET-val"), "--yaml", str(YAML), *LIMIT_VAL], check=True)

# %% Build YOLO26-P2 and load pretrained weights — fail loudly if the backbone did not transfer
import torch
from ultralytics import YOLO

WEIGHTS_DIR = TMP / "weights"  # pretrained .pt files (and the AMP-check model) download into the cwd
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WEIGHTS_DIR)  # keeps them out of /kaggle/working (all absolute paths below)

if RESUME_FROM:
    model = YOLO(RESUME_FROM)
else:
    model = YOLO(f"yolo26{SCALE}-p2.yaml")
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    model.load(f"yolo26{SCALE}.pt")
    after = model.model.state_dict()
    keys = [k for k in before if not k.endswith("num_batches_tracked")]
    changed = {k for k in keys if not torch.equal(before[k], after[k])}
    backbone = [k for k in keys if int(k.split(".")[1]) <= 10]
    missing = [k for k in backbone if k not in changed]
    print(f"pretrained transfer: {len(changed)}/{len(keys)} tensors ({len(changed) / len(keys):.0%}); "
          f"backbone {len(backbone) - len(missing)}/{len(backbone)}")
    assert not missing, f"backbone tensors NOT transferred: {missing[:5]}"

# %% Train
device = [0, 1] if USE_BOTH_GPUS and torch.cuda.device_count() > 1 else 0
if RESUME_FROM:
    model.train(resume=True)
else:
    budget_h = None
    if not SMOKE:  # Ultralytics' `time` only bounds the training loop; subtract the setup already spent
        used_h = (time.time() - T0) / 3600
        budget_h = compute_time_budget(TIME_HOURS, used_h, SESSION_LIMIT_H, RESERVE_H)
        print(f"training time budget: {budget_h:.2f} h (requested {TIME_HOURS} h, setup used {used_h:.2f} h, "
              f"session limit {SESSION_LIMIT_H} h, reserve {RESERVE_H} h)")
    model.train(
        data=str(YAML), imgsz=640 if SMOKE else IMGSZ, batch=BATCH,
        epochs=1 if SMOKE else 300, time=budget_h,
        fraction=0.25 if SMOKE else 1.0,
        pretrained=f"yolo26{SCALE}.pt",       # also covers the DDP code path
        device=device, workers=4, cos_lr=True, close_mosaic=10,
        flipud=0.0, fliplr=0.5, scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        save_period=5, project=str(WORK / "runs"), name=RUN_NAME, exist_ok=True,
        plots=True, seed=42,
    )

# %% Validate best.pt and save metrics + weights to /kaggle/working
run_dir = Path(model.trainer.save_dir)
final = WORK / f"yolo26{SCALE}-p2-visdrone.pt"
shutil.copy(run_dir / "weights" / "best.pt", final)
metrics = YOLO(str(final)).val(data=str(YAML), imgsz=640 if SMOKE else IMGSZ, batch=BATCH,
                               max_det=1000, plots=False)
names = metrics.names
summary = {
    "scale": SCALE, "imgsz": IMGSZ, "smoke": SMOKE,
    "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
    "per_class_AP50": {names[int(c)]: float(ap) for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)},
}
(WORK / f"metrics_{SCALE}.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
