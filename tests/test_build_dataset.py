import filecmp
from pathlib import Path

import cv2
import numpy as np

from src.data.visdrone import DEFAULT_CLASS_MAP
from tests.helpers import load_script

bd = load_script("01_build_dataset.py")


def _make_det(root: Path) -> Path:
    det = root / "VisDrone2019-DET-train"
    (det / "images").mkdir(parents=True)
    (det / "annotations").mkdir()
    cv2.imwrite(str(det / "images" / "a.jpg"), np.full((600, 1000, 3), 50, np.uint8))
    (det / "annotations" / "a.txt").write_text(
        "684,8,273,116,0,0,0,0\n1,1,10,20,1,1,0,0\n50,60,30,15,1,4,0,1\n")
    return det


def test_det_train_masks_ignore_regions(tmp_path):
    det = _make_det(tmp_path)
    out = tmp_path / "ds"
    stats = bd.build_det_split(det, out, "train", DEFAULT_CLASS_MAP, mask=True)
    assert stats == {"images": 1, "boxes": 2}
    img = cv2.imread(str(out / "images" / "train" / "a.jpg"))
    assert abs(img[10:120, 690:950].mean() - 114) < 4
    assert abs(img[300:400, 100:400].mean() - 50) < 4
    lines = (out / "labels" / "train" / "a.txt").read_text().splitlines()
    assert [l.split()[0] for l in lines] == ["0", "3"]


def test_det_val_copies_image_unmodified(tmp_path):
    det = _make_det(tmp_path)
    out = tmp_path / "ds"
    bd.build_det_split(det, out, "val", DEFAULT_CLASS_MAP, mask=False)
    assert filecmp.cmp(det / "images" / "a.jpg", out / "images" / "val" / "a.jpg", shallow=False)


def test_masked_rebuild_does_not_corrupt_source_via_hardlink(tmp_path):
    det = _make_det(tmp_path)
    src = det / "images" / "a.jpg"
    original = src.read_bytes()
    out = tmp_path / "ds"
    bd.build_det_split(det, out, "train", DEFAULT_CLASS_MAP, mask=False)   # hardlinks the source
    bd.build_det_split(det, out, "train", DEFAULT_CLASS_MAP, mask=True)    # must not write through the link
    assert src.read_bytes() == original
    img = cv2.imread(str(out / "images" / "train" / "a.jpg"))
    assert abs(img[10:120, 690:950].mean() - 114) < 4


def test_mot_subsamples_with_stride(tmp_path):
    mot = tmp_path / "VisDrone2019-MOT-train"
    seq = mot / "sequences" / "uav_test"
    seq.mkdir(parents=True)
    (mot / "annotations").mkdir()
    rows = []
    for f in range(1, 11):
        cv2.imwrite(str(seq / f"{f:07d}.jpg"), np.zeros((48, 64, 3), np.uint8))
        rows.append(f"{f},1,5,5,10,20,1,1,0,0")
    (mot / "annotations" / "uav_test.txt").write_text("\n".join(rows) + "\n")
    out = tmp_path / "ds"
    stats = bd.build_mot_split(mot, out, "train", stride=5, class_map=DEFAULT_CLASS_MAP, mask=True)
    assert stats == {"images": 2, "boxes": 2}
    names = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert names == ["uav_test_0000001.jpg", "uav_test_0000006.jpg"]


def test_yaml_lists_ten_names(tmp_path):
    yaml_path = tmp_path / "cfg" / "visdrone10.yaml"
    bd.write_dataset_yaml(tmp_path / "ds", yaml_path, list(bd.VISDRONE_NAMES))
    text = yaml_path.read_text()
    assert "train: images/train" in text and "val: images/val" in text
    assert "nc: 10" in text and "  9: motor" in text
