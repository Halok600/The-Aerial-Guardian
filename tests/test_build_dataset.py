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


def _make_det_n(root: Path, n: int) -> Path:
    det = root / "VisDrone2019-DET-train"
    (det / "images").mkdir(parents=True)
    (det / "annotations").mkdir()
    for i in range(n):
        cv2.imwrite(str(det / "images" / f"img{i:03d}.jpg"), np.full((60, 100, 3), 50, np.uint8))
        (det / "annotations" / f"img{i:03d}.txt").write_text("10,10,20,20,1,4,0,0\n")
    return det


def test_det_limit_takes_first_n_in_sorted_order(tmp_path):
    det = _make_det_n(tmp_path, 6)
    out = tmp_path / "ds"
    stats = bd.build_det_split(det, out, "train", DEFAULT_CLASS_MAP, mask=False, limit=3)
    assert stats == {"images": 3, "boxes": 3}
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["img000.jpg", "img001.jpg", "img002.jpg"]
    assert len(list((out / "labels" / "train").iterdir())) == 3


def test_det_limit_none_or_zero_means_no_limit(tmp_path):
    det = _make_det_n(tmp_path, 4)
    assert bd.build_det_split(det, tmp_path / "a", "train", DEFAULT_CLASS_MAP, mask=False)["images"] == 4
    assert bd.build_det_split(det, tmp_path / "b", "train", DEFAULT_CLASS_MAP, mask=False, limit=0)["images"] == 4


def test_det_limit_does_not_count_missing_images(tmp_path):
    det = _make_det_n(tmp_path, 4)
    (det / "images" / "img000.jpg").unlink()
    stats = bd.build_det_split(det, tmp_path / "ds", "train", DEFAULT_CLASS_MAP, mask=False, limit=2)
    assert stats["images"] == 2
    assert sorted(p.name for p in (tmp_path / "ds" / "images" / "train").iterdir()) == ["img001.jpg", "img002.jpg"]


def _make_mot_seqs(root: Path) -> Path:
    mot = root / "VisDrone2019-MOT-train"
    (mot / "annotations").mkdir(parents=True)
    for name in ("seq_a", "seq_b"):
        seq = mot / "sequences" / name
        seq.mkdir(parents=True)
        rows = []
        for f in range(1, 11):
            cv2.imwrite(str(seq / f"{f:07d}.jpg"), np.zeros((48, 64, 3), np.uint8))
            rows.append(f"{f},1,5,5,10,20,1,1,0,0")
        (mot / "annotations" / f"{name}.txt").write_text("\n".join(rows) + "\n")
    return mot


def test_mot_limit_applies_after_stride_and_spans_first_sequences(tmp_path):
    mot = _make_mot_seqs(tmp_path)
    out = tmp_path / "ds"
    # stride 5 -> frames 1 and 6 per sequence (4 samples in total); limit 3 -> seq_a x2, seq_b x1
    stats = bd.build_mot_split(mot, out, "train", stride=5, class_map=DEFAULT_CLASS_MAP, mask=False, limit=3)
    assert stats["images"] == 3
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == [
        "seq_a_0000001.jpg", "seq_a_0000006.jpg", "seq_b_0000001.jpg"]


def test_mot_limit_smaller_than_one_sequence(tmp_path):
    mot = _make_mot_seqs(tmp_path)
    stats = bd.build_mot_split(mot, tmp_path / "ds", "train", stride=5, class_map=DEFAULT_CLASS_MAP,
                               mask=False, limit=1)
    assert stats["images"] == 1


def test_cli_limit_flag_limits_each_source_dir(tmp_path, monkeypatch):
    det = _make_det_n(tmp_path, 5)
    mot = _make_mot_seqs(tmp_path)
    out = tmp_path / "ds"
    monkeypatch.setattr("sys.argv", ["01_build_dataset.py", "--split", "train", "--out", str(out),
                                     "--det", str(det), "--mot", str(mot), "--limit", "2"])
    bd.main()
    names = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert names == ["img000.jpg", "img001.jpg", "seq_a_0000001.jpg", "seq_a_0000006.jpg"]
