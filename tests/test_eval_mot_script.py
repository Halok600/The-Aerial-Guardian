from tests.helpers import load_script


def _make_split(root, n_frames=6):
    seq = root / "sequences" / "s1"
    seq.mkdir(parents=True)
    for i in range(1, n_frames + 1):
        (seq / f"{i:07d}.jpg").write_bytes(b"")
    (root / "annotations").mkdir()
    lines = [f"{f},1,10,10,20,40,1,1,0,0" for f in range(1, n_frames + 1)]
    lines += [f"{f},2,100,50,60,30,1,4,0,0" for f in range(1, n_frames + 1)]
    lines.append("1,-1,500,0,100,100,0,0,0,0")
    (root / "annotations" / "s1.txt").write_text("\n".join(lines) + "\n")


def test_prepare_gt_truncates_gt_and_seqlength(tmp_path):
    mod = load_script("05_eval_mot.py")
    _make_split(tmp_path / "split")
    ignores = mod.prepare_gt(tmp_path / "split", ["s1"], tmp_path / "gt", max_frames=3)
    person = tmp_path / "gt" / "person-test" / "s1"
    assert "seqLength=3" in (person / "seqinfo.ini").read_text()
    frames = {int(l.split(",")[0]) for l in (person / "gt" / "gt.txt").read_text().splitlines()}
    assert frames == {1, 2, 3}
    assert max(ignores["s1"]) == 3 and len(ignores["s1"][1]) == 1


def test_prepare_gt_without_limit_uses_all_frames(tmp_path):
    mod = load_script("05_eval_mot.py")
    _make_split(tmp_path / "split")
    mod.prepare_gt(tmp_path / "split", ["s1"], tmp_path / "gt")
    assert "seqLength=6" in (tmp_path / "gt" / "vehicle-test" / "s1" / "seqinfo.ini").read_text()


def test_table_orders_baseline_first():
    mod = load_script("05_eval_mot.py")
    res = {"person": {k: {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0, "IDSW": 4}
                      for k in ("ours_cmc", "ours_nocmc", "ultralytics_bytetrack")}}
    rows = [l for l in mod.table(res, mod.TABLE_NAMES).splitlines() if l.startswith("| ") and "Tracker" not in l]
    assert [r.split("|")[1].strip() for r in rows] == [
        "Ultralytics ByteTrack (default)", "Ours (no camera-motion comp.)", "Ours + ECC camera-motion comp."]
