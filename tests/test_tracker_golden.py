"""The tracker golden fixtures must match the CURRENT Python tracker and configs/tracker.yaml.

web/tests/*.test.mjs replay these fixtures through the JS port, so a stale fixture would keep
both suites green while the browser demo drifts from the Python numbers.
"""
import json
from dataclasses import asdict

from src.tracker.byte_tracker import TrackerConfig
from tests.helpers import ROOT, load_script

gen = load_script("dev/make_tracker_golden.py")
STALE = "golden fixture is stale — run: .venv/Scripts/python scripts/dev/make_tracker_golden.py"
TOL = 1e-6


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _first_diff(want, got, where="$"):
    """First difference between two JSON trees (floats within TOL), or None."""
    if isinstance(want, dict) and isinstance(got, dict):
        if want.keys() != got.keys():
            return f"{where}: keys {sorted(want)} != {sorted(got)}"
        for k in want:
            d = _first_diff(want[k], got[k], f"{where}.{k}")
            if d:
                return d
        return None
    if isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return f"{where}: length {len(want)} != {len(got)}"
        for i, (w, g) in enumerate(zip(want, got)):
            d = _first_diff(w, g, f"{where}[{i}]")
            if d:
                return d
        return None
    if isinstance(want, float) or isinstance(got, float):
        ok = isinstance(want, (int, float)) and isinstance(got, (int, float)) and abs(want - got) <= TOL
    else:
        ok = want == got and type(want) is type(got)
    return None if ok else f"{where}: {want!r} != {got!r}"


def _roundtrip(obj):
    return json.loads(gen.dumps_fixture(obj))


def test_fixture_configs_equal_tracker_yaml():
    yaml_cfg = asdict(TrackerConfig.from_yaml(ROOT / "configs" / "tracker.yaml"))
    assert _load(gen.GOLDEN_PATH)["config"] == yaml_cfg, STALE
    for sc in _load(gen.PARITY_PATH)["scenarios"]:
        assert {**yaml_cfg, **sc["overrides"]} == sc["config"], f"{sc['name']}: {STALE}"


def test_golden_fixture_matches_current_tracker():
    diff = _first_diff(_load(gen.GOLDEN_PATH), _roundtrip(gen.build_golden()))
    assert diff is None, f"{STALE} ({diff})"


def test_parity_fixture_matches_current_tracker():
    parity, counts = gen.build_parity()            # raises if a scenario stops exercising its branch
    diff = _first_diff(_load(gen.PARITY_PATH), _roundtrip(parity))
    assert diff is None, f"{STALE} ({diff})"
    assert all(counts[e] for e in gen.REQUIRED_EVENTS)


def test_stale_detection_actually_fires():
    fixture = _load(gen.PARITY_PATH)
    fixture["scenarios"][0]["frames"][5]["tracks"][0][1] += 1e-3
    assert _first_diff(fixture, _roundtrip(gen.build_parity()[0])) is not None
