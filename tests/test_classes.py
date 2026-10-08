import pytest

from src.utils.classes import (
    CLASS_TO_GROUP, GROUP_COLORS_BGR, GROUP_LABELS, GROUP_ORDER, GROUPS,
    VISDRONE_NAMES, class_name, group_of,
)


def test_ten_classes_in_visdrone_order():
    assert VISDRONE_NAMES == (
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor",
    )


def test_every_class_in_exactly_one_group():
    members = sorted(c for cs in GROUPS.values() for c in cs)
    assert members == list(range(10))
    assert set(CLASS_TO_GROUP) == set(range(10))


def test_group_lookup():
    assert group_of(0) == "person" and group_of(1) == "person"
    assert group_of(3) == "vehicle" and group_of(8) == "vehicle"
    assert group_of(9) == "two-wheeler"
    assert group_of(7) == "tricycle"
    assert class_name(3) == "car"


def test_unknown_class_raises():
    with pytest.raises(ValueError):
        group_of(10)


def test_group_metadata_complete():
    assert GROUP_ORDER == ("person", "vehicle", "two-wheeler", "tricycle")
    for g in GROUP_ORDER:
        assert g in GROUP_LABELS and g in GROUP_COLORS_BGR
