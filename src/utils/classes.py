"""src/utils/classes.py — VisDrone classes, tracker class groups and colours.

Single source of truth shared by the dataset builder, tracker (gating),
renderer (colours, counters) and evaluation. web/js/classes.js mirrors it.
"""
from __future__ import annotations

VISDRONE_NAMES: tuple[str, ...] = (
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
)

# Groups gate tracker association (a person can never take over a car's ID)
# and drive the per-group unique counters in the HUD.
GROUPS: dict[str, tuple[int, ...]] = {
    "person": (0, 1),
    "vehicle": (3, 4, 5, 8),
    "two-wheeler": (2, 9),
    "tricycle": (6, 7),
}
GROUP_ORDER: tuple[str, ...] = ("person", "vehicle", "two-wheeler", "tricycle")
GROUP_LABELS: dict[str, str] = {
    "person": "Persons",
    "vehicle": "Vehicles",
    "two-wheeler": "Two-wheelers",
    "tricycle": "Tricycles",
}
GROUP_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "person": (235, 120, 30),       # blue
    "vehicle": (80, 200, 60),       # green
    "two-wheeler": (20, 150, 255),  # orange
    "tricycle": (200, 80, 170),     # purple
}
CLASS_TO_GROUP: dict[int, str] = {c: g for g, cs in GROUPS.items() for c in cs}


def group_of(cls: int) -> str:
    """Return the tracker group of a YOLO class index (0-9)."""
    try:
        return CLASS_TO_GROUP[int(cls)]
    except KeyError:
        raise ValueError(f"unknown VisDrone class index: {cls}") from None


def class_name(cls: int) -> str:
    return VISDRONE_NAMES[int(cls)]
