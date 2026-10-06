import numpy as np
import pytest

from src.tracker.matching import (
    GATED_COST, fuse_score, gate_groups, ioa_matrix, iou_matrix, linear_assignment,
)


def test_iou_values():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert np.allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_empty_shapes():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    matches, ur, uc = linear_assignment(np.zeros((0, 3)), 0.5)
    assert matches.shape == (0, 2) and ur == [] and uc == [0, 1, 2]


def test_ioa_is_relative_to_first_box():
    small = np.array([[0, 0, 10, 10]], float)
    big = np.array([[0, 0, 100, 100]], float)
    assert ioa_matrix(small, big)[0, 0] == pytest.approx(1.0)
    assert ioa_matrix(big, small)[0, 0] == pytest.approx(0.01)


def test_fuse_score():
    assert fuse_score(np.array([[0.2]]), np.array([0.5]))[0, 0] == pytest.approx(0.6)


def test_gate_groups_blocks_cross_group_pairs():
    gated = gate_groups(np.zeros((2, 2)), ["person", "vehicle"], ["person", "person"])
    assert np.array_equal(gated, [[0, 0], [GATED_COST, GATED_COST]])


def test_linear_assignment_respects_threshold():
    cost = np.array([[0.1, 0.9], [0.8, 0.95]])
    matches, ur, uc = linear_assignment(cost, thresh=0.85)
    assert matches.tolist() == [[0, 0]]
    assert ur == [1] and uc == [1]
