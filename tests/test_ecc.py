import cv2
import numpy as np
import pytest

from src.tracker.ecc_compensator import ECCCompensator


def _texture(w, h, seed=0):
    base = np.random.default_rng(seed).integers(0, 255, (h // 8, w // 8)).astype(np.uint8)
    return cv2.cvtColor(cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)


def _shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("w,h,dx,dy", [(1280, 720, 12, -8), (1000, 1000, 24, 16)])
def test_recovers_translation(w, h, dx, dy):
    prev = _texture(w, h)
    cmc = ECCCompensator()
    assert np.allclose(cmc.update(prev), np.eye(2, 3))     # first frame -> identity
    warp = cmc.update(_shift(prev, dx, dy))
    assert warp[0, 2] == pytest.approx(dx, abs=1.0)
    assert warp[1, 2] == pytest.approx(dy, abs=1.0)


def test_flat_frames_fall_back_to_identity():
    cmc = ECCCompensator()
    flat = np.zeros((360, 640, 3), np.uint8)
    cmc.update(flat)
    assert np.allclose(cmc.update(flat), np.eye(2, 3))


def test_rescale_warp_is_exact_similarity_transform():
    theta = np.deg2rad(2.0)
    small = np.array([[np.cos(theta), -np.sin(theta), 3.0],
                      [np.sin(theta), np.cos(theta), -2.0]])
    sx, sy = 4.0, 5.0
    full = ECCCompensator.rescale_warp(small, sx, sy)
    p_full = np.array([400.0, 300.0])
    via_small = small @ np.r_[p_full / [sx, sy], 1.0] * [sx, sy]
    assert np.allclose(full @ np.r_[p_full, 1.0], via_small)
