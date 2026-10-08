import numpy as np
import pytest

from src.tracker.kalman_filter import KalmanFilterXYAH, tlbr_to_xyah, xyah_to_tlbr


def test_roundtrip_conversion():
    tlbr = np.array([10.0, 20.0, 30.0, 60.0])
    xyah = tlbr_to_xyah(tlbr)
    assert np.allclose(xyah, [20.0, 40.0, 0.5, 40.0])
    assert np.allclose(xyah_to_tlbr(xyah), tlbr)


def test_initiate_shapes_and_zero_velocity():
    mean, cov = KalmanFilterXYAH().initiate(np.array([100.0, 50.0, 0.5, 40.0]))
    assert mean.shape == (8,) and cov.shape == (8, 8)
    assert np.allclose(mean[:4], [100, 50, 0.5, 40]) and np.allclose(mean[4:], 0)


def test_predict_applies_velocity_and_grows_uncertainty():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 40.0]))
    mean[4] = 3.0
    m2, c2 = kf.predict(mean, cov)
    assert m2[0] == pytest.approx(103.0)
    assert c2[0, 0] > cov[0, 0]


def test_update_moves_toward_measurement_and_shrinks_uncertainty():
    kf = KalmanFilterXYAH()
    mean, cov = kf.predict(*kf.initiate(np.array([100.0, 50.0, 0.5, 40.0])))
    m3, c3 = kf.update(mean, cov, np.array([110.0, 50.0, 0.5, 40.0]))
    assert 100.0 < m3[0] <= 110.0
    assert c3[0, 0] < cov[0, 0]


def test_noise_scales_with_box_height():
    kf = KalmanFilterXYAH()
    _, small = kf.initiate(np.array([0.0, 0.0, 0.5, 10.0]))
    _, big = kf.initiate(np.array([0.0, 0.0, 0.5, 100.0]))
    assert big[0, 0] / small[0, 0] == pytest.approx(100.0)
