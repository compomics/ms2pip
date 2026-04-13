import numpy as np

from ms2pip.correlation import ms2pip_pearson, spectral_angle


def test_ms2pip_pearson():
    true = np.array([100.0, 200.0, 50.0, 300.0])
    pred = np.array([90.0, 210.0, 40.0, 310.0])

    corr = ms2pip_pearson(true, pred)

    assert isinstance(corr, float)
    assert 0.9 < corr <= 1.0


def test_ms2pip_pearson_identical():
    arr = np.array([100.0, 200.0, 50.0, 300.0])
    corr = ms2pip_pearson(arr, arr)

    assert abs(corr - 1.0) < 1e-6


def test_ms2pip_pearson_anticorrelated():
    true = np.array([100.0, 200.0, 300.0])
    pred = np.array([300.0, 200.0, 100.0])

    corr = ms2pip_pearson(true, pred)

    assert corr < 0


def test_spectral_angle():
    true = np.array([100.0, 200.0, 50.0, 300.0])
    pred = np.array([90.0, 210.0, 40.0, 310.0])

    sa = spectral_angle(true, pred)

    assert isinstance(sa, float)
    assert 0.0 < sa <= 1.0


def test_spectral_angle_identical():
    arr = np.array([100.0, 200.0, 50.0, 300.0])
    sa = spectral_angle(arr, arr)

    assert abs(sa - 1.0) < 1e-6


def test_spectral_angle_orthogonal():
    true = np.array([1.0, 0.0])
    pred = np.array([0.0, 1.0])

    sa = spectral_angle(true, pred)

    assert sa < 0.5
