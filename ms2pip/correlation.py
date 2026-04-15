import numpy as np


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Pearson correlation for two 1D arrays."""
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm @ xm) * (ym @ ym))
    if denom == 0:
        return 0.0
    return float(xm @ ym / denom)


def ms2pip_pearson(true, pred):
    """Calculate Pearson correlation, including tic-normalization and log-transformation."""

    def tic_norm(x):
        return x / np.sum(x)

    def log_transform(x):
        return np.log2(x + 0.001)

    return pearson(log_transform(tic_norm(true)), log_transform(tic_norm(pred)))


def spectral_angle(true, pred, epsilon=1e-7):
    """
    Calculate square root normalized spectral angle.

    See https://doi.org/10.1074/mcp.O113.036475.
    """
    pred_norm = pred / max(np.linalg.norm(pred), epsilon)
    true_norm = true / max(np.linalg.norm(true), epsilon)
    spectral_angle = 1 - (2 * np.arccos(np.dot(pred_norm, true_norm)) / np.pi)
    return spectral_angle
