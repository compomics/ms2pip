import numpy as np


def ms2pip_pearson(true, pred):
    """
    Return pearson of tic-normalized, log-transformed intensities,
    the MS2PIP way.
    """
    tic_norm = lambda x: x / np.sum(x)
    log_transform = lambda x: np.log2(x + 0.001)
    corr = np.corrcoef(log_transform(tic_norm(true)), log_transform(tic_norm(pred)))[0][1]
    return corr


def spectral_angle(true, pred, epsilon=1e-7):
    """
    Return square root normalized spectral angle.
    See https://doi.org/10.1074/mcp.O113.036475
    """
    pred_norm = pred / max(np.linalg.norm(pred), epsilon)
    true_norm = true / max(np.linalg.norm(true), epsilon)
    spectral_angle = 1 - (2 * np.arccos(np.dot(pred_norm, true_norm)) / np.pi)
    return spectral_angle
