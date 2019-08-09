import numpy as np
from math import acos, sqrt
from scipy.stats import pearsonr


def calc_correlations(df):
	correlations = df.groupby(['spec_id', 'ion'])[['target', 'prediction']].corr().iloc[::2]['prediction']
	correlations.index = correlations.index.droplevel(2)
	correlations = correlations.to_frame().reset_index()
	correlations.columns = ['spec_id', 'ion', 'pearsonr']
	return correlations


def ms2pip_pearson(true, pred):
    """
    Return pearson of tic-normalized, log-transformed intensities, 
    the MS2PIP way.
    """
    tic_norm = lambda x: x / np.sum(x)
    log_transform = lambda x: np.log2(x + 0.001)
    corr = pearsonr(
        log_transform(tic_norm(true)), 
        log_transform(tic_norm(pred))
    )[0]
    return corr


def spectral_angle(true, pred, epsilon=1e-7):
    """
    Return square root normalized spectral angle.
    See https://doi.org/10.1074/mcp.O113.036475
    """
    
    l2_normalize = lambda x: x / sqrt(max(sum(x**2), epsilon))
    
    pred_norm = l2_normalize(pred)
    true_norm = l2_normalize(true)
    
    spectral_angle = 1 - (2 * acos(np.dot(pred_norm, true_norm)) / np.pi)

    return spectral_angle
