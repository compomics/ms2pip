    """
    Calculate percentage of found peaks
    
    Returns the percentage of values in the series that ar not close to log2(0.001),
    which is the value MS2PIP gives a peak if it was not found in the spectrum.
    """

import sys
from math import isclose

import numpy as np
import pandas as pd


def calc_percentage_found_peaks(series):
    """
    Calculate percentage of found peaks
    
    Returns the percentage of values in the series that ar not close to log2(0.001),
    which is the value MS2PIP gives a peak if it was not found in the spectrum.
    """
    zero = np.log2(0.001)
    try:
        x = series.apply(lambda x: isclose(x, zero, abs_tol=0.001)).value_counts(normalize=True)[False]
    except KeyError:
        x = 0.0
    return x
    
    
def main():
    if sys.argv[1].endswith('.csv'):
        vectors = pd.read_csv(sys.argv[1])
    elif sys.argv[1].endswith('.h5'):
        vectors = pd.read_hdf(sys.argv[1], key='table')
    elif sys.argv[1].endswith('pkl'):
        vectors = pd.read_pkl(sys.argv[1])
    else:
        print('Unknown file extension')
        exit(1)
    
    for ion_type in [col.split('targets_')[1] for col in vectors.columns if col.startswith('targets_')]:
        perc_found = calc_percentage_found_peaks(vectors['targets_' + ion_type])
        print('Percentage of found {}-ion peaks: {}'.format(ion_type, perc_found))
        
if __name__ == '__main__':
    main()