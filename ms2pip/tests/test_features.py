from subprocess import call
import pandas as pd
import numpy as np

# Run ms2pipC to extract features and targets from an .mgf and .PEPREC files
call(['python', '../ms2pipC.py', '-s', 'hard_test2.mgf', '-w', 'test.h5', 'test.PEPREC', '-c', 'config.file'])
call(['ls'])
test_data = pd.read_hdf('test.h5', 'table')
# Load target values
target_data = pd.read_hdf('target_vectors.h5', 'table')

def test_get_feature_vectors():
    assert test_data[test_data.columns[:-3]].equals(target_data[target_data.columns[:-3]])

def test_get_targetsB():
    for i in range(3):
        assert (np.isclose(test_data[test_data.columns[-2]][i], target_data[target_data.columns[-2]][i]))

def test_get_targetsY():
    for i in range(3):
        assert (np.isclose(test_data[test_data.columns[-1]][i], target_data[target_data.columns[-1]][i]))

def test_get_psmid():
    assert test_data[test_data.columns[-3]].equals(target_data[target_data.columns[-3]])

call(['rm', 'test.h5'])
