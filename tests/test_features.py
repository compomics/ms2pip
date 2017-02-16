from subprocess import call
import pandas as pd

# Run ms2pipC to extract features and targets from an .mgf and .PEPREC files
call(['python', '../ms2pipC.py', '-s', 'hard_test2.mgf', '-w', 'test', 'test.PEPREC'])
call(['ls'])
test_data = pd.read_hdf('test_vectors.h5', 'table')
# Load target values
target_data = pd.read_hdf('target_vectors.h5', 'table')

def test_get_feature_vectors():
    assert test_data[test_data.columns[:-3]].equals(target_data[target_data.columns[:-3]])

def test_get_targets():
    assert test_data[test_data.columns[-3:]].equals(target_data[target_data.columns[-3:]])

call(['rm', 'test_vectors.h5'])
