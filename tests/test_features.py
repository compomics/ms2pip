from subprocess import call
import pandas as pd

# Run ms2pipC to extract features and targets from an .mgf and .PEPREC files
call(['python', '../ms2pipC.py', '-s', 'hard_test2.mgf', '-w', 'test', 'test.PEPREC'])
test_data = pd.read_pickle('test_vectors.pkl')

# Load target values
target_data = pd.read_pickle('target_features.pkl')

def test_get_feature_vectors():
    assert test_data[test_data.columns[:-3]].equals(target_data[target_data.columns[:-3]])

def test_get_targets():
    assert test_data[test_data.columns[-3:]].equals(target_data[target_data.columns[-3:]])
