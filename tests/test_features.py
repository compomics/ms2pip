from subprocess import call
import pandas as pd

def test_get_feature_vectors():
    call(['python', '../ms2pipC.py', '-s', 'hard_test2.mgf', '-w', 'test', 'test.PEPREC'])
    target_data = pd.read_pickle('target_features.pkl')
    test_data = pd.read_pickle('test_vectors.pkl')

    assert test_data.equals(target_data)
