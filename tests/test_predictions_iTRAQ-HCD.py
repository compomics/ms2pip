from subprocess import call
import pandas as pd
import numpy as np

# Run ms2pipC to predict peak intensities from a PEPREC file (iTRAQ HCD model)
call(['python', '../ms2pipC.py', 'test.PEPREC', '-c', 'config.file', '-i'])
call(['ls'])
test_data = pd.read_csv('test.PEPREC_predictions.csv')
# Load target values & peptide file
target_data = pd.read_csv('target_iTRAQ-HCD.PEPREC_predictions.csv')
pepfile = pd.read_csv('test.PEPREC', sep=" ", index_col=False,
                        dtype={"spec_id": str, "modifications": str})

def test_all_spec():
    assert set(test_data.spec_id.unique()) == set(pepfile.spec_id)

def test_amount_peaks():
    for pep in ['peptide1', 'peptide2', 'peptide3']:
        peplen = len(pepfile[pepfile.spec_id == pep].peptide.values[0])
        assert len(test_data[test_data.spec_id == pep]) == (2 * peplen) - 2

def test_peak_ints_b():
    for pep in target_data.spec_id.unique():
        tmp_test = test_data[test_data.spec_id == pep]
        tmp_test = tmp_test[tmp_test.ion == 'b']
        tmp_target = target_data[target_data.spec_id == pep]
        tmp_target = tmp_target[tmp_target.ion == 'b']
        for no in tmp_target.ionnumber:
            assert(tmp_test[tmp_test.ionnumber == no]['prediction'].values[0] == tmp_target[tmp_target.ionnumber == no]['prediction'].values[0])

def test_peak_ints_y():
    for pep in target_data.spec_id.unique():
        tmp_test = test_data[test_data.spec_id == pep]
        tmp_test = tmp_test[tmp_test.ion == 'y']
        tmp_target = target_data[target_data.spec_id == pep]
        tmp_target = tmp_target[tmp_target.ion == 'y']
        for no in tmp_target.ionnumber:
            assert(tmp_test[tmp_test.ionnumber == no]['prediction'].values[0] == tmp_target[tmp_target.ionnumber == no]['prediction'].values[0])

call(['rm', 'test.PEPREC_predictions.csv'])
