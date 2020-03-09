import os

import pandas as pd
import numpy as np

from ms2pip.ms2pipC import MS2PIP

class TestPredictions:
    def __init__(self):
        # Run ms2pipC to predict peak intensities from a PEPREC file (HCD model)
        params = {
            "ms2pip": {
                "ptm": [
                    "Oxidation,15.994915,opt,M",
                    "Carbamidomethyl,57.021464,opt,C",
                    "Acetyl,42.010565,opt,N-term",
                ],
                "sptm": [],
                "gptm": [],
                "frag_method": "HCD",
                "frag_error": 0.02,
                "out": "csv",
            }
        }
        ms2pip = MS2PIP("tests/test_data/test.peprec", params=params)
        ms2pip.run()

        self.test_data = pd.read_csv("tests/test_data/test_HCD_predictions.csv")
        self.target_data = pd.read_csv("tests/test_data/target_HCD_predictions.csv")
        self.pepfile = pd.read_csv(
            "tests/test_data/test.peprec",
            sep=" ",
            index_col=False,
            dtype={"spec_id": str, "modifications": str},
        )

    def __del__(self):
        os.remove("tests/test_data/test_HCD_predictions.csv")

    def test_all_spec(self):
        assert set(self.test_data.spec_id.unique()) == set(self.pepfile.spec_id)

    def test_amount_peaks(self):
        for pep in ["peptide1", "peptide2", "peptide3"]:
            peplen = len(self.pepfile[self.pepfile.spec_id == pep].peptide.values[0])
            assert len(self.test_data[self.test_data.spec_id == pep]) == (2 * peplen) - 2

    def test_peak_ints_b(self):
        for pep in self.target_data.spec_id.unique():
            tmp_test = self.test_data[self.test_data.spec_id == pep]
            tmp_test = tmp_test[tmp_test.ion == "b"]
            tmp_target = self.target_data[self.target_data.spec_id == pep]
            tmp_target = tmp_target[tmp_target.ion == "b"]
            for no in tmp_target.ionnumber:
                assert (
                    tmp_test[tmp_test.ionnumber == no]["prediction"].values[0]
                    == tmp_target[tmp_target.ionnumber == no]["prediction"].values[0]
                )

    def test_peak_ints_y(self):
        for pep in self.target_data.spec_id.unique():
            tmp_test = self.test_data[self.test_data.spec_id == pep]
            tmp_test = tmp_test[tmp_test.ion == "y"]
            tmp_target = self.target_data[self.target_data.spec_id == pep]
            tmp_target = tmp_target[tmp_target.ion == "y"]
            for no in tmp_target.ionnumber:
                assert (
                    tmp_test[tmp_test.ionnumber == no]["prediction"].values[0]
                    == tmp_target[tmp_target.ionnumber == no]["prediction"].values[0]
                )
