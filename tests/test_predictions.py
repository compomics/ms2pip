import os

import pandas as pd

from ms2pip.ms2pipC import MS2PIP


TEST_DIR = os.path.dirname(__file__)


def run_ms2pip():
    """Run ms2pipC to predict peak intensities from a PEPREC file (HCD model). """
    params = {
        "ms2pip": {
            "ptm": [
                "Oxidation,15.994915,opt,M",
                "Carbamidomethyl,57.021464,opt,C",
                "Acetyl,42.010565,opt,N-term",
            ],
            "sptm": [],
            "gptm": [],
            "frag_method": "HCD2019",
            "frag_error": 0.02,
            "out": "csv",
        }
    }
    ms2pip = MS2PIP(os.path.join(TEST_DIR, "test_data/test.peprec"), params=params)
    ms2pip.run()

    test_data = pd.read_csv(
        os.path.join(TEST_DIR, "test_data/test_HCD2019_predictions.csv")
    )
    target_data = pd.read_csv(
        os.path.join(TEST_DIR, "test_data/target_HCD2019_predictions.csv")
    )
    pepfile = pd.read_csv(
        os.path.join(TEST_DIR, "test_data/test.peprec"),
        sep=" ",
        index_col=False,
        dtype={"spec_id": str, "modifications": str},
    )
    return test_data, target_data, pepfile


TEST_DATA, TARGET_DATA, PEPFILE = run_ms2pip()


class TestPredictions:
    def test_all_spec(self):
        assert set(TEST_DATA.spec_id.unique()) == set(PEPFILE.spec_id)

    def test_amount_peaks(self):
        for pep in ["peptide1", "peptide2", "peptide3"]:
            peplen = len(PEPFILE[PEPFILE.spec_id == pep].peptide.values[0])
            assert len(TEST_DATA[TEST_DATA.spec_id == pep]) == (2 * peplen) - 2

    def test_peak_ints_b(self):
        for pep in TARGET_DATA.spec_id.unique():
            tmp_test = TEST_DATA[TEST_DATA.spec_id == pep]
            tmp_test = tmp_test[tmp_test.ion == "b"]
            tmp_target = TARGET_DATA[TARGET_DATA.spec_id == pep]
            tmp_target = tmp_target[tmp_target.ion == "b"]
            for no in tmp_target.ionnumber:
                assert (
                    tmp_test[tmp_test.ionnumber == no]["prediction"].values[0]
                    == tmp_target[tmp_target.ionnumber == no]["prediction"].values[0]
                )

    def test_peak_ints_y(self):
        for pep in TARGET_DATA.spec_id.unique():
            tmp_test = TEST_DATA[TEST_DATA.spec_id == pep]
            tmp_test = tmp_test[tmp_test.ion == "y"]
            tmp_target = TARGET_DATA[TARGET_DATA.spec_id == pep]
            tmp_target = tmp_target[tmp_target.ion == "y"]
            for no in tmp_target.ionnumber:
                assert (
                    tmp_test[tmp_test.ionnumber == no]["prediction"].values[0]
                    == tmp_target[tmp_target.ionnumber == no]["prediction"].values[0]
                )
