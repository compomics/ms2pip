import os

import pandas as pd
import numpy as np

from ms2pip.ms2pipC import MS2PIP


TEST_DIR = os.path.dirname(__file__)


class TestFeatureExtraction:
    def _assert_get_feature_vectors(self, test_data, target_data):
        assert test_data[test_data.columns[:-3]].equals(
            target_data[target_data.columns[:-3]]
        )

    def _assert_get_targetsB(self, test_data, target_data):
        for i in range(3):
            assert np.isclose(test_data["targets_B"][i], target_data["targets_B"][i])

    def _assert_get_targetsY(self, test_data, target_data):
        for i in range(3):
            assert np.isclose(test_data["targets_Y"][i], target_data["targets_Y"][i])

    def _assert_get_psmid(self, test_data, target_data):
        assert test_data["psmid"].equals(target_data["psmid"])

    def test_dummy_spectrum(self):
        # Run ms2pipC to extract features and targets from an .mgf and .PEPREC files
        params = {
            "ms2pip": {
                "ptm": [],
                "sptm": [],
                "gptm": [],
                "frag_method": "HCD2019",
                "frag_error": 0.02,
                "out": "csv",
            }
        }
        ms2pip = MS2PIP(
            os.path.join(TEST_DIR, "test_data/test.peprec"),
            spec_file=os.path.join(TEST_DIR, "test_data/hard_test.mgf"),
            vector_file=os.path.join(TEST_DIR, "test_data/test.h5"),
            params=params,
        )
        ms2pip.run()

        # Load target values
        test_data = pd.read_hdf(os.path.join(TEST_DIR, "test_data/test.h5"), "table")
        target_data = pd.read_hdf(
            os.path.join(TEST_DIR, "test_data/hard_test_targetvectors.h5"), "table"
        )

        # Test
        self._assert_get_feature_vectors(test_data, target_data)
        self._assert_get_targetsB(test_data, target_data)
        self._assert_get_targetsY(test_data, target_data)
        self._assert_get_psmid(test_data, target_data)

        os.remove(os.path.join(TEST_DIR, "test_data/test.h5"))

    def test_real_spectra(self):
        # Run ms2pipC to extract features and targets from an .mgf and .PEPREC files
        params = {
            "ms2pip": {
                "ptm": [
                    "Oxidation,15.994915,opt,M",
                    "Carbamidomethyl,57.021464,opt,C",
                    "Pyro_glu,-18.010565,opt,E",
                    "Deamidation,0.984016,opt,N",
                    "Acetyl,42.010565,opt,N-term",
                    "Carbamyl,43.005814,opt,N-term",
                ],
                "sptm": [],
                "gptm": [],
                "frag_method": "HCD2019",
                "frag_error": 0.02,
                "out": "csv",
            }
        }
        ms2pip = MS2PIP(
            os.path.join(TEST_DIR, "test_data/massivekb_selected_500.peprec"),
            spec_file=os.path.join(TEST_DIR, "test_data/massivekb_selected_500.mgf"),
            vector_file=os.path.join(
                TEST_DIR, "test_data/massivekb_selected_500_test.h5"
            ),
            params=params,
            num_cpu=1,
        )
        ms2pip.run()

        # Load target values
        test_data = pd.read_hdf(
            os.path.join(TEST_DIR, "test_data/massivekb_selected_500_test.h5"), "table"
        )
        target_data = pd.read_hdf(
            os.path.join(TEST_DIR, "test_data/massivekb_selected_500_targetvectors.h5"),
            "table",
        )

        # Test
        self._assert_get_feature_vectors(test_data, target_data)
        self._assert_get_targetsB(test_data, target_data)
        self._assert_get_targetsY(test_data, target_data)
        self._assert_get_psmid(test_data, target_data)

        os.remove(os.path.join(TEST_DIR, "test_data/massivekb_selected_500_test.h5"))
