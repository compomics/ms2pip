import os
import re

import numpy as np
import pandas as pd

from ms2pip.ms2pip_tools.spectrum_output import SpectrumOutput


TEST_DIR = os.path.dirname(__file__)


class TestSpectrumOutput:
    def test_integration(self):
        def compare_line(test_line, target_line):
            """Assert if two lines in spectrum output are the same."""

            # Extract float values from line and use assert_allclose, to allow for
            # float imprecisions
            float_pattern = re.compile(r"[0-9]*[.][0-9]+")
            test_floats = float_pattern.findall(test_line)
            target_floats = float_pattern.findall(target_line)
            assert len(test_floats) == len(target_floats)
            [
                np.testing.assert_allclose(float(te), float(ta), rtol=1e-5)
                for te, ta in zip(test_floats, target_floats)
            ]
            assert float_pattern.sub(test_line, "") == float_pattern.sub(
                target_line, ""
            )

        peprec = pd.read_pickle(
            os.path.join(TEST_DIR, "test_data/spectrum_output/input_peprec.pkl")
        )
        all_preds = pd.read_pickle(
            os.path.join(TEST_DIR, "test_data/spectrum_output/input_preds.pkl")
        )

        params = {
            "ptm": [
                "Oxidation,15.994915,opt,M",
                "Carbamidomethyl,57.021464,opt,C",
                "Glu->pyro-Glu,-18.010565,opt,E",
                "Gln->pyro-Glu,-17.026549,opt,Q",
                "Acetyl,42.010565,opt,N-term",
            ],
            "sptm": [],
            "gptm": [],
            "model": "HCD",
            "frag_error": "0.02",
            "out": "csv",
        }

        peprec_tmp = peprec.sample(5, random_state=10).copy()
        all_preds_tmp = all_preds[
            all_preds["spec_id"].isin(peprec_tmp["spec_id"])
        ].copy()

        so = SpectrumOutput(
            all_preds_tmp,
            peprec_tmp,
            params,
            output_filename="test",
            return_stringbuffer=True,
        )

        target_filename_base = os.path.join(
            TEST_DIR, "test_data/spectrum_output/target"
        )

        # Test general output
        test_cases = [
            (so.write_mgf, "_predictions.mgf"),
            (so.write_msp, "_predictions.msp"),
            (so.write_spectronaut, "_predictions_spectronaut.csv"),
        ]

        for test_function, file_ext in test_cases:
            test = test_function()
            test.seek(0)
            with open(target_filename_base + file_ext) as target:
                for test_line, target_line in zip(test.readlines(), target.readlines()):
                    compare_line(test_line, target_line)

        # Test bibliospec output
        bibliospec_ssl, bibliospec_ms2 = so.write_bibliospec()
        test_cases = [
            (bibliospec_ssl, "_predictions.ssl"),
            (bibliospec_ms2, "_predictions.ms2"),
        ]

        for test, file_ext in test_cases:
            test.seek(0)
            with open(target_filename_base + file_ext) as target:
                for test_line, target_line in zip(test.readlines(), target.readlines()):
                    test_line = test_line.replace(
                        "test_predictions.ms2", "target_predictions.ms2"
                    )
                    if not "CreationDate" in target_line:
                        compare_line(test_line, target_line)
