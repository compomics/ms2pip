import numpy as np
from psm_utils import PSM, Peptidoform
import pandas as pd

from ms2pip.core import get_training_data, predict_single
from ms2pip.result import ProcessingResult


def _test_get_training_data():
    expected_df = pd.read_feather("tests/test_data/massivekb_selected_500.feather")
    output_df = get_training_data(
        "tests/test_data/massivekb_selected_500.peprec",
        "tests/test_data/massivekb_selected_500.mgf",
        model="HCD",
        ms2_tolerance=0.02,
        processes=1
    )
    pd.testing.assert_frame_equal(expected_df, output_df)

def test_predict_single():
    pep = Peptidoform("ACDE/2")
    result = predict_single(pep)

    expected = ProcessingResult(
        psm_index=0,
        psm=PSM(peptidoform=pep, spectrum_id=0),
        theoretical_mz={
            "b": np.array([72.04435, 175.05354, 290.08047], dtype=np.float32),
            "y": np.array([148.0604, 263.0873, 366.0965], dtype=np.float32),
        },
        predicted_intensity={
            "b": np.array([-9.14031, -7.6102686, -7.746709], dtype=np.float32),
            "y": np.array([-5.8988147, -5.811797, -7.069088], dtype=np.float32),
        },
        observed_intensity=None,
        correlation=None,
        feature_vectors=None,
    )

    assert result.psm_index == expected.psm_index
    assert result.psm == expected.psm
    np.testing.assert_array_almost_equal(result.theoretical_mz["b"], expected.theoretical_mz["b"])
    np.testing.assert_array_almost_equal(result.theoretical_mz["y"], expected.theoretical_mz["y"])
    np.testing.assert_array_almost_equal(result.predicted_intensity["b"], expected.predicted_intensity["b"])
    np.testing.assert_array_almost_equal(result.predicted_intensity["y"], expected.predicted_intensity["y"])
    assert result.observed_intensity == expected.observed_intensity
    assert result.correlation == expected.correlation
    assert result.feature_vectors == expected.feature_vectors
