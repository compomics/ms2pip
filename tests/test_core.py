import pandas as pd

from ms2pip.core import get_training_data


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
