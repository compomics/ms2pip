from subprocess import call
import pandas as pd
import numpy as np


def assert_get_feature_vectors(test_data, target_data):
    assert test_data[test_data.columns[:-3]].equals(
        target_data[target_data.columns[:-3]]
    )


def assert_get_targetsB(test_data, target_data):
    for i in range(3):
        assert np.isclose(test_data["targets_B"][i], target_data["targets_B"][i])


def assert_get_targetsY(test_data, target_data):
    for i in range(3):
        assert np.isclose(test_data["targets_Y"][i], target_data["targets_Y"][i])


def assert_get_psmid(test_data, target_data):
    assert test_data["psmid"].equals(target_data["psmid"])


def test_dummy_spectrum():
    # Run ms2pipC to extract features and targets from an .mgf and .PEPREC files
    call(
        [
            "ms2pip",
            "-s",
            "tests/test_data/hard_test.mgf",
            "-w",
            "tests/test_data/test.h5",
            "-c",
            "tests/test_data/config.txt",
            "tests/test_data/test.peprec",
        ]
    )

    # Load target values
    test_data = pd.read_hdf("tests/test_data/test.h5", "table")
    target_data = pd.read_hdf("tests/test_data/hard_test_targetvectors.h5", "table")

    # Test
    assert_get_feature_vectors(test_data, target_data)
    assert_get_targetsB(test_data, target_data)
    assert_get_targetsY(test_data, target_data)
    assert_get_psmid(test_data, target_data)

    call(["rm", "tests/test_data/test.h5"])


def test_real_spectra():
    # Run ms2pipC to extract features and targets from an .mgf and .PEPREC files
    call(
        [
            "ms2pip",
            "-c",
            "tests/test_data/massivekb_selected_500_config.txt",
            "-s",
            "tests/test_data/massivekb_selected_500.mgf",
            "-w",
            "tests/test_data/massivekb_selected_500_test.h5",
            "-m1",
            "tests/test_data/massivekb_selected_500.peprec",
        ]
    )

    # Load target values
    test_data = pd.read_hdf("tests/test_data/massivekb_selected_500_test.h5", "table")
    target_data = pd.read_hdf(
        "tests/test_data/massivekb_selected_500_targetvectors.h5", "table"
    )

    # Test
    assert_get_feature_vectors(test_data, target_data)
    assert_get_targetsB(test_data, target_data)
    assert_get_targetsY(test_data, target_data)
    assert_get_psmid(test_data, target_data)

    call(["rm", "tests/test_data/massivekb_selected_500_test.h5"])
