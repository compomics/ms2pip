import numpy as np
from psm_utils import PSM, PSMList, Peptidoform

from ms2pip.core import predict_batch, predict_library, predict_single
from ms2pip.result import ProcessingResult


def test_predict_single():
    pep = Peptidoform("ACDE/2")
    result = predict_single(pep)

    expected = ProcessingResult(
        psm_index=0,
        psm=PSM(peptidoform=pep, spectrum_id=0),
        theoretical_mz={
            "b": np.array([72.04439, 175.05357, 290.0805], dtype=np.float32),
            "y": np.array([148.06044, 263.08737, 366.09656], dtype=np.float32),
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
    np.testing.assert_array_almost_equal(
        result.predicted_intensity["b"], expected.predicted_intensity["b"]
    )
    np.testing.assert_array_almost_equal(
        result.predicted_intensity["y"], expected.predicted_intensity["y"]
    )
    assert result.observed_intensity == expected.observed_intensity
    assert result.correlation == expected.correlation
    assert result.feature_vectors == expected.feature_vectors


def test_predict_single_modified():
    result = predict_single("AC[+57.0215]M[+15.9949]DEK/2")

    assert result.predicted_intensity is not None
    assert result.theoretical_mz is not None
    assert set(result.theoretical_mz.keys()) == {"b", "y"}
    # 6 residues → 5 cleavage sites
    assert len(result.theoretical_mz["b"]) == 5
    assert len(result.theoretical_mz["y"]) == 5
    assert len(result.predicted_intensity["b"]) == 5
    # m/z values should be positive and increasing for b-ions
    assert all(result.theoretical_mz["b"] > 0)
    assert all(np.diff(result.theoretical_mz["b"]) > 0)


def test_predict_batch():
    psm_list = PSMList(
        psm_list=[
            PSM(peptidoform=Peptidoform("ACDE/2"), spectrum_id=0),
            PSM(peptidoform=Peptidoform("PEPTIDEK/3"), spectrum_id=1),
            PSM(peptidoform=Peptidoform("AAAAAAA/2"), spectrum_id=2),
        ]
    )
    results = predict_batch(psm_list)

    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.psm_index == i
        assert result.predicted_intensity is not None
        assert result.theoretical_mz is not None
        assert set(result.theoretical_mz.keys()) == {"b", "y"}
        assert result.feature_vectors is None

    # Check correct number of ions per peptide
    assert len(results[0].theoretical_mz["b"]) == 3  # ACDE: 4 residues → 3
    assert len(results[1].theoretical_mz["b"]) == 7  # PEPTIDEK: 8 residues → 7
    assert len(results[2].theoretical_mz["b"]) == 6  # AAAAAAA: 7 residues → 6


def test_predict_library():
    batches = list(
        predict_library(fasta_file="tests/test_data/test.fasta", batch_size=100)
    )

    assert len(batches) >= 1
    for batch in batches:
        assert isinstance(batch, list)
        for result in batch:
            assert isinstance(result, ProcessingResult)
            assert result.predicted_intensity is not None
            assert result.theoretical_mz is not None
