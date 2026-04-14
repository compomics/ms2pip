import numpy as np
import pytest
from psm_utils import PSM, Peptidoform
from ms2rescore_rs import AnnotatedMS2Spectrum, FragmentAnnotation, MS2Spectrum, Precursor

from ms2pip.core import correlate, correlate_single
from ms2pip.spectrum import ObservedSpectrum


def _make_observed_spectrum(peptidoform="ACDEFK/2"):
    """Create an ObservedSpectrum with realistic peaks for testing."""
    return ObservedSpectrum(
        mz=np.array(
            [72.044, 147.113, 175.054, 276.167, 290.081, 389.251, 488.319, 566.192],
            dtype=np.float32,
        ),
        intensity=np.array(
            [50.0, 100.0, 80.0, 200.0, 60.0, 150.0, 300.0, 40.0],
            dtype=np.float32,
        ),
        identifier="test_spectrum",
        peptidoform=peptidoform,
        precursor_mz=341.65,
        precursor_charge=2,
        retention_time=100.0,
    )


def test_correlate_single():
    obs = _make_observed_spectrum()
    result = correlate_single(obs, ms2_tolerance=0.02, model="HCD")

    assert result.predicted_intensity is not None
    assert result.observed_intensity is not None
    assert result.correlation is not None
    assert not np.isnan(result.correlation)
    assert set(result.predicted_intensity.keys()) == {"b", "y"}
    assert set(result.observed_intensity.keys()) == {"b", "y"}


def test_correlate_single_ppm_tolerance():
    obs = _make_observed_spectrum()
    result = correlate_single(obs, ms2_tolerance=20.0, ms2_tolerance_mode="ppm", model="HCD")

    assert result.predicted_intensity is not None
    assert result.observed_intensity is not None
    assert result.correlation is not None


def test_correlate_single_requires_peptidoform():
    obs = ObservedSpectrum(
        mz=np.array([100.0], dtype=np.float32),
        intensity=np.array([1.0], dtype=np.float32),
        identifier="test",
    )
    with pytest.raises(ValueError, match="Peptidoform must be set"):
        correlate_single(obs)


def test_correlate_annotated():
    ann_spec = AnnotatedMS2Spectrum(
        identifier="test_spectrum",
        mz=[175.054, 276.167, 488.319],
        intensity=[100.0, 200.0, 300.0],
        precursor=Precursor(mz=341.65, charge=2, rt=100.0),
        peak_annotations=[
            [FragmentAnnotation(series="b", position=2, charge=1)],
            [FragmentAnnotation(series="y", position=4, charge=1)],
            [],
        ],
    )
    psm = PSM(
        peptidoform=Peptidoform("ACDEFK/2"),
        spectrum_id="test_spectrum",
        spectrum=ann_spec,
    )

    results = correlate([psm], model="HCD")

    assert len(results) == 1
    result = results[0]
    assert result.predicted_intensity is not None
    assert result.observed_intensity is not None
    assert set(result.observed_intensity.keys()) == {"b", "y"}
    # Annotated b2 position should have non-floor intensity
    floor = np.log2(0.001)
    assert result.observed_intensity["b"][1] > floor


def test_correlate_raw():
    raw_spec = MS2Spectrum(
        identifier="test_spectrum",
        mz=[175.054, 276.167, 488.319],
        intensity=[100.0, 200.0, 300.0],
        precursor=Precursor(mz=341.65, charge=2, rt=100.0),
    )
    psm = PSM(
        peptidoform=Peptidoform("ACDEFK/2"),
        spectrum_id="test_spectrum",
        spectrum=raw_spec,
    )

    results = correlate(
        [psm], model="HCD", ms2_tolerance=0.02, ms2_tolerance_mode="Da"
    )

    assert len(results) == 1
    assert results[0].predicted_intensity is not None
    assert results[0].observed_intensity is not None


def test_correlate_ppm_tolerance():
    raw_spec = MS2Spectrum(
        identifier="test_spectrum",
        mz=[175.054, 276.167],
        intensity=[100.0, 200.0],
        precursor=Precursor(mz=341.65, charge=2, rt=100.0),
    )
    psm = PSM(
        peptidoform=Peptidoform("ACDEFK/2"),
        spectrum_id="test_spectrum",
        spectrum=raw_spec,
    )

    results = correlate(
        [psm], model="HCD", ms2_tolerance=20.0, ms2_tolerance_mode="ppm"
    )

    assert len(results) == 1
    assert results[0].observed_intensity is not None


def test_correlate_no_spectra_no_file():
    psm = PSM(
        peptidoform=Peptidoform("ACDEFK/2"),
        spectrum_id="test",
    )
    with pytest.raises(ValueError, match="spectrum_file.*must be provided"):
        correlate([psm])


def test_correlate_multiple_psms():
    psms = []
    for i, pep in enumerate(["ACDEFK/2", "PEPTIDEK/3"]):
        spec = MS2Spectrum(
            identifier=f"spec_{i}",
            mz=[175.054, 276.167],
            intensity=[100.0, 200.0],
            precursor=Precursor(mz=400.0, charge=2, rt=float(i * 10)),
        )
        psms.append(PSM(peptidoform=Peptidoform(pep), spectrum_id=f"spec_{i}", spectrum=spec))

    results = correlate(psms, model="HCD")

    assert len(results) == 2
    for result in results:
        assert result.predicted_intensity is not None
        assert result.observed_intensity is not None
