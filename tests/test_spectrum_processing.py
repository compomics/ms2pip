import numpy as np
from ms2rescore_rs import AnnotatedMS2Spectrum, FragmentAnnotation, Precursor
from psm_utils import PSM, Peptidoform, PSMList

from ms2pip._spectrum_processing import (
    _preloaded_to_annotations,
    annotate_spectrum,
    proforma_to_mass_shift,
)
from ms2rescore_rs import ms2pip_extract_targets


def _make_annotated_spectrum(peak_annotations, intensity, seq_len=4):
    """Helper to build an AnnotatedMS2Spectrum for testing ms2pip_extract_targets."""
    mz = list(range(len(intensity)))  # dummy m/z values
    return AnnotatedMS2Spectrum(
        identifier="test",
        mz=mz,
        intensity=list(intensity),
        precursor=Precursor(mz=0.0, charge=2, rt=0.0),
        peak_annotations=peak_annotations,
    )


def test_extract_targets_basic():
    """Test basic target extraction from annotations."""
    # 4-residue peptide -> 3 cleavage sites
    peak_annotations = [
        [FragmentAnnotation(series="b", position=1, charge=1)],
        [],
        [FragmentAnnotation(series="y", position=2, charge=1)],
    ]
    intensity = np.array([5.0, 3.0, 8.0], dtype=np.float32)
    annotated = _make_annotated_spectrum(peak_annotations, intensity)

    targets = ms2pip_extract_targets(
        annotated_spectra=[annotated],
        intensities=[intensity],
        ion_types=["b", "y"],
        seq_lens=[4],
    )[0]

    floor = np.float32(np.log2(0.001))
    assert targets["b"][0] == 5.0  # b1 at position 0
    assert targets["b"][1] == floor  # b2 unmatched
    assert targets["b"][2] == floor  # b3 unmatched
    assert targets["y"][0] == floor  # y1 unmatched
    assert targets["y"][1] == 8.0  # y2 at position 1
    assert targets["y"][2] == floor  # y3 unmatched


def test_extract_targets_charge2():
    """Test that charge-2 annotations map to b2/y2 ion types."""
    peak_annotations = [
        [FragmentAnnotation(series="b", position=1, charge=2)],
    ]
    intensity = np.array([10.0], dtype=np.float32)
    annotated = _make_annotated_spectrum(peak_annotations, intensity)

    targets = ms2pip_extract_targets(
        annotated_spectra=[annotated],
        intensities=[intensity],
        ion_types=["b", "y", "b2", "y2"],
        seq_lens=[4],
    )[0]

    assert targets["b2"][0] == 10.0
    floor = np.float32(np.log2(0.001))
    assert targets["b"][0] == floor  # charge-1 b is not matched


def test_extract_targets_highest_intensity():
    """Test that the highest intensity is kept when multiple peaks match."""
    peak_annotations = [
        [FragmentAnnotation(series="b", position=1, charge=1)],
        [FragmentAnnotation(series="b", position=1, charge=1)],
    ]
    intensity = np.array([3.0, 7.0], dtype=np.float32)
    annotated = _make_annotated_spectrum(peak_annotations, intensity)

    targets = ms2pip_extract_targets(
        annotated_spectra=[annotated],
        intensities=[intensity],
        ion_types=["b", "y"],
        seq_lens=[4],
    )[0]

    assert targets["b"][0] == 7.0  # highest wins


def test_extract_targets_ignores_unknown_ion_types():
    """Test that annotations for ion types not in the target list are ignored."""
    peak_annotations = [
        [FragmentAnnotation(series="c", position=1, charge=1)],
    ]
    intensity = np.array([10.0], dtype=np.float32)
    annotated = _make_annotated_spectrum(peak_annotations, intensity)

    targets = ms2pip_extract_targets(
        annotated_spectra=[annotated],
        intensities=[intensity],
        ion_types=["b", "y"],
        seq_lens=[4],
    )[0]

    floor = np.float32(np.log2(0.001))
    assert all(v == floor for v in targets["b"])
    assert all(v == floor for v in targets["y"])


def _make_preloaded_annotated(n_peaks: int, identifier: str) -> AnnotatedMS2Spectrum:
    """AnnotatedMS2Spectrum with ``n_peaks`` peaks, one b1 annotation, rest unmatched."""
    peak_annotations = [[FragmentAnnotation(series="b", position=1, charge=1)]]
    peak_annotations += [[] for _ in range(n_peaks - 1)]
    return AnnotatedMS2Spectrum(
        identifier=identifier,
        mz=[100.0 + i for i in range(n_peaks)],
        intensity=[float(n_peaks - i) for i in range(n_peaks)],
        precursor=Precursor(mz=250.0, charge=2, rt=100.0),
        peak_annotations=peak_annotations,
    )


def test_preloaded_multirun_shared_spectrum_id_stays_aligned():
    """
    Two runs can share a spectrum_id (e.g. ``scan=1``) while pointing at different
    spectra with different peak counts. The observed intensities and the annotated
    spectrum of each MatchedSpectrum must come from the same run, so their lengths
    stay aligned.
    """
    psm_a = PSM(peptidoform=Peptidoform("PEPTIDE/2"), spectrum_id="scan=1", run="runA")
    psm_a.spectrum = _make_preloaded_annotated(3, "scan=1")
    psm_b = PSM(peptidoform=Peptidoform("PEPTIDE/2"), spectrum_id="scan=1", run="runB")
    psm_b.spectrum = _make_preloaded_annotated(5, "scan=1")
    psm_list = PSMList(psm_list=[psm_a, psm_b])

    matched = _preloaded_to_annotations(psm_list, "HCD", 0.02, "Da")

    assert len(matched) == 2
    for m in matched:
        assert len(m.spectrum.intensity) == len(m.annotated_spectrum.mz)
    # Each PSM keeps its own run's peak count
    by_run = {m.psm.run: len(m.annotated_spectrum.mz) for m in matched}
    assert by_run == {"runA": 3, "runB": 5}


def test_preloaded_single_run_deduplicates_shared_spectrum_id():
    """Within one run, two PSMs sharing a spectrum_id reuse one observed spectrum."""
    psm_1 = PSM(peptidoform=Peptidoform("PEPTIDE/2"), spectrum_id="scan=1", run="runA")
    psm_1.spectrum = _make_preloaded_annotated(4, "scan=1")
    psm_2 = PSM(peptidoform=Peptidoform("PEPTIDEK/3"), spectrum_id="scan=1", run="runA")
    psm_2.spectrum = _make_preloaded_annotated(4, "scan=1")
    psm_list = PSMList(psm_list=[psm_1, psm_2])

    matched = _preloaded_to_annotations(psm_list, "HCD", 0.02, "Da")

    assert len(matched) == 2
    for m in matched:
        assert len(m.spectrum.intensity) == len(m.annotated_spectrum.mz)


def test_annotate_spectrum():
    """Test that annotate_spectrum returns an AnnotatedMS2Spectrum."""
    spectrum = __import__("ms2pip.spectrum", fromlist=["ObservedSpectrum"]).ObservedSpectrum(
        mz=np.array([72.044, 175.054, 290.081], dtype=np.float32),
        intensity=np.array([100.0, 200.0, 300.0], dtype=np.float32),
        identifier="test",
        precursor_mz=250.0,
        precursor_charge=2,
        retention_time=100.0,
    )
    psm = PSM(peptidoform=Peptidoform("ACDE/2"), spectrum_id="test")

    result = annotate_spectrum(spectrum, psm, "HCD", 0.02, "Da")

    assert isinstance(result, AnnotatedMS2Spectrum)
    # Should have one annotation list per peak
    assert len(result.peak_annotations) == 3
    # At least some peaks should be annotated
    annotated_peaks = [i for i, anns in enumerate(result.peak_annotations) if len(anns) > 0]
    assert len(annotated_peaks) > 0


def test_proforma_to_mass_shift_unmodified():
    result = proforma_to_mass_shift(Peptidoform("PEPTIDE/2"))
    assert result == "PEPTIDE/2"


def test_proforma_to_mass_shift_unimod_names():
    result = proforma_to_mass_shift(Peptidoform("PEPTC[UNIMOD:Carbamidomethyl]M[UNIMOD:Oxidation]IDE/2"))
    assert "[+" in result
    assert "Carbamidomethyl" not in result
    assert "Oxidation" not in result
    assert result.startswith("PEPTC[+57.0215]M[+15.9949]")
    assert result.endswith("/2")


def test_proforma_to_mass_shift_nterm():
    result = proforma_to_mass_shift(Peptidoform("[UNIMOD:Acetyl]-PEPTIDE/2"))
    assert result.startswith("[+42.0106]-PEPTIDE")
    assert result.endswith("/2")


def test_proforma_to_mass_shift_already_mass_shift():
    result = proforma_to_mass_shift(Peptidoform("PEPTC[+57.0215]IDE/2"))
    assert result == "PEPTC[+57.0215]IDE/2"


def test_proforma_to_mass_shift_no_charge():
    result = proforma_to_mass_shift(Peptidoform("PEPTIDE"))
    assert result == "PEPTIDE"
    assert "/" not in result
