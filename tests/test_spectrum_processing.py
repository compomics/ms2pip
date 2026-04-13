import numpy as np
from psm_utils import PSM, Peptidoform

from ms2pip._spectrum_processing import annotate_spectrum, targets_from_annotations


def test_targets_from_annotations_basic():
    """Test basic target extraction from annotations."""
    # 4-residue peptide → 3 cleavage sites
    peak_annotations = [
        [("b", 1, 1)],  # peak 0 matches b1
        [],  # peak 1 unmatched
        [("y", 2, 1)],  # peak 2 matches y2
    ]
    intensity = np.array([5.0, 3.0, 8.0], dtype=np.float32)

    targets = targets_from_annotations(peak_annotations, intensity, ["b", "y"], seq_len=4)

    floor = np.float32(np.log2(0.001))
    assert targets["b"][0] == 5.0  # b1 at position 0
    assert targets["b"][1] == floor  # b2 unmatched
    assert targets["b"][2] == floor  # b3 unmatched
    assert targets["y"][0] == floor  # y1 unmatched
    assert targets["y"][1] == 8.0  # y2 at position 1
    assert targets["y"][2] == floor  # y3 unmatched


def test_targets_from_annotations_charge2():
    """Test that charge-2 annotations map to b2/y2 ion types."""
    peak_annotations = [
        [("b", 1, 2)],  # charge 2 → maps to "b2"
    ]
    intensity = np.array([10.0], dtype=np.float32)

    targets = targets_from_annotations(
        peak_annotations, intensity, ["b", "y", "b2", "y2"], seq_len=4
    )

    assert targets["b2"][0] == 10.0
    floor = np.float32(np.log2(0.001))
    assert targets["b"][0] == floor  # charge-1 b is not matched


def test_targets_from_annotations_highest_intensity():
    """Test that the highest intensity is kept when multiple peaks match."""
    peak_annotations = [
        [("b", 1, 1)],  # peak 0: b1 with intensity 3.0
        [("b", 1, 1)],  # peak 1: b1 with intensity 7.0
    ]
    intensity = np.array([3.0, 7.0], dtype=np.float32)

    targets = targets_from_annotations(peak_annotations, intensity, ["b", "y"], seq_len=4)

    assert targets["b"][0] == 7.0  # highest wins


def test_targets_from_annotations_ignores_unknown_ion_types():
    """Test that annotations for ion types not in the target list are ignored."""
    peak_annotations = [
        [("c", 1, 1)],  # c-ion not in requested types
    ]
    intensity = np.array([10.0], dtype=np.float32)

    targets = targets_from_annotations(peak_annotations, intensity, ["b", "y"], seq_len=4)

    floor = np.float32(np.log2(0.001))
    assert all(v == floor for v in targets["b"])
    assert all(v == floor for v in targets["y"])


def test_targets_from_annotations_fragment_annotation_objects():
    """Test that FragmentAnnotation objects (not tuples) also work."""
    from ms2rescore_rs import FragmentAnnotation

    peak_annotations = [
        [FragmentAnnotation(series="b", position=1, charge=1)],
    ]
    intensity = np.array([5.0], dtype=np.float32)

    targets = targets_from_annotations(peak_annotations, intensity, ["b", "y"], seq_len=4)

    assert targets["b"][0] == 5.0


def test_annotate_spectrum():
    """Test that annotate_spectrum returns annotations for matching peaks."""
    spectrum = __import__("ms2pip.spectrum", fromlist=["ObservedSpectrum"]).ObservedSpectrum(
        mz=np.array([72.044, 175.054, 290.081], dtype=np.float32),
        intensity=np.array([100.0, 200.0, 300.0], dtype=np.float32),
        identifier="test",
        precursor_mz=250.0,
        precursor_charge=2,
        retention_time=100.0,
    )
    psm = PSM(peptidoform=Peptidoform("ACDE/2"), spectrum_id="test")

    annotations = annotate_spectrum(spectrum, psm, "HCD", 0.02, "Da")

    # Should return one list of annotations per peak
    assert len(annotations) == 3
    # At least some peaks should be annotated (b1 at ~72.044, b2 at ~175.054)
    annotated_peaks = [i for i, anns in enumerate(annotations) if len(anns) > 0]
    assert len(annotated_peaks) > 0
