"""Read, annotate, and match MS2 spectra."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Union

import numpy as np
from psm_utils import PSM, PSMList
from ms2rescore_rs import (
    MS2Spectrum,
    Precursor,
    annotate_ms2_spectra,
    get_ms2_spectra,
)

import ms2pip.exceptions as exceptions
from ms2pip.constants import MODELS
from ms2pip.spectrum import ObservedSpectrum


def read_spectrum_file(spectrum_file: str) -> Generator[ObservedSpectrum, None, None]:
    """
    Read MS2 spectra from a supported file format; inferring the type from the filename extension.

    Parameters
    ----------
    spectrum_file
        Path to MGF or mzML file.

    Yields
    ------
    ObservedSpectrum

    Raises
    ------
    UnsupportedSpectrumFiletypeError
        If the file extension is not supported.

    """
    try:
        spectra = get_ms2_spectra(str(spectrum_file))
    except ValueError as e:
        raise exceptions.UnsupportedSpectrumFiletypeError(Path(spectrum_file).suffixes) from e

    for spectrum in spectra:
        obs_spectrum = ObservedSpectrum(
            mz=np.array(spectrum.mz, dtype=np.float32),
            intensity=np.array(spectrum.intensity, dtype=np.float32),
            identifier=str(spectrum.identifier),
            precursor_mz=float(spectrum.precursor.mz),
            precursor_charge=int(spectrum.precursor.charge),
            retention_time=float(spectrum.precursor.rt),
        )
        # Workaround for mobiusklein/mzdata#3
        if (
            obs_spectrum.identifier == ""
            or obs_spectrum.mz.shape[0] == 0
            or obs_spectrum.intensity.shape[0] == 0
        ):
            continue
        yield obs_spectrum


def _read_raw_spectra(spectrum_file: str) -> Generator[MS2Spectrum, None, None]:
    """Read MS2 spectra as raw ms2rescore-rs objects (no conversion to ObservedSpectrum)."""
    try:
        spectra = get_ms2_spectra(str(spectrum_file))
    except ValueError as e:
        raise exceptions.UnsupportedSpectrumFiletypeError(Path(spectrum_file).suffixes) from e

    for spectrum in spectra:
        if (
            str(spectrum.identifier) == ""
            or len(spectrum.mz) == 0
            or len(spectrum.intensity) == 0
        ):
            continue
        yield spectrum


def _to_observed_spectrum(spectrum: MS2Spectrum) -> ObservedSpectrum:
    """Convert an MS2Spectrum to an ObservedSpectrum."""
    return ObservedSpectrum(
        mz=np.array(spectrum.mz, dtype=np.float32),
        intensity=np.array(spectrum.intensity, dtype=np.float32),
        identifier=str(spectrum.identifier),
        precursor_mz=float(spectrum.precursor.mz),
        precursor_charge=int(spectrum.precursor.charge),
        retention_time=float(spectrum.precursor.rt),
    )


def annotate_spectrum(
    spectrum: ObservedSpectrum,
    psm: PSM,
    model: str,
    ms2_tolerance: float,
    ms2_tolerance_mode: str,
) -> List[List[tuple]]:
    """
    Annotate an ObservedSpectrum using ms2rescore-rs.

    Returns peak annotations as plain Python lists of ``(series, position, charge)`` tuples.
    """
    ms2_spectrum = MS2Spectrum(
        identifier=spectrum.identifier or "",
        mz=list(spectrum.mz),
        intensity=list(spectrum.intensity),
        precursor=Precursor(
            mz=float(spectrum.precursor_mz) if spectrum.precursor_mz else 0.0,
            charge=int(spectrum.precursor_charge) if spectrum.precursor_charge else 0,
            rt=float(spectrum.retention_time) if spectrum.retention_time else 0.0,
        ),
    )
    frag_model = MODELS[model]["fragmentation"]
    proforma = str(psm.peptidoform.proforma)
    seq_len = len(psm.peptidoform.parsed_sequence)

    annotated = annotate_ms2_spectra(
        spectra=[ms2_spectrum],
        proformas=[proforma],
        seq_lens=[seq_len],
        fragmentation_model=frag_model,
        mass_mode="monoisotopic",
        tolerance_value=float(ms2_tolerance),
        tolerance_mode=ms2_tolerance_mode.lower(),
    )
    return [
        [(a.series, a.position, a.charge) for a in peak_anns]
        for peak_anns in annotated[0].peak_annotations
    ]


def targets_from_annotations(
    peak_annotations: List,
    intensity: np.ndarray,
    ion_types: List[str],
    seq_len: int,
) -> Dict[str, np.ndarray]:
    """
    Extract observed intensity targets from peak annotations.

    Converts per-peak fragment annotations into per-ion-type intensity arrays.

    Parameters
    ----------
    peak_annotations
        Per-peak annotations. Each element is a list of annotations for that peak.
        Annotations can be :py:class:`ms2rescore_rs.FragmentAnnotation` objects or
        ``(series, position, charge)`` tuples.
    intensity
        Preprocessed intensity array (TIC-normalized, log2-transformed).
    ion_types
        Ion types to extract, e.g. ``["b", "y"]`` or ``["b", "y", "b2", "y2"]``.
    seq_len
        Length of the peptide sequence (number of amino acids).

    Returns
    -------
    targets
        Dict mapping ion type to intensity array of length ``seq_len - 1``.

    """
    n_ions = seq_len - 1
    floor_value = np.float32(np.log2(0.001))
    targets = {ion: np.full(n_ions, floor_value, dtype=np.float32) for ion in ion_types}

    for peak_idx, annotations in enumerate(peak_annotations):
        for ann in annotations:
            if isinstance(ann, tuple):
                series, position, charge = ann
            else:
                series, position, charge = ann.series, ann.position, ann.charge

            ion_key = series if charge == 1 else f"{series}{charge}"

            if ion_key not in targets:
                continue

            pos = position - 1
            if 0 <= pos < n_ions:
                if intensity[peak_idx] > targets[ion_key][pos]:
                    targets[ion_key][pos] = intensity[peak_idx]

    return targets


def load_and_match_spectra(
    psm_list: PSMList,
    spectrum_file: Union[str, Path],
    spectrum_id_pattern: str,
    model: str,
    ms2_tolerance: float,
    ms2_tolerance_mode: str,
) -> List[Tuple[int, PSM, ObservedSpectrum, List]]:
    """
    Read spectra from file, annotate, preprocess, and match to PSMs.

    Reads raw MS2Spectrum objects, matches to PSMs by spectrum ID, batch-annotates
    all matched spectra in a single Rust call, then converts to ObservedSpectrum
    and preprocesses.

    Returns list of (psm_index, psm, preprocessed_spectrum, peak_annotations) tuples.
    """
    try:
        spectrum_id_regex = re.compile(spectrum_id_pattern)
    except TypeError:
        spectrum_id_regex = re.compile(r"(.*)")

    psms_by_specid = defaultdict(list)
    for i, psm in enumerate(psm_list):
        psms_by_specid[str(psm.spectrum_id)].append((i, psm))

    # Step 1: Read raw spectra and match to PSMs (no conversion yet)
    matched_raw: List[Tuple[str, MS2Spectrum, List[Tuple[int, PSM]]]] = []
    for spectrum in _read_raw_spectra(str(spectrum_file)):
        match = spectrum_id_regex.search(str(spectrum.identifier))
        try:
            spectrum_id = match[1]
        except (TypeError, IndexError):
            raise exceptions.TitlePatternError(
                f"Spectrum title pattern `{spectrum_id_pattern}` could not be matched to "
                f"spectrum ID `{spectrum.identifier}`. "
                " Are you sure that the regex contains a capturing group?"
            )

        if spectrum_id not in psms_by_specid:
            continue

        matched_raw.append((spectrum_id, spectrum, psms_by_specid[spectrum_id]))

    if not matched_raw:
        return []

    # Step 2: Batch annotate all matched spectra (single Rust call, Rayon-parallelized)
    # Build parallel lists for the batch call: one entry per (spectrum, psm) pair
    batch_spectra = []
    batch_proformas = []
    batch_seq_lens = []
    batch_indices = []  # (matched_raw_idx, psm_within_spectrum_idx)

    for raw_idx, (_, spectrum, psm_pairs) in enumerate(matched_raw):
        for psm_idx, (_, psm) in enumerate(psm_pairs):
            batch_spectra.append(spectrum)
            batch_proformas.append(str(psm.peptidoform.proforma))
            batch_seq_lens.append(len(psm.peptidoform.parsed_sequence))
            batch_indices.append((raw_idx, psm_idx))

    frag_model = MODELS[model]["fragmentation"]
    annotated_spectra = annotate_ms2_spectra(
        spectra=batch_spectra,
        proformas=batch_proformas,
        seq_lens=batch_seq_lens,
        fragmentation_model=frag_model,
        mass_mode="monoisotopic",
        tolerance_value=float(ms2_tolerance),
        tolerance_mode=ms2_tolerance_mode.lower(),
    )

    # Step 3: Convert to ObservedSpectrum, preprocess, and assemble results
    # Cache converted/preprocessed spectra by spectrum_id to avoid redundant work
    preprocessed_cache: Dict[str, ObservedSpectrum] = {}
    results = []

    for batch_idx, (raw_idx, psm_idx) in enumerate(batch_indices):
        spec_id, raw_spectrum, psm_pairs = matched_raw[raw_idx]
        psm_index, psm = psm_pairs[psm_idx]

        if spec_id not in preprocessed_cache:
            obs = _to_observed_spectrum(raw_spectrum)
            for label_type in ["iTRAQ", "TMT"]:
                if label_type in model:
                    obs.remove_reporter_ions(label_type)
            obs.tic_norm()
            obs.log2_transform()
            preprocessed_cache[spec_id] = obs

        # Convert annotations to picklable tuples
        peak_annotations = [
            [(a.series, a.position, a.charge) for a in peak_anns]
            for peak_anns in annotated_spectra[batch_idx].peak_annotations
        ]

        results.append((psm_index, psm, preprocessed_cache[spec_id], peak_annotations))

    return results
