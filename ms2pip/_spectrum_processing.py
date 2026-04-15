"""Internal spectrum annotation, target extraction, and PSM-spectrum matching."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
from ms2rescore_rs import (
    AnnotatedMS2Spectrum,  # type: ignore[ty:unresolved-import]
    MS2Spectrum,  # type: ignore[ty:unresolved-import]
    Precursor,  # type: ignore[ty:unresolved-import]
    annotate_ms2_spectra,  # type: ignore[ty:unresolved-import]
    get_ms2_spectra,  # type: ignore[ty:unresolved-import]
)
from psm_utils import PSM, Peptidoform, PSMList

import ms2pip.exceptions as exceptions
from ms2pip.constants import MODELS
from ms2pip.spectrum import ObservedSpectrum

logger = logging.getLogger(__name__)


def _annotations_to_tuples(peak_annotations: list) -> list[list[tuple]]:
    """Convert FragmentAnnotation lists to plain (series, position, charge) tuples."""
    return [
        [(a.series, a.position, a.charge) for a in peak_anns] for peak_anns in peak_annotations
    ]


class MatchedSpectrum(NamedTuple):
    """A PSM matched to its preprocessed observed spectrum and peak annotations."""

    psm_index: int
    psm: PSM
    spectrum: ObservedSpectrum
    peak_annotations: list


def _read_raw_spectra(spectrum_file: str) -> Generator[MS2Spectrum, None, None]:
    """Read MS2 spectra as raw ms2rescore-rs objects (no conversion to ObservedSpectrum)."""
    try:
        spectra = get_ms2_spectra(str(spectrum_file))
    except ValueError as e:
        raise exceptions.UnsupportedSpectrumFiletypeError(Path(spectrum_file).suffixes) from e

    for spectrum in spectra:
        if str(spectrum.identifier) == "" or len(spectrum.mz) == 0 or len(spectrum.intensity) == 0:
            continue
        yield spectrum


def _to_observed_spectrum(spectrum: MS2Spectrum) -> ObservedSpectrum:
    """Convert an MS2Spectrum to an ObservedSpectrum (skips Pydantic validation)."""
    return ObservedSpectrum.model_construct(
        mz=np.array(spectrum.mz, dtype=np.float32),
        intensity=np.array(spectrum.intensity, dtype=np.float32),
        identifier=str(spectrum.identifier),
        precursor_mz=float(spectrum.precursor.mz),
        precursor_charge=int(spectrum.precursor.charge),
        retention_time=float(spectrum.precursor.rt),
    )


def _preprocess_spectrum(spectrum: ObservedSpectrum, model: str) -> None:
    """Remove reporter ions (if applicable), TIC-normalize, and log2-transform in place."""
    for label_type in ["iTRAQ", "TMT"]:
        if label_type in model:
            spectrum.remove_reporter_ions(label_type)
    spectrum.tic_norm()
    spectrum.log2_transform()


@lru_cache(maxsize=None)
def proforma_to_mass_shift(peptidoform: Peptidoform) -> str:
    """
    Convert a Peptidoform to a mass-shift ProForma string.

    Replaces all modification labels with numeric mass shifts so that
    ms2rescore-rs can parse them. Handles sequence modifications and
    N/C-terminal modifications.

    Note: This does not handle ProForma features like labile modifications,
    unlocalized modifications, tagged intervals, or isotope labels. These are
    not used by ms2pip.
    """
    parts = []
    n_term = peptidoform.properties.get("n_term")
    if n_term:
        for mod in n_term:
            parts.append(f"[{mod.mass:+.4f}]-")
    for aa, mods in peptidoform.parsed_sequence:
        parts.append(aa)
        if mods:
            for mod in mods:
                parts.append(f"[{mod.mass:+.4f}]")  # type: ignore[ty:unresolved-attribute]
    c_term = peptidoform.properties.get("c_term")
    if c_term:
        for mod in c_term:
            parts.append(f"-[{mod.mass:+.4f}]")
    if peptidoform.precursor_charge:
        parts.append(f"/{peptidoform.precursor_charge}")
    return "".join(parts)


def annotate_spectrum(
    spectrum: ObservedSpectrum,
    psm: PSM,
    model: str,
    ms2_tolerance: float,
    ms2_tolerance_mode: str,
) -> list[list[tuple]]:
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
    proforma = proforma_to_mass_shift(psm.peptidoform)
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
    return _annotations_to_tuples(annotated[0].peak_annotations)


def targets_from_annotations(
    peak_annotations: list,
    intensity: np.ndarray,
    ion_types: list[str],
    seq_len: int,
) -> dict[str, np.ndarray]:
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

    for peak_idx, peak_anns in enumerate(peak_annotations):
        for ann in peak_anns:
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


def _load_and_match_spectra(
    psm_list: PSMList,
    spectrum_file: str | Path,
    spectrum_id_pattern: str,
    model: str,
    ms2_tolerance: float,
    ms2_tolerance_mode: str,
) -> list[MatchedSpectrum]:
    """
    Read spectra from file, annotate, preprocess, and match to PSMs.

    Reads raw MS2Spectrum objects, matches to PSMs by spectrum ID, batch-annotates
    all matched spectra in a single Rust call, then converts to ObservedSpectrum
    and preprocesses.

    Returns list of :class:`MatchedSpectrum` instances.
    """
    try:
        spectrum_id_regex = re.compile(spectrum_id_pattern)
    except TypeError:
        spectrum_id_regex = re.compile(r"(.*)")

    psms_by_specid = defaultdict(list)
    for i, psm in enumerate(psm_list):
        psms_by_specid[str(psm.spectrum_id)].append((i, psm))

    # Step 1: Read raw spectra and match to PSMs (no conversion yet)
    logger.info("Reading spectra from file...")
    matched_raw: list[tuple[str, MS2Spectrum, list[tuple[int, PSM]]]] = []
    for spectrum in _read_raw_spectra(str(spectrum_file)):
        match = spectrum_id_regex.search(str(spectrum.identifier))
        try:
            spectrum_id = match[1]  # type: ignore[ty:not-subscriptable]
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
    logger.debug("Annotating %d matched spectra...", len(matched_raw))
    batch_spectra = []
    batch_proformas = []
    batch_seq_lens = []
    batch_indices = []  # (matched_raw_idx, psm_within_spectrum_idx)

    for raw_idx, (_, spectrum, psm_pairs) in enumerate(matched_raw):
        for psm_idx, (_, psm) in enumerate(psm_pairs):
            batch_spectra.append(spectrum)
            batch_proformas.append(proforma_to_mass_shift(psm.peptidoform))
            batch_seq_lens.append(len(psm.peptidoform.parsed_sequence))
            batch_indices.append((raw_idx, psm_idx))

    frag_model = MODELS[model]["fragmentation"]
    logger.debug("Starting annotation...")
    annotated_spectra = annotate_ms2_spectra(
        spectra=batch_spectra,
        proformas=batch_proformas,
        seq_lens=batch_seq_lens,
        fragmentation_model=frag_model,
        mass_mode="monoisotopic",
        tolerance_value=float(ms2_tolerance),
        tolerance_mode=ms2_tolerance_mode.lower(),
    )
    logger.debug("Annotation complete.")

    # Step 3: Convert to ObservedSpectrum, preprocess, and assemble results
    preprocessed_cache: dict[str, ObservedSpectrum] = {}
    results = []

    for batch_idx, (raw_idx, psm_idx) in enumerate(batch_indices):
        spec_id, raw_spectrum, psm_pairs = matched_raw[raw_idx]
        psm_index, psm = psm_pairs[psm_idx]

        if spec_id not in preprocessed_cache:
            obs = _to_observed_spectrum(raw_spectrum)
            _preprocess_spectrum(obs, model)
            preprocessed_cache[spec_id] = obs

        peak_annotations = _annotations_to_tuples(annotated_spectra[batch_idx].peak_annotations)

        results.append(
            MatchedSpectrum(psm_index, psm, preprocessed_cache[spec_id], peak_annotations)
        )

    return results


def _preloaded_to_annotations(
    psm_list: PSMList,
    model: str,
    ms2_tolerance: float,
    ms2_tolerance_mode: str,
) -> list[MatchedSpectrum]:
    """
    Convert preloaded MS2Spectrum/AnnotatedMS2Spectrum objects to matched spectra.

    Returns the same format as :func:`_load_and_match_spectra`: a list of
    :class:`MatchedSpectrum` instances.
    """
    first_spectrum = psm_list["spectrum"][0]
    spectra_are_annotated = isinstance(first_spectrum, AnnotatedMS2Spectrum)

    # Convert to ObservedSpectrum and preprocess; store raw spectra and annotations
    preloaded_spectra: dict[str, ObservedSpectrum] = {}
    raw_spectra: dict[str, MS2Spectrum] = {}
    preloaded_annotations: dict[str, list] | None = {} if spectra_are_annotated else None
    for psm in psm_list:
        spec_id = str(psm.spectrum_id)
        if spec_id in preloaded_spectra:
            continue
        spectrum = psm.spectrum
        assert spectrum is not None
        obs = _to_observed_spectrum(spectrum)
        _preprocess_spectrum(obs, model)
        preloaded_spectra[spec_id] = obs
        raw_spectra[spec_id] = spectrum  # keep original for annotation
        if spectra_are_annotated:
            assert isinstance(spectrum, AnnotatedMS2Spectrum)
            assert preloaded_annotations is not None
            preloaded_annotations[spec_id] = _annotations_to_tuples(spectrum.peak_annotations)

    # Build MatchedSpectrum list
    psm_spectrum_annotations: list[MatchedSpectrum] = []
    needs_annotation: list[int] = []

    for i, psm in enumerate(psm_list):
        spec_id = str(psm.spectrum_id)
        obs_spectrum = preloaded_spectra.get(spec_id)
        if obs_spectrum is None:
            continue
        if preloaded_annotations is not None and spec_id in preloaded_annotations:
            psm_spectrum_annotations.append(
                MatchedSpectrum(i, psm, obs_spectrum, preloaded_annotations[spec_id])
            )
        else:
            psm_spectrum_annotations.append(MatchedSpectrum(i, psm, obs_spectrum, []))
            needs_annotation.append(len(psm_spectrum_annotations) - 1)

    # Batch annotate any unannotated spectra using original MS2Spectrum objects
    if needs_annotation:
        frag_model = MODELS[model]["fragmentation"]
        batch_spectra = []
        batch_proformas = []
        batch_seq_lens = []
        for idx in needs_annotation:
            m = psm_spectrum_annotations[idx]
            batch_spectra.append(raw_spectra[str(m.psm.spectrum_id)])
            batch_proformas.append(proforma_to_mass_shift(m.psm.peptidoform))
            batch_seq_lens.append(len(m.psm.peptidoform.parsed_sequence))

        annotated = annotate_ms2_spectra(
            spectra=batch_spectra,
            proformas=batch_proformas,
            seq_lens=batch_seq_lens,
            fragmentation_model=frag_model,
            mass_mode="monoisotopic",
            tolerance_value=float(ms2_tolerance),
            tolerance_mode=ms2_tolerance_mode.lower(),
        )

        for j, idx in enumerate(needs_annotation):
            m = psm_spectrum_annotations[idx]
            peak_anns = _annotations_to_tuples(annotated[j].peak_annotations)
            psm_spectrum_annotations[idx] = m._replace(peak_annotations=peak_anns)

    return psm_spectrum_annotations


def resolve_spectra(
    psm_list: PSMList,
    spectrum_file: str | Path | None,
    spectrum_id_pattern: str | None,
    model: str,
    ms2_tolerance: float,
    ms2_tolerance_mode: str,
) -> list[MatchedSpectrum]:
    """
    Resolve spectra from preloaded PSM attributes or a spectrum file.

    Auto-detects whether PSMs carry preloaded spectra (``MS2Spectrum`` or
    ``AnnotatedMS2Spectrum``) or whether spectra should be read from file.
    """
    has_spectrum = [
        isinstance(psm.spectrum, (MS2Spectrum, AnnotatedMS2Spectrum)) for psm in psm_list
    ]
    if all(has_spectrum):
        if spectrum_file is not None:
            logger.warning("PSMs already have preloaded spectra; `spectrum_file` will be ignored.")
        matched = _preloaded_to_annotations(psm_list, model, ms2_tolerance, ms2_tolerance_mode)
    elif not any(has_spectrum):
        if spectrum_file is None:
            raise ValueError(
                "PSMs do not have preloaded spectra; `spectrum_file` must be provided."
            )
        spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"
        if len(psm_list.collections) != 1 or len(psm_list.runs) != 1:
            raise exceptions.InvalidInputError("PSMs should be for a single run and collection.")
        matched = _load_and_match_spectra(
            psm_list, spectrum_file, spectrum_id_pattern, model, ms2_tolerance, ms2_tolerance_mode
        )
    else:
        raise ValueError(
            "All PSMs must either have preloaded spectra or none of them should. "
            "Found a mix of PSMs with and without spectrum objects."
        )

    if not matched:
        raise exceptions.NoMatchingSpectraFound(
            "No spectra matching spectrum IDs from PSM list could be found."
        )

    return matched
