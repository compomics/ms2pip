"""Process batch of peptides and spectra in parallel."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from rich.progress import track

import ms2pip.exceptions as exceptions
from ms2pip._utils.peptides import apply_modifications, encode_peptide
from ms2pip.constants import MODELS
from ms2pip.cython_modules import ms2pip_pyx
from ms2pip.result import ProcessingResult
from ms2pip.spectrum_input import read_spectrum_file


def process_peptides(
    worker_num: int,
    data: pd.DataFrame,
    afile: str,
    modfile: str,
    modfile2: str,
    ptm_ids: Dict[str, int],
    model: str,
):
    """
    Predict spectrum for each entry in PeptideRecord DataFrame.

    Parameters
    ----------
    worker_num: int
        Index of worker if using multiprocessing
    data: pandas.DataFrame
        PeptideRecord as Pandas DataFrame
    afile: str
        Filename of tempfile with amino acids definition for C code
    modfile: str
        Filename of tempfile with modification definition for C code
    modfile2: str
        Filename of tempfile with second instance of modification definition for C code
    ptm_ids: dict[str, int]
        Mapping of modification name -> modified residue integer encoding
    model: str
        Name of prediction model to be used

    Yields
    -------
    ProcessingResult

    """
    ms2pip_pyx.ms2pip_init(afile, modfile, modfile2)

    model_id = MODELS[model]["id"]
    peaks_version = MODELS[model]["peaks_version"]
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    # Track progress for only one worker (good approximation of all workers' progress)
    for entry in track(
        data.itertuples(),
        total=len(data),
        disable=worker_num != 0,
        transient=True,
        description="",
    ):
        try:
            enc_peptide = encode_peptide(entry.peptide)
            enc_peptidoform = apply_modifications(enc_peptide, entry.modifications, ptm_ids)
        except (
            exceptions.InvalidPeptideError,
            exceptions.InvalidAminoAcidError,
            exceptions.InvalidModificationFormattingError,
            exceptions.UnknownModificationError,
        ):
            continue

        # Get ion mzs and map to ion types
        mzs = ms2pip_pyx.get_mzs(enc_peptidoform, peaks_version)
        mzs = {i: np.array(mz, dtype=np.float32) for i, mz in zip(ion_types, mzs)}

        # If using xgboost model file, get feature vectors to predict outside of MP.
        # Predictions will be added in `_merge_predictions` function.
        if "xgboost_model_files" in MODELS[model].keys():
            predictions = None
            feature_vectors = np.array(
                ms2pip_pyx.get_vector(enc_peptide, enc_peptidoform, entry.charge),
                dtype=np.uint16,
            )
        # Else, get predictions from C models.
        else:
            predictions = ms2pip_pyx.get_predictions(
                enc_peptide,
                enc_peptidoform,
                entry.charge,
                model_id,
                peaks_version,
                entry.ce,
            )
            predictions = {
                i: np.array(p, dtype=np.float32) for i, p in zip(ion_types, predictions)
            }
            feature_vectors = None

        yield ProcessingResult(
            psm_id=entry.psm_id,
            spectrum_id=entry.spec_id,
            sequence=entry.peptide,
            modifications=entry.modifications,
            charge=entry.charge,
            retention_time=entry.rt if "rt" in data.columns else None,
            theoretical_mz=mzs,
            predicted_intensity=predictions,
            observed_intensity=None,
            feature_vectors=feature_vectors,
        )


def process_spectra(
    worker_num: int,
    data: pd.DataFrame,
    spec_file: str,
    vector_file: bool,
    afile: str,
    modfile: str,
    modfile2: str,
    ptm_ids: Dict[str, list],
    model: str,
    fragerror: float,
    spectrum_id_pattern: str,
):
    """
    Perform requested tasks for each spectrum in spectrum file.

    Parameters
    ----------
    worker_num
        Index of worker if using multiprocessing
    data
        PeptideRecord as Pandas DataFrame
    spec_file
        Filename of spectrum file
    vector_file
        If feature vectors should be extracted instead of predictions
    afile
        Filename of tempfile with amino acids definition for C code
    modfile
        Filename of tempfile with modification definition for C code
    modfile2
        Filename of tempfile with second instance of modification definition for C code
    ptm_ids
        Mapping of modification name -> modified residue integer encoding
    model
        Name of prediction model to be used
    fragerror
        Fragmentation spectrum m/z error tolerance in Dalton
    spectrum_id_pattern
        Regular expression pattern to apply to spectrum titles before matching to
        peptide file entries

    Returns
    -------
    ProcessingResult

    """
    ms2pip_pyx.ms2pip_init(afile, modfile, modfile2)

    model_id = MODELS[model]["id"]
    peaks_version = MODELS[model]["peaks_version"]
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    try:
        spectrum_id_regex = re.compile(spectrum_id_pattern)
    except TypeError:
        spectrum_id_regex = re.compile(r"(.*)")

    # Restructure PeptideRecord entries as spec_id -> [psm_1, psm_2, ...]
    entries_by_specid = defaultdict(list)
    for entry in data.itertuples():
        entries_by_specid[entry.spec_id].append(entry)

    # Track progress for only one worker (good approximation of all workers' progress)
    for spectrum in track(
        read_spectrum_file(spec_file),
        total=len(data),
        disable=worker_num != 0,
        transient=True,
        description="",
    ):
        # Match spectrum ID with provided regex, use first match group as new ID
        match = spectrum_id_regex.search(spectrum.identifier)
        try:
            spectrum_id = match[1]
        except (TypeError, IndexError):
            raise exceptions.TitlePatternError(
                "Spectrum title pattern could not be matched to spectrum IDs "
                f"`{spectrum.identifier}`. "
                " Are you sure that the regex contains a capturing group?"
            )

        # Spectrum preprocessing:
        # Remove reporter ions and precursor peak, normalize, transform
        for label_type in ["iTRAQ", "TMT"]:
            if label_type in model:
                spectrum.remove_reporter_ions(label_type)
        # spectrum.remove_precursor()  # TODO: Decide to implement this or not
        spectrum.tic_norm()
        spectrum.log2_transform()

        for entry in entries_by_specid[spectrum_id]:
            try:
                enc_peptide = encode_peptide(entry.peptide)
                enc_peptidoform = apply_modifications(enc_peptide, entry.modifications, ptm_ids)
            except (
                exceptions.InvalidPeptideError,
                exceptions.InvalidAminoAcidError,
                exceptions.InvalidModificationFormattingError,
                exceptions.UnknownModificationError,
            ):
                continue

            targets = ms2pip_pyx.get_targets(
                enc_peptidoform,
                spectrum.mz,
                spectrum.intensity,
                float(fragerror),
                peaks_version,
            )
            targets = {i: np.array(t, dtype=np.float32) for i, t in zip(ion_types, targets)}

            if not vector_file:
                mzs = ms2pip_pyx.get_mzs(enc_peptidoform, peaks_version)
                mzs = {i: np.array(mz, dtype=np.float32) for i, mz in zip(ion_types, mzs)}
            else:
                mzs = None

            # If using xgboost model file, get feature vectors to predict outside of MP.
            # Predictions will be added in `_merge_predictions` function.
            if vector_file or "xgboost_model_files" in MODELS[model].keys():
                predictions = None
                feature_vectors = np.array(
                    ms2pip_pyx.get_vector(enc_peptide, enc_peptidoform, entry.charge),
                    dtype=np.uint16,
                )
            # Else, get predictions from C models in multiprocessing.
            else:
                predictions = ms2pip_pyx.get_predictions(
                    enc_peptide,
                    enc_peptidoform,
                    entry.charge,
                    model_id,
                    peaks_version,
                    entry.ce,
                )
                predictions = {
                    i: np.array(p, dtype=np.float32) for i, p in zip(ion_types, predictions)
                }
                feature_vectors = None

            yield ProcessingResult(
                spectrum_id=spectrum_id,
                psm_id=entry.psm_id,
                sequence=entry.peptide,
                modifications=entry.modifications,
                charge=entry.charge,
                retention_time=entry.rt if "rt" in data.columns else None,
                theoretical_mz=mzs,
                predicted_intensity=predictions,
                observed_intensity=targets,
                feature_vectors=feature_vectors,
            )
