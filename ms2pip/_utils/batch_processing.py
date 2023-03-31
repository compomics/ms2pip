"""Process batch of peptides and spectra in parallel."""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from psm_utils import PSMList
from rich.progress import track

import ms2pip.exceptions as exceptions
from ms2pip._utils.encoder import Encoder
from ms2pip.constants import MODELS
from ms2pip.cython_modules import ms2pip_pyx
from ms2pip.result import ProcessingResult
from ms2pip.spectrum_input import read_spectrum_file


def process_peptides(
    worker_num: int,
    psm_list: PSMList,
    encoder: Encoder,
    model: str,
):
    """
    Predict spectrum for each entry in PeptideRecord DataFrame.

    Parameters
    ----------
    worker_num: int
        Index of worker if using multiprocessing
    psm_list: PSMList
        Peptides as PSMList
    encoder: Encoder
        Configured encoder to use for peptide and peptidoform encoding
    model: str
        Name of prediction model to be used

    Returns
    -------
    list[ProcessingResult]

    """
    ms2pip_pyx.ms2pip_init(encoder.afile, encoder.mod_file, encoder.mod_file2)

    results = []

    model_id = MODELS[model]["id"]
    peaks_version = MODELS[model]["peaks_version"]
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    # Track progress for only one worker (good approximation of all workers' progress)
    for psm in track(
        psm_list,
        total=len(psm_list),
        disable=worker_num != 0,
        transient=True,
        description="",
    ):
        try:
            enc_peptide = encoder.encode_peptide(psm.peptidoform)
            enc_peptidoform = encoder.encode_peptidoform(psm.peptidoform)
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
                ms2pip_pyx.get_vector(
                    enc_peptide, enc_peptidoform, psm.peptidoform.precursor_charge
                ),
                dtype=np.uint16,
            )
        # Else, get predictions from C models.
        else:
            predictions = ms2pip_pyx.get_predictions(
                enc_peptide,
                enc_peptidoform,
                psm.peptidoform.precursor_charge,
                model_id,
                peaks_version,
                30,  # CE, TODO: Remove
            )
            predictions = {
                i: np.array(p, dtype=np.float32) for i, p in zip(ion_types, predictions)
            }
            feature_vectors = None

        results.append(
            ProcessingResult(
                psm=psm,
                theoretical_mz=mzs,
                predicted_intensity=predictions,
                observed_intensity=None,
                feature_vectors=feature_vectors,
            )
        )
    return results


def process_spectra(
    worker_num: int,
    psm_list: PSMList,
    spec_file: str,
    vector_file: bool,
    encoder: Encoder,
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
    psm_list: PSMList
        PSMs as PSMList
    spec_file
        Filename of spectrum file
    vector_file
        If feature vectors should be extracted instead of predictions
    encoder: Encoder
        Configured encoder to use for peptide and peptidoform encoding
    model
        Name of prediction model to be used
    fragerror
        Fragmentation spectrum m/z error tolerance in Dalton
    spectrum_id_pattern
        Regular expression pattern to apply to spectrum titles before matching to
        peptide file entries

    Returns
    -------
    list[ProcessingResult]

    """
    ms2pip_pyx.ms2pip_init(encoder.afile, encoder.mod_file, encoder.mod_file2)

    results = []

    model_id = MODELS[model]["id"]
    peaks_version = MODELS[model]["peaks_version"]
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    try:
        spectrum_id_regex = re.compile(spectrum_id_pattern)
    except TypeError:
        spectrum_id_regex = re.compile(r"(.*)")

    # Restructure PeptideRecord entries as spec_id -> [psm_1, psm_2, ...]
    psms_by_specid = defaultdict(list)
    for psm in psm_list:
        psms_by_specid[psm.spectrum_id].append(psm)

    # Track progress for only one worker (good approximation of all workers' progress)
    for spectrum in track(
        read_spectrum_file(spec_file),
        total=len(psm_list),
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

        for psm in psms_by_specid[spectrum_id]:
            try:
                enc_peptide = encoder.encode_peptide(psm.peptidoform)
                enc_peptidoform = encoder.encode_peptidoform(psm.peptidoform)
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
                    ms2pip_pyx.get_vector(enc_peptide, enc_peptidoform, psm.peptidoform.charge),
                    dtype=np.uint16,
                )
            # Else, get predictions from C models in multiprocessing.
            else:
                predictions = ms2pip_pyx.get_predictions(
                    enc_peptide,
                    enc_peptidoform,
                    psm.peptidoform.precursor_charge,
                    model_id,
                    peaks_version,
                    30,  # CE, TODO: Remove,
                )
                predictions = {
                    i: np.array(p, dtype=np.float32) for i, p in zip(ion_types, predictions)
                }
                feature_vectors = None

            results.append(
                ProcessingResult(
                    psm=psm,
                    theoretical_mz=mzs,
                    predicted_intensity=predictions,
                    observed_intensity=targets,
                    feature_vectors=feature_vectors,
                )
            )

    return results
