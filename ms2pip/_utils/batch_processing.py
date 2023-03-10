from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from rich.progress import track

from ms2pip import exceptions
from ms2pip._utils.peptides import apply_modifications, encode_peptide
from ms2pip.constants import MODELS
from ms2pip.cython_modules import ms2pip_pyx
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

    Returns
    -------
    psm_id_buf: list
    spec_id_buf: list
    peplen_buf: list
    charge_buf: list
    mz_buf: list
    target_buf: list
    prediction_buf: list
    vector_buf: list

    """
    ms2pip_pyx.ms2pip_init(afile, modfile, modfile2)

    model_id = MODELS[model]["id"]
    peaks_version = MODELS[model]["peaks_version"]

    # Prepare output variables
    psm_id_buf = []
    spec_id_buf = []
    peplen_buf = []
    charge_buf = []
    mz_buf = []
    target_buf = None
    prediction_buf = []
    vector_buf = []

    # Track progress for only one worker (good approximation of all workers' progress)
    for entry in track(
        data.itertuples(),
        total=len(data),
        disable=worker_num != 0,
        transient=True,
        description="",
    ):
        psm_id_buf.append(entry.psm_id)
        spec_id_buf.append(entry.spec_id)
        peplen_buf.append(len(entry.peptide))
        charge_buf.append(entry.charge)

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

        # Get ion mzs
        mzs = ms2pip_pyx.get_mzs(enc_peptidoform, peaks_version)
        mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])

        # If using xgboost model file, get feature vectors to predict outside of MP.
        # Predictions will be added in `_merge_predictions` function.
        if "xgboost_model_files" in MODELS[model].keys():
            vectors = np.array(
                ms2pip_pyx.get_vector(enc_peptide, enc_peptidoform, entry.charge),
                dtype=np.uint16,
            )
            vector_buf.append(vectors)
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
            prediction_buf.append([np.array(p, dtype=np.float32) for p in predictions])

    return (
        psm_id_buf,
        spec_id_buf,
        peplen_buf,
        charge_buf,
        mz_buf,
        target_buf,
        prediction_buf,
        vector_buf,
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
    psm_id_buf: list
    spec_id_buf: list
    peplen_buf: list
    charge_buf: list
    mz_buf: list
    target_buf: list
    prediction_buf: list
    vector_buf: list

    """
    ms2pip_pyx.ms2pip_init(afile, modfile, modfile2)

    model_id = MODELS[model]["id"]
    peaks_version = MODELS[model]["peaks_version"]

    # if vector_file
    dvectors = []
    dtargets = dict()
    psmids = []

    #
    psm_id_buf = []
    spec_id_buf = []
    peplen_buf = []
    charge_buf = []
    mz_buf = []
    target_buf = []
    prediction_buf = []
    vector_buf = []

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

            if vector_file:
                targets = ms2pip_pyx.get_targets(
                    enc_peptidoform,
                    spectrum.mz,
                    spectrum.intensity,
                    float(fragerror),
                    peaks_version,
                )
                psmids.extend([spectrum_id] * (len(targets[0])))
                dvectors.append(
                    np.array(
                        ms2pip_pyx.get_vector(enc_peptide, enc_peptidoform, entry.charge),
                        dtype=np.uint16,
                    )
                )

                # Restructure to dict with entries per ion type
                # Flip the order for C-terminal ion types
                for i, t in enumerate(targets):
                    if i in dtargets.keys():
                        if i % 2 == 0:
                            dtargets[i].extend(t)
                        else:
                            dtargets[i].extend(t[::-1])
                    else:
                        if i % 2 == 0:
                            dtargets[i] = [t]
                        else:
                            dtargets[i] = [t[::-1]]

            else:
                # Predict the b- and y-ion intensities from the peptide
                psm_id_buf.append(entry.psm_id)
                spec_id_buf.append(spectrum_id)
                peplen_buf.append(len(entry.peptide))
                charge_buf.append(entry.charge)

                targets = ms2pip_pyx.get_targets(
                    enc_peptidoform,
                    spectrum.mz,
                    spectrum.intensity,
                    float(fragerror),
                    peaks_version,
                )
                target_buf.append([np.array(t, dtype=np.float32) for t in targets])

                mzs = ms2pip_pyx.get_mzs(enc_peptidoform, peaks_version)
                mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])

                # If using xgboost model file, get feature vectors to predict outside of MP.
                # Predictions will be added in `_merge_predictions` function.
                if "xgboost_model_files" in MODELS[model].keys():
                    vector_buf.append(
                        np.array(
                            ms2pip_pyx.get_vector(enc_peptide, enc_peptidoform, entry.charge),
                            dtype=np.uint16,
                        )
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
                    prediction_buf.append([np.array(p, dtype=np.float32) for p in predictions])

    # If feature vectors requested, return specific data
    if vector_file:
        if dvectors:
            # If processes > number of spectra, dvectors can be empty
            if len(dvectors) >= 1:
                # Concatenate dvectors into 2D ndarray before making DataFrame to reduce
                # memory usage
                dvectors = np.concatenate(dvectors)
            df = pd.DataFrame(dvectors, dtype=np.uint16, copy=False)
            df.columns = df.columns.astype(str)
        else:
            df = pd.DataFrame()
        return psmids, df, dtargets

    # Else, return general data
    return (
        psm_id_buf,
        spec_id_buf,
        peplen_buf,
        charge_buf,
        mz_buf,
        target_buf,
        prediction_buf,
        vector_buf,
    )
