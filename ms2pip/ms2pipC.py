#!/usr/bin/env python
from __future__ import annotations

import csv
import glob
import itertools
import logging
import multiprocessing
import multiprocessing.dummy
import os
import re
from collections import defaultdict
from pathlib import Path
from random import shuffle
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from rich.progress import track

import ms2pip.exceptions as exceptions
import ms2pip.peptides
from ms2pip.constants import MODELS, SUPPORTED_OUTPUT_FORMATS
from ms2pip.cython_modules import ms2pip_pyx
from ms2pip.match_spectra import MatchSpectra
from ms2pip.ms2pip_tools import calc_correlations, spectrum_output
from ms2pip.predict_xgboost import (get_predictions_xgb,
                                    validate_requested_xgb_model)
from ms2pip.retention_time import RetentionTime
from ms2pip.spectrum import read_spectrum_file

logger = logging.getLogger(__name__)


def process_peptides(
    worker_num: int,
    data: pd.DataFrame,
    afile: str,
    modfile: str,
    modfile2: str,
    ptm_ids: dict[str, int],
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
            enc_peptide = ms2pip.peptides.encode_peptide(entry.peptide)
            enc_peptidoform = ms2pip.peptides.apply_modifications(
                enc_peptide, entry.modifications, ptm_ids
            )
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
    worker_num,
    data,
    spec_file,
    vector_file,
    afile,
    modfile,
    modfile2,
    ptm_ids,
    model,
    fragerror,
    spectrum_id_pattern,
):
    """
    Perform requested tasks for each spectrum in spectrum file.

    Parameters
    ----------
    worker_num: int
        Index of worker if using multiprocessing
    data: pandas.DataFrame
        PeptideRecord as Pandas DataFrame
    spec_file: str
        Filename of spectrum file
    vector_file: str, None
        Output filename for feature vector file
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
    fragerror: float
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
        match = spectrum_id_regex.search(spectrum.title)
        try:
            spectrum_id = match[1]
        except (TypeError, IndexError):
            raise exceptions.TitlePatternError(
                "Spectrum title pattern could not be matched to spectrum IDs "
                f"`{spectrum.title}`. "
                " Are you sure that the regex contains a capturing group?"
            )

        # Spectrum preprocessing:
        # Remove reporter ions and precursor peak, normalize, transform
        for label_type in ["iTRAQ", "TMT"]:
            if label_type in model:
                spectrum.remove_reporter_ions(label_type)
        # spectrum.remove_precursor()
        spectrum.tic_norm()
        spectrum.log2_transform()

        for entry in entries_by_specid[spectrum_id]:
            try:
                enc_peptide = ms2pip.peptides.encode_peptide(entry.peptide)
                enc_peptidoform = ms2pip.peptides.apply_modifications(
                    enc_peptide, entry.modifications, ptm_ids
                )
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
                    spectrum.msms,
                    spectrum.peaks,
                    float(fragerror),
                    peaks_version,
                )
                psmids.extend([spectrum_id] * (len(targets[0])))
                dvectors.append(
                    np.array(
                        ms2pip_pyx.get_vector(
                            enc_peptide, enc_peptidoform, entry.charge
                        ),
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
                    spectrum.msms,
                    spectrum.peaks,
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
                            ms2pip_pyx.get_vector(
                                enc_peptide, enc_peptidoform, entry.charge
                            ),
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
                    prediction_buf.append(
                        [np.array(p, dtype=np.float32) for p in predictions]
                    )

    # If feature vectors requested, return specific data
    if vector_file:
        if dvectors:
            # If num_cpu > number of spectra, dvectors can be empty
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


class MS2PIP:
    def __init__(
        self,
        params: Optional[Dict] = None,
        limit: Optional[int] = None,
        model: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
        output_formats: Optional[List[str]] = None,
        num_cpu: Optional[int] = None,
    ):
        """
        MS²PIP peak intensity predictor.

        Parameters
        ----------
        params : dict
            Configuration dictionary with ``model``, ``frag_error``, ``ptm``, and ``out``.
        limit : int, optional
            Limit to first N peptides in peptide file.
        model : str
            Name of the model to use for predictions. Overrides configuration file.
        model_dir : str, optional
            Custom directory for downloaded XGBoost model files. By default, `~/.ms2pip` is used.
        output_formats : list[str], optional
            List of output formats to use. Overrides configuration file.
        num_cpu : int, optional
            Number of parallel processes for multiprocessing steps. By default, all available.

        Examples
        --------
        >>> from ms2pip.ms2pipC import MS2PIP
        >>> params = {
        ...     "ms2pip": {
        ...         "ptm": [
        ...             "Oxidation,15.994915,M",
        ...             "Carbamidomethyl,57.021464,C",
        ...             "Acetyl,42.010565,N-term",
        ...         ],
        ...         "model": "HCD",
        ...         "frag_error": 0.02,
        ...         "out": "csv",
        ...     }
        ... }
        >>> ms2pip = MS2PIP(params=params)
        >>> ms2pip.predict("test.peprec")

        """
        self.params = params
        self.limit = limit
        self.model_dir = model_dir
        self.num_cpu = num_cpu if num_cpu else multiprocessing.cpu_count()

        self.afile = None
        self.modfile = None
        self.modfile2 = None

        # Set default parameters if none provided
        if self.params is None or "ms2pip" not in self.params:
            logger.debug("No parameters provided, using default parameters.")
            self.params = {
                "ms2pip": {
                    "ptm": [],
                    "sptm": [],
                    "model": "HCD",
                    "frag_error": 0.02,
                    "out": "csv",
                }
            }

        # Validate parameters
        if model:
            self.model = model
        elif "model" in self.params["ms2pip"]:
            self.model = self.params["ms2pip"]["model"]
        elif "frag_method" in self.params["ms2pip"]:
            self.model = self.params["ms2pip"]["frag_method"]
        else:
            raise exceptions.FragmentationModelRequiredError()
        self.fragerror = self.params["ms2pip"]["frag_error"]
        if output_formats:
            self.output_formats = self._validate_output_formats(output_formats)
        else:
            self.output_format = self._validate_output_formats(self.params["ms2pip"]["out"])

        # Validate model_dir
        if not self.model_dir:
            self.model_dir = os.path.join(os.path.expanduser("~"), ".ms2pip")

        # Validate requested model
        if self.model in MODELS.keys():
            logger.debug("Using %s models", self.model)
            if "xgboost_model_files" in MODELS[self.model].keys():
                validate_requested_xgb_model(
                    MODELS[self.model]["xgboost_model_files"],
                    MODELS[self.model]["model_hash"],
                    self.model_dir,
                )
        else:
            raise exceptions.UnknownFragmentationMethodError(self.model)

        # Set up multiprocessing
        logger.debug(f"Starting workers (num_cpu={self.num_cpu})...")
        if multiprocessing.current_process().daemon:
            logger.warn(
                "MS2PIP is running in a daemon process. Disabling multiprocessing as daemonic processes can't have children."
            )
            self.myPool = multiprocessing.dummy.Pool(1)
        elif self.num_cpu == 1:
            logger.debug("Using dummy multiprocessing pool.")
            self.myPool = multiprocessing.dummy.Pool(1)
        else:
            self.myPool = multiprocessing.Pool(self.num_cpu)

        self.mods = ms2pip.peptides.Modifications()
        for mod_type in ("sptm", "ptm"):
            self.mods.add_from_ms2pip_modstrings(
                self.params["ms2pip"][mod_type], mod_type=mod_type
            )

    def predict(
        self,
        peptides: Union[str, Path, pd.DataFrame],
        output_filename: Optional[str] = None,
        add_retention_time: bool = False,
        return_results: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Predict peptide fragment ion intensities.

        Parameters
        ----------
        peptides : str, Path, pandas.DataFrame
            Path to file or ``pandas.DataFrame`` with peptide information (see
            https://github.com/compomics/ms2pip_c#peprec-file)
        output_filename : str, optional
            Filepath prefix for output files
        add_retention_time : bool, default: False
            Add retention time predictions with DeepLC (Requires optional DeepLC dependency).
        return_results : bool, default: False
            Return results instead of writing to output files.

        Returns
        -------
        predictions: pandas.DataFrame
            Predicted spectra. Only returned if ``return_results`` is True.

        """
        self._setup_modification_files()
        peptides = self._read_peptide_information(peptides)
        output_filename = self._get_output_filename(output_filename, peptides, return_results)

        if add_retention_time:
            logger.info("Adding retention time predictions")
            rt_predictor = RetentionTime(config=self.params, num_cpu=self.num_cpu)
            rt_predictor.add_rt_predictions(peptides)

        logger.info("Processing peptides...")
        results = self._process_peptides(peptides)

        logger.debug("Merging results ...")
        predictions = self._merge_predictions(results)

        if not return_results:
            self._write_predictions(predictions)
        else:
            return predictions

    def correlate(
        self,
        peptides: Union[str, Path, pd.DataFrame],
        spectrum_file: Union[str, Path],
        spectrum_id_pattern: Optional[str] = None,
        output_filename: Optional[str] = None,
        compute_correlations: bool = False,
        add_retention_time: bool = False,
        return_results: bool = False,
    ) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        """
        Compare predicted and observed intensities and optionally .

        Parameters
        ----------
        peptides : str, Path, pandas.DataFrame
            Path to file or ``pandas.DataFrame`` with peptide information (see
            https://github.com/compomics/ms2pip_c#peprec-file)
        spectrum_file : str, Path, optional
            Path to spectrum file with target intensities.
        spectrum_id_pattern : str, optional
            Regular expression pattern to apply to spectrum titles before matching to
            peptide file ``spec_id`` entries.
        output_filename : str, optional
            Filepath prefix for output files
        compute_correlations : bool, default: False
            Compute correlations between predictions and targets.
        add_retention_time : bool, default: False
            Add retention time predictions with DeepLC (Requires optional DeepLC dependency).
        return_results : bool, default: False
            Return results after prediction (`MS2PIP.run()`) instead of writing to output files.

        Returns
        -------
        pred_and_emp: pandas.DataFrame, optional, correlations: pandas.DataFrame, optional
            Tuple of ``pandas.DataFrame`` with predicted and empirical intensities and (if
            requested) ``pandas.DataFrame`` with correlations. Only returned if ``return_results``
            is True.

        """
        self._setup_modification_files()
        peptides = self._read_peptide_information(peptides)
        output_filename = self._get_output_filename(output_filename, peptides, return_results)
        spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

        if add_retention_time:
            logger.info("Adding retention time predictions")
            rt_predictor = RetentionTime(config=self.params, num_cpu=self.num_cpu)
            rt_predictor.add_rt_predictions(peptides)

        logger.info("Processing spectra and peptides...")
        results = self._process_spectra(peptides, spectrum_file, spectrum_id_pattern)

        logger.debug("Merging results")
        pred_and_emp = self._merge_predictions(results)

        # Correlations also requested
        if compute_correlations:
            logger.info("Computing correlations")
            correlations = calc_correlations.calc_correlations(pred_and_emp)
            logger.info(
                "Median correlations: \n%s",
                str(correlations.groupby("ion")["pearsonr"].median()),
            )
        else:
            correlations = None

        if not return_results:
            # Write output to files
            pred_and_emp_filename = self.output_filename + "_pred_and_emp.csv"
            logger.info(f"Writing file {pred_and_emp_filename}...")
            try:
                pred_and_emp.to_csv(pred_and_emp_filename, index=False, lineterminator="\n")
            except TypeError:  # Pandas < 1.5 (Required for Python 3.7 support)
                pred_and_emp.to_csv(pred_and_emp_filename, index=False, line_terminator="\n")
            if correlations:
                corr_filename = self.output_filename + "_correlations.csv"
                logger.info(f"Writing file {corr_filename}")
                try:
                    correlations.to_csv(corr_filename, index=False, lineterminator="\n")
                except TypeError:  # Pandas < 1.5 (Required for Python 3.7 support)
                    correlations.to_csv(corr_filename, index=False, line_terminator="\n")
        else:
            return pred_and_emp, correlations

    def get_features(
        self,
        peptides: Union[str, Path, pd.DataFrame],
        spectrum_file: Union[str, Path],
        spectrum_id_pattern: Optional[str] = None,
        output_filename: Optional[str] = None,
        return_results: bool = False,
    ):
        """
        Extract feature vectors and target intensities from spectra.

        Parameters
        ----------
        peptides : str, Path, pandas.DataFrame
            Path to file or ``pandas.DataFrame`` with peptide information (see
            https://github.com/compomics/ms2pip_c#peprec-file)
        spectrum_file : str, Path, optional
            Path to spectrum file with target intensities.
        spectrum_id_pattern : str, optional
            Regular expression pattern to apply to spectrum titles before matching to
            peptide file ``spec_id`` entries.
        output_filename : str, optional
            Filepath prefix for output files
        return_results : bool, default: False
            Return results instead of writing to output files.

        Returns
        -------
        features: pandas.DataFrame, optional
            ``pandas.DataFrame`` with feature vectors and targets. Only returned if
            ``return_results`` is True.

        """
        self._setup_modification_files()
        peptides = self._read_peptide_information(peptides)
        output_filename = self._get_output_filename(output_filename, peptides, return_results)
        spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

        if self.add_retention_time:
            logger.info("Adding retention time predictions")
            rt_predictor = RetentionTime(config=self.params, num_cpu=self.num_cpu)
            rt_predictor.add_rt_predictions(peptides)

        logger.info("Processing spectra and peptides...")
        vector_filename = output_filename + "_vectors.csv"
        results = self._process_spectra(
            peptides,
            spectrum_file,
            spectrum_id_pattern,
            vector_file=vector_filename
        )
        vectors = self._write_vector_file(results)
        if return_results:
            return vectors

    def match_spectra(
        self,
        peptides,
        spectrum_file: Union[str, Path],
        sqldb_uri: str,
    ):
        """
        Match spectra to peptides based on peak intensities.

        Match spectra in `spectrum_file` or `sqldb_uri` to peptides in `pep_file` based on
        predicted intensities (experimental).

        Parameters
        ----------
        peptides : str, Path, pandas.DataFrame
            Path to file or ``pandas.DataFrame`` with peptide information (see
            https://github.com/compomics/ms2pip_c#peprec-file)
        spectrum_file : str, Path, optional
            Path to spectrum file with target intensities.
        sqldb_uri : str, optional
            URI to SQL database for `match_spectra` feature.

        """
        self._setup_modification_files()
        peptides = self._read_peptide_information(peptides)
        output_filename = self._get_output_filename(
            output_filename, peptides, return_results=False
        )

        # Set spec_files based on spec_file or sqldb_uri
        spectrum_file = None
        if sqldb_uri:
            spectrum_files = None
        elif os.path.isdir(spectrum_file):
            spectrum_files = glob.glob("{}/*.mgf".format(spectrum_file))
        else:
            spectrum_files = [spectrum_file]
        logger.debug("Using spectrum files %s", spectrum_files)

        # Process
        results = self._process_peptides(peptides)
        matched_spectra = self._match_spectra(results, peptides, spectrum_files, sqldb_uri)
        self._write_matched_spectra(matched_spectra, output_filename)

    def cleanup(self):
        """Cleanup temporary files."""
        if self.afile:
            os.remove(self.afile)
        if self.modfile:
            os.remove(self.modfile)
        if self.modfile2:
            os.remove(self.modfile2)

    def _setup_modification_files(self):
        """Write modification config to files for reading by C-code."""
        self.afile = ms2pip.peptides.write_amino_acid_masses()
        self.modfile = self.mods.write_modifications_file(mod_type="ptm")
        self.modfile2 = self.mods.write_modifications_file(mod_type="sptm")

    def _validate_output_formats(self, output_formats: List[str], return_results: bool) -> List[str]:
        """Validate requested output formats."""
        if "out" in self.params["ms2pip"]:
            output_formats = [
                out_f.lower().strip() for out_f in self.params["ms2pip"]["out"].split(",")
            ]
            for output_format in output_formats:
                if output_format not in SUPPORTED_OUTPUT_FORMATS:
                    raise exceptions.UnknownOutputFormatError(output_format)
        else:
            if not return_results:
                logger.info("No output format specified; defaulting to CSV.")
                self.output_formats = ["csv"]
            else:
                self.output_formats = []

    def _get_output_filename(
            self,
            output_filename: Optional[str],
            peptides: Union[str, Path, pd.DataFrame, None],
            return_results: bool,
        ) -> str:
        """Get output filename from input filename if not defined."""
        if isinstance(peptides, Path):
            peptides = str(peptides)
        # Only set output filename if not defined, required, and peptides is a filepath
        if output_filename is None and not return_results and isinstance(peptides, str):
            return "{}_{}".format(".".join(peptides.split(".")[:-1]), self.model)
        else:
            return output_filename

    def _read_peptide_information(self, peptides: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Validate, read, and process peptide information."""
        # Read
        if isinstance(peptides, (str, Path)):
            with open(peptides, "rt") as f:
                line = f.readline()
                if line[:7] != "spec_id":
                    raise exceptions.InvalidPEPRECError()
                sep = line[7]
            data = pd.read_csv(
                peptides,
                sep=sep,
                index_col=False,
                dtype={"spec_id": str, "modifications": str},
                nrows=self.limit,
            )
        elif isinstance(peptides, pd.DataFrame):
            data = peptides
        else:
            raise TypeError("Invalid type for peptide file")

        # Validate
        if len(data) == 0:
            raise exceptions.NoValidPeptideSequencesError()
        data = data.fillna("-")
        if not "ce" in data.columns:
            data["ce"] = 30
        else:
            data["ce"] = data["ce"].astype(int)

        data["charge"] = data["charge"].astype(int)

        # Filter for unsupported peptides
        num_pep = len(data)
        data = data[
            ~(data["peptide"].str.contains("B|J|O|U|X|Z"))
            & ~(data["peptide"].str.len() < 3)
            & ~(data["peptide"].str.len() > 99)
        ].copy()
        num_pep_filtered = num_pep - len(data)
        if num_pep_filtered > 0:
            logger.warning(
                f"Removed {num_pep_filtered} unsupported peptide sequences (< 3, > 99 amino "
                f"acids, or containing B, J, O, U, X or Z). Retained {len(data)} entries."
            )

        if len(data) == 0:
            raise exceptions.NoValidPeptideSequencesError()

        if not "psm_id" in data.columns:
            data.reset_index(inplace=True)
            data["psm_id"] = data["index"].astype(str)
            data.rename({"index": "psm_id"}, axis=1)

        return data

    @staticmethod
    def _prepare_titles(titles, num_cpu: int):
        """Split list of spec_ids over number of CPUs."""
        shuffle(titles)  # Shuffling to improve parallel speeds
        split_titles = [
            titles[i * len(titles) // num_cpu : (i + 1) * len(titles) // num_cpu]
            for i in range(num_cpu)
        ]
        logger.debug(
            "{} spectra (~{:.0f} per cpu)".format(
                len(titles), np.mean([len(a) for a in split_titles])
            )
        )
        return split_titles

    def _execute_in_pool(self, peptides: pd.DataFrame, func: Callable, args: tuple):
        split_spec_ids = self._prepare_titles(peptides["spec_id"].to_list(), self.num_cpu)
        results = []
        for i in range(self.num_cpu):
            tmp = split_spec_ids[i]
            results.append(
                self.myPool.apply_async(
                    func,
                    args=(i, peptides[peptides.spec_id.isin(tmp)], *args),
                )
            )
        self.myPool.close()
        self.myPool.join()
        return results

    def _process_peptides(self, peptides: pd.DataFrame):
        return self._execute_in_pool(
            peptides,
            process_peptides,
            (self.afile, self.modfile, self.modfile2, self.mods.ptm_ids, self.model),
        )

    def _process_spectra(
            self,
            peptides: pd.DataFrame,
            spectrum_file: Union[str, Path],
            spectrum_id_pattern: str,
            vector_file: Optional[Union[str, Path]] = None,
        ):
        """
        When an mgf file is provided, MS2PIP either saves the feature vectors to
        train models with or writes a file with the predicted spectra next to
        the empirical one.
        """
        return self._execute_in_pool(
            peptides,
            process_spectra,
            (
                spectrum_file,
                vector_file,
                self.afile,
                self.modfile,
                self.modfile2,
                self.mods.ptm_ids,
                self.model,
                self.fragerror,
                spectrum_id_pattern,
            ),
        )

    def _write_vector_file(self, results):
        all_results = []
        for r in results:
            psmids, df, dtargets = r.get()

            # dtargets is a dict, containing targets for every ion type (keys are int)
            for i, t in dtargets.items():
                df[
                    "targets_{}".format(MODELS[self.model]["ion_types"][i])
                ] = np.concatenate(t, axis=None)
            df["psmid"] = psmids

            all_results.append(df)

        # Only concat DataFrames with content (we get empty ones if more CPUs than peptides)
        all_results = pd.concat([df for df in all_results if len(df) != 0])

        logger.info("Writing vector file %s...", self.vector_file)
        # TODO Consider writing to DMatrix XGBoost binary file instead.
        # write result. write format depends on extension:
        ext = self.vector_file.split(".")[-1]
        if ext == "pkl":
            all_results.to_pickle(self.vector_file + ".pkl")
        elif ext == "csv":
            try:
                all_results.to_csv(self.vector_file, lineterminator="\n")
            except TypeError:  # Pandas < 1.5 (Required for Python 3.7 support)
                all_results.to_csv(self.vector_file, line_terminator="\n")
        else:
            # "table" is a tag used to read back the .h5
            all_results.to_hdf(self.vector_file, "table")

        return all_results

    def _merge_predictions(self, peptides: pd.DataFrame, results: multiprocessing.pool.AsyncResult):
        spec_id_bufs = []
        peplen_bufs = []
        charge_bufs = []
        mz_bufs = []
        target_bufs = []
        prediction_bufs = []
        vector_bufs = []
        psm_id_bufs = []
        for r in results:
            (
                spec_id_buf,
                peplen_buf,
                charge_buf,
                mz_buf,
                target_buf,
                prediction_buf,
                vector_buf,
                psm_id_buf
            ) = r.get()
            spec_id_bufs.extend(spec_id_buf)
            peplen_bufs.extend(peplen_buf)
            charge_bufs.extend(charge_buf)
            mz_bufs.extend(mz_buf)
            psm_id_bufs.extend(psm_id_buf)
            if target_buf:
                target_bufs.extend(target_buf)
            if prediction_buf:
                prediction_bufs.extend(prediction_buf)
            if vector_buf:
                vector_bufs.extend(vector_buf)

        # Validate number of results
        if not mz_bufs:
            raise exceptions.NoMatchingSpectraFound(
                "No spectra matching titles/IDs from PEPREC could be found in "
                "provided spectrum file."
            )
        logger.debug(f"Gathered data for {len(mz_bufs)} peptides/spectra.")

        # If XGBoost model files are used, first predict outside of MP
        # Temporary hack to move XGB prediction step out of MP; ultimately does not
        # make sense to do this in the `_merge_predictions` step...
        if "xgboost_model_files" in MODELS[self.model].keys():
            logger.debug("Converting feature vectors to XGBoost DMatrix...")
            xgb_vector = xgb.DMatrix(np.vstack(vector_bufs))
            num_ions = [l - 1 for l in peplen_bufs]
            prediction_bufs = get_predictions_xgb(
                xgb_vector,
                num_ions,
                MODELS[self.model],
                self.model_dir,
                num_cpu=self.num_cpu,
            )

        # Reconstruct DataFrame
        logger.debug("Constructing DataFrame with results...")
        num_ion_types = len(MODELS[self.model]["ion_types"])
        ions = []
        ionnumbers = []
        charges = []
        pepids = []
        psm_ids = []
        for pi, pl in enumerate(peplen_bufs):
            [
                ions.extend([ion_type] * (pl - 1))
                for ion_type in MODELS[self.model]["ion_types"]
            ]
            ionnumbers.extend([x + 1 for x in range(pl - 1)] * num_ion_types)
            charges.extend([charge_bufs[pi]] * (num_ion_types * (pl - 1)))
            pepids.extend([spec_id_bufs[pi]] * (num_ion_types * (pl - 1)))
            psm_ids.extend([psm_id_bufs[pi]] * (num_ion_types * (pl - 1)))
        all_preds = pd.DataFrame()
        all_preds["spec_id"] = pepids
        all_preds["charge"] = charges
        all_preds["ion"] = ions
        all_preds["ionnumber"] = ionnumbers
        all_preds["mz"] = np.concatenate(mz_bufs, axis=None)
        all_preds["prediction"] = np.concatenate(prediction_bufs, axis=None)
        if target_bufs:
            all_preds["target"] = np.concatenate(target_bufs, axis=None)
        if "rt" in peptides.columns():
            all_preds = all_preds.merge(
                peptides[["spec_id", "rt"]], on="spec_id", copy=False
            )

        all_preds["psm_id"] = psm_ids

        return all_preds[["psm_id", "spec_id", "charge", "ion", "ionnumber", "mz", "prediction", "target"]]

    def _write_predictions(self, all_preds: pd.DataFrame, peptides: pd.DataFrame, output_filename: str):
        spec_out = spectrum_output.SpectrumOutput(
            all_preds,
            peptides,
            self.params["ms2pip"],
            output_filename=output_filename,
        )
        spec_out.write_results(self.output_formats)

    def _match_spectra(self, results, peptides, spectrum_files=None, sqldb_uri=None):
        mz_bufs, prediction_bufs, _, _, _, pepid_bufs = zip(*(r.get() for r in results))
        match_spectra = MatchSpectra(
            peptides,
            self.mods,
            itertools.chain.from_iterable(pepid_bufs),
            itertools.chain.from_iterable(mz_bufs),
            itertools.chain.from_iterable(prediction_bufs),
        )
        if spectrum_files:
            return match_spectra.match_mgfs(spectrum_files)
        elif sqldb_uri:
            return match_spectra.match_sqldb(sqldb_uri)
        else:
            raise NotImplementedError

    def _write_matched_spectra(self, matched_spectra, output_filename):
        filename = f"{output_filename}_matched_spectra.csv"
        logger.info("Writing file %s...", filename)

        with open(filename, mode="w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(("spec_id", "matched_file" "matched_title"))
            for pep, spec_file, spec in matched_spectra:
                csv_writer.writerow((pep, spec_file, spec["params"]["title"]))
