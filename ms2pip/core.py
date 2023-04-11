#!/usr/bin/env python
from __future__ import annotations

import csv
import glob
import itertools
import logging
import multiprocessing
import multiprocessing.dummy
import os
from pathlib import Path
from random import shuffle
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from psm_utils import PSMList

import ms2pip.exceptions as exceptions
from ms2pip import spectrum_output
from ms2pip._utils.batch_processing import process_peptides, process_spectra
from ms2pip._utils.encoder import Encoder
from ms2pip._utils.match_spectra import MatchSpectra
from ms2pip._utils.psm_input import read_psms
from ms2pip._utils.retention_time import RetentionTime
from ms2pip._utils.xgb_models import get_predictions_xgb, validate_requested_xgb_model
from ms2pip.constants import MODELS, SUPPORTED_OUTPUT_FORMATS
from ms2pip.result import ProcessingResult, calculate_correlations

logger = logging.getLogger(__name__)


def predict_single():
    """
    Predict fragmentation spectrum for a single peptide.\f
    """
    pass


def predict_batch(
    psms: Union[PSMList, str, Path],
    add_retention_time: bool = False,
    model: Optional[str] = "HCD",
    model_dir: Optional[Union[str, Path]] = None,
    processes: Optional[int] = None,
) -> List[ProcessingResult]:
    """
    Predict fragmentation spectra for a batch of peptides.\f

    Parameters
    ----------
    psms
        PSMList or path to PSM file that is supported by psm_utils.
    add_retention_time
        Add retention time predictions with DeepLC (Requires optional DeepLC dependency).
    model
        Model to use for prediction. Default: "HCD".
    model_dir
        Directory where XGBoost model files are stored. Default: `~/.ms2pip`.
    processes
        Number of parallel processes for multiprocessing steps. By default, all available.

    Returns
    -------
    predictions: List[ProcessingResult]
        Predicted spectra with theoretical m/z and predicted intensity values.

    """
    psm_list, encoder = read_psms(psms)

    if add_retention_time:
        logger.info("Adding retention time predictions")
        rt_predictor = RetentionTime(processes=processes)
        rt_predictor.add_rt_predictions(psm_list)

    with _Core(
        encoder=encoder,
        model=model,
        model_dir=model_dir,
        processes=processes,
    ) as ms2pip_core:
        logger.info("Processing peptides...")
        results = ms2pip_core.process_peptides(psm_list)

    return results


def predict_library():
    """Predict spectral library from protein FASTA file."""
    pass


def correlate(
    psms: Union[PSMList, str, Path],
    spectrum_file: Union[str, Path],
    spectrum_id_pattern: Optional[str] = None,
    compute_correlations: bool = False,
    add_retention_time: bool = False,
    model: Optional[str] = "HCD",
    model_dir: Optional[Union[str, Path]] = None,
    ms2_tolerance: float = 0.02,
    processes: Optional[int] = None,
) -> List[ProcessingResult]:
    """
    Compare predicted and observed intensities and optionally compute correlations.\f

    Parameters
    ----------
    psms
        PSMList or path to PSM file that is supported by psm_utils.
    spectrum_file
        Path to spectrum file with target intensities.
    spectrum_id_pattern
        Regular expression pattern to apply to spectrum titles before matching to
        peptide file ``spec_id`` entries.
    compute_correlations
        Compute correlations between predictions and targets.
    add_retention_time
        Add retention time predictions with DeepLC (Requires optional DeepLC dependency).
    model
        Model to use for prediction. Default: "HCD".
    model_dir
        Directory where XGBoost model files are stored. Default: `~/.ms2pip`.
    ms2_tolerance
        MS2 tolerance in Da for observed spectrum peak annotation. By default, 0.02 Da.
    processes
        Number of parallel processes for multiprocessing steps. By default, all available.

    Returns
    -------
    results: List[ProcessingResult]
        Predicted spectra with theoretical m/z and predicted intensity values, and optionally,
        correlations.

    """
    psm_list, encoder = read_psms(psms)
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    if add_retention_time:
        logger.info("Adding retention time predictions")
        rt_predictor = RetentionTime(processes=processes)
        rt_predictor.add_rt_predictions(psm_list)

    with _Core(
        encoder=encoder,
        model=model,
        model_dir=model_dir,
        ms2_tolerance=ms2_tolerance,
        processes=processes,
    ) as ms2pip_core:
        logger.info("Processing spectra and peptides...")
        results = ms2pip_core.process_spectra(psm_list, spectrum_file, spectrum_id_pattern)

    # Correlations also requested
    if compute_correlations:
        logger.info("Computing correlations")
        calculate_correlations(results)
        logger.info(f"Median correlation: {np.median(list(r.correlation for r in results))}")

    return results


def get_training_data(
    psms: Union[PSMList, str, Path],
    spectrum_file: Union[str, Path],
    spectrum_id_pattern: Optional[str] = None,
    ms2_tolerance: float = 0.02,
    processes: Optional[int] = None,
):
    """
    Extract feature vectors and target intensities from observed spectra for training.\f

    Parameters
    ----------
    psms
        PSMList or path to PSM file that is supported by psm_utils.
    spectrum_file
        Path to spectrum file with target intensities.
    spectrum_id_pattern
        Regular expression pattern to apply to spectrum titles before matching to
        peptide file ``spec_id`` entries.
    ms2_tolerance
        MS2 tolerance in Da for observed spectrum peak annotation. By default, 0.02 Da.
    processes
        Number of parallel processes for multiprocessing steps. By default, all available.

    Returns
    -------
    features
        :py:class:`pandas.DataFrame` with feature vectors and targets.

    """
    psm_list, encoder = read_psms(psms)
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    with _Core(
        encoder=encoder,
        ms2_tolerance=ms2_tolerance,
        processes=processes,
    ) as ms2pip_core:
        logger.info("Processing spectra and peptides...")
        results = ms2pip_core.process_spectra(
            psm_list, spectrum_file, spectrum_id_pattern, vector_file=True
        )
        logger.debug("Merging results")
        vectors = ms2pip_core.write_vector_file(results)

    return vectors


def match_spectra(
    psms: Union[PSMList, str, Path],
    spectrum_file: Union[str, Path],
    sqldb_uri: str,
    model: Optional[str] = "HCD",
    model_dir: Optional[Union[str, Path]] = None,
    ms2_tolerance: float = 0.02,
    processes: Optional[int] = None,
):
    """
    Match spectra to peptides based on peak intensities (experimental).\f

    Match spectra in `spectrum_file` or `sqldb_uri` to peptides in `pep_file` based on
    predicted intensities.

    Parameters
    ----------
    psms
        PSMList or path to PSM file that is supported by psm_utils.
    spectrum_file
        Path to spectrum file or directory with spectrum files.
    sqldb_uri
        URI to prebuilt SQL database with spectra.
    model
        Model to use for prediction. Default: "HCD".
    model_dir
        Directory where XGBoost model files are stored. Default: `~/.ms2pip`.
    ms2_tolerance
        MS2 tolerance in Da for observed spectrum peak annotation. By default, 0.02 Da.
    processes
        Number of parallel processes for multiprocessing steps. By default, all available.

    """
    psm_list, encoder = read_psms(psms)

    # Set spec_files based on spec_file or sqldb_uri
    if sqldb_uri:
        spectrum_files = None
    elif os.path.isdir(spectrum_file):
        spectrum_files = glob.glob("{}/*.mgf".format(spectrum_file))
    else:
        spectrum_files = [spectrum_file]
    logger.debug("Using spectrum files %s", spectrum_files)

    # Process
    with _Core(
        encoder=encoder,
        model=model,
        model_dir=model_dir,
        ms2_tolerance=ms2_tolerance,
        processes=processes,
    ) as ms2pip_core:
        logger.info("Processing spectra and peptides...")
        results = ms2pip_core.process_peptides(psm_list)
        logger.debug("Matching spectra")
        matched_spectra = ms2pip_core.match_spectra(
            results, psm_list, spectrum_files, sqldb_uri
        )  # TODO IMPLEMENT
        logger.debug("Writing results")
    return matched_spectra


class _Core:
    """MS²PIP core class implementing common functionality accross usage modes."""

    def __init__(
        self,
        encoder: Encoder = None,
        model: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
        ms2_tolerance: float = 0.02,
        processes: Optional[int] = None,
    ):
        """
        MS²PIP core class.

        Parameters
        ----------
        encoding
            Configured encoding class instance. Required if input peptides contain modifications.
        model
            Name of the model to use for predictions. Overrides configuration file.
        model_dir
            Custom directory for downloaded XGBoost model files. By default, `~/.ms2pip` is used.
        ms2_tolerance
            MS2 tolerance in Da for observed spectrum peak annotation. By default, 0.02 Da.
        processes
            Number of parallel processes for multiprocessing steps. By default, all available.

        """
        # Input parameters
        self.encoder = encoder if encoder else Encoder()
        self.model = model
        self.model_dir = model_dir if model_dir else Path.home() / ".ms2pip"
        self.ms2_tolerance = ms2_tolerance
        self.processes = processes if processes else multiprocessing.cpu_count()

        # Instance variables
        self.mp_pool = None

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
            raise exceptions.UnknownModelError(self.model)

        # Set up multiprocessing
        self._setup_multiprocessing()

        # Set up modifications and write to files for C-code
        self.encoder.write_encoding_configuration()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.encoder.cleanup()

    def _setup_multiprocessing(self):
        """Setup multiprocessing."""
        logger.debug(f"Starting workers (processes={self.processes})...")
        if multiprocessing.current_process().daemon:
            logger.warn(
                "MS²PIP is running in a daemon process. Disabling multiprocessing as daemonic "
                "processes cannot have children."
            )
            self.mp_pool = multiprocessing.dummy.Pool(1)
        elif self.processes == 1:
            logger.debug("Using dummy multiprocessing pool.")
            self.mp_pool = multiprocessing.dummy.Pool(1)
        else:
            self.mp_pool = multiprocessing.Pool(self.processes)

    def _validate_output_formats(self, output_formats: List[str]) -> List[str]:
        """Validate requested output formats."""
        if not output_formats:
            self.output_formats = ["csv"]
        else:
            for output_format in output_formats:
                if output_format not in SUPPORTED_OUTPUT_FORMATS:
                    raise exceptions.UnknownOutputFormatError(output_format)
            self.output_formats = output_formats

    @staticmethod
    def _prepare_titles(titles, processes: int):
        """Split list of spec_ids over number of CPUs."""
        shuffle(titles)  # Shuffling to improve parallel speeds
        split_titles = [
            titles[i * len(titles) // processes : (i + 1) * len(titles) // processes]
            for i in range(processes)
        ]
        logger.debug(
            "{} spectra (~{:.0f} per cpu)".format(
                len(titles), np.mean([len(a) for a in split_titles])
            )
        )
        return split_titles

    def _execute_in_pool(self, psm_list: PSMList, func: Callable, args: tuple):
        split_spec_ids = self._prepare_titles(list(psm_list["spectrum_id"]), self.processes)
        results = []
        for i in range(self.processes):
            results.append(
                self.mp_pool.apply_async(
                    func,
                    args=(i, psm_list[np.isin(psm_list["spectrum_id"], split_spec_ids[i])], *args),
                )
            )
        self.mp_pool.close()
        self.mp_pool.join()
        return results

    def process_peptides(self, psm_list: PSMList) -> List[ProcessingResult]:
        """Process PSMs in parallel."""
        # Process peptides in parallel
        mp_results = self._execute_in_pool(
            psm_list,
            process_peptides,
            (self.encoder, self.model),
        )
        results = list(itertools.chain.from_iterable([r.get() for r in mp_results]))

        # Validate number of results
        if not results:
            raise exceptions.NoValidPeptideSequencesError(
                "No valid peptides were found in the input file."
            )
        logger.debug(f"Gathered data for {len(results)} peptides.")

        # Add XGBoost predictions if required
        if "xgboost_model_files" in MODELS[self.model].keys():
            results = self._add_xgboost_predictions(results)

        return results

    def process_spectra(
        self,
        psm_list: PSMList,
        spectrum_file: Union[str, Path],
        spectrum_id_pattern: str,
        vector_file: bool = False,
    ) -> List[ProcessingResult]:
        """Process PSMs and observed spectra in parallel."""
        # Process spectra in parallel
        args = (
            spectrum_file,
            vector_file,
            self.encoder,
            self.model,
            self.ms2_tolerance,
            spectrum_id_pattern,
        )
        mp_results = self._execute_in_pool(psm_list, process_spectra, args)
        results = list(itertools.chain.from_iterable([r.get() for r in mp_results]))

        # Validate number of results
        if not results:
            raise exceptions.NoMatchingSpectraFound(
                "No spectra matching spectrum IDs from PSM list could be found in provided file."
            )
        logger.debug(f"Gathered data for {len(results)} PSMs.")

        # Add XGBoost predictions if required
        if not vector_file and "xgboost_model_files" in MODELS[self.model].keys():
            results = self._add_xgboost_predictions(results)

        return results

    def _add_xgboost_predictions(self, results: List[ProcessingResult]) -> List[ProcessingResult]:
        """
        Add XGBoost predictions to results.

        Notes
        -----
        This functions is applied after the parallel processing, as XGBoost implements its own
        multiprocessing.
        """

        if not "xgboost_model_files" in MODELS[self.model].keys():
            raise ValueError("XGBoost model files not found in MODELS dictionary.")

        logger.debug("Converting feature vectors to XGBoost DMatrix...")
        import xgboost as xgb

        xgb_vector = xgb.DMatrix(np.vstack(list(r.feature_vectors for r in results)))
        num_ions = [len(r.psm.peptidoform.parsed_sequence) - 1 for r in results]

        predictions = get_predictions_xgb(
            xgb_vector,
            num_ions,
            MODELS[self.model],
            self.model_dir,
            processes=self.processes,
        )

        logger.debug("Adding XGBoost predictions to results...")
        for r, preds in zip(results, predictions):
            r.predicted_intensity = preds
            r.feature_vectors = None

        return results

    # TODO IMPLEMENT
    def write_vector_file(self, results: List[ProcessingResult]):
        all_results = []
        for r in results:
            psmids, df, dtargets = r.get()

            # dtargets is a dict, containing targets for every ion type (keys are int)
            for i, t in dtargets.items():
                df["targets_{}".format(MODELS[self.model]["ion_types"][i])] = np.concatenate(
                    t, axis=None
                )
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

    # TODO IMPLEMENT
    def write_predictions(
        self, all_preds: pd.DataFrame, peptides: pd.DataFrame, output_filename: str
    ):
        spec_out = spectrum_output.SpectrumOutput(
            all_preds,
            peptides,
            self.params["ms2pip"],
            output_filename=output_filename,
        )
        spec_out.write_results(self.output_formats)

    def match_spectra(self, results, peptides, spectrum_files=None, sqldb_uri=None):
        psm_id_bufs, _, _, _, mz_bufs, _, prediction_bufs, _ = zip(*(r.get() for r in results))

        match_spectra = MatchSpectra(
            peptides,
            self.mods,
            itertools.chain.from_iterable(psm_id_bufs),
            itertools.chain.from_iterable(mz_bufs),
            itertools.chain.from_iterable(prediction_bufs),
        )
        if spectrum_files:
            return match_spectra.match_mgfs(spectrum_files, max_error=self.ms2_tolerance)
        elif sqldb_uri:
            return match_spectra.match_sqldb(sqldb_uri, max_error=self.ms2_tolerance)
        else:
            raise NotImplementedError

    def write_matched_spectra(self, matched_spectra, output_filename):
        filename = f"{output_filename}_matched_spectra.csv"
        logger.info("Writing file %s...", filename)

        with open(filename, mode="w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(("spec_id", "matched_file" "matched_title"))
            for pep, spec_file, spec in matched_spectra:
                csv_writer.writerow((pep, spec_file, spec["params"]["title"]))
