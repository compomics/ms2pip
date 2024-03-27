#!/usr/bin/env python
from __future__ import annotations

import itertools
import logging
import multiprocessing
import multiprocessing.dummy
import re
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from psm_utils import PSM, Peptidoform, PSMList
from rich.progress import track

import ms2pip.exceptions as exceptions
from ms2pip import spectrum_output
from ms2pip._cython_modules import ms2pip_pyx
from ms2pip._utils.encoder import Encoder
from ms2pip._utils.feature_names import get_feature_names
from ms2pip._utils.psm_input import read_psms
from ms2pip._utils.retention_time import RetentionTime
from ms2pip._utils.xgb_models import get_predictions_xgb, validate_requested_xgb_model
from ms2pip.constants import MODELS, SUPPORTED_OUTPUT_FORMATS
from ms2pip.result import ProcessingResult, calculate_correlations
from ms2pip.spectrum_input import read_spectrum_file

logger = logging.getLogger(__name__)


def predict_single(
    peptidoform: Union[Peptidoform, str],
    model: Optional[str] = "HCD",
    model_dir: Optional[Union[str, Path]] = None,
) -> ProcessingResult:
    """
    Predict fragmentation spectrum for a single peptide.\f
    """
    if isinstance(peptidoform, str):
        peptidoform = Peptidoform(peptidoform)
    psm = PSM(peptidoform=peptidoform, spectrum_id=0)
    model_dir = model_dir if model_dir else Path.home() / ".ms2pip"
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    with Encoder.from_peptidoform(peptidoform) as encoder:
        ms2pip_pyx.ms2pip_init(*encoder.encoder_files)
        result = _process_peptidoform(0, psm, model, encoder, ion_types=ion_types)

        if "xgboost_model_files" in MODELS[model].keys():
            validate_requested_xgb_model(
                MODELS[model]["xgboost_model_files"],
                MODELS[model]["model_hash"],
                model_dir,
            )
            predictions = np.array(
                get_predictions_xgb(
                    result.feature_vectors,
                    [len(peptidoform.parsed_sequence) - 1],
                    MODELS[model],
                    model_dir,
                )
            )
            result.predicted_intensity = predictions[0]  # Only one spectrum in predictions
            result.feature_vectors = None

    return result


def predict_batch(
    psms: Union[PSMList, str, Path],
    add_retention_time: bool = False,
    psm_filetype: Optional[str] = None,
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
    psm_filetype
        Filetype of the PSM file. By default, None. Should be one of the supported psm_utils
        filetypes. See https://psm-utils.readthedocs.io/en/stable/#supported-file-formats.
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
    psm_list = read_psms(psms, filetype=psm_filetype)

    if add_retention_time:
        logger.info("Adding retention time predictions")
        rt_predictor = RetentionTime(processes=processes)
        rt_predictor.add_rt_predictions(psm_list)

    with Encoder.from_psm_list(psm_list) as encoder:
        ms2pip_parallelized = _Parallelized(
            encoder=encoder,
            model=model,
            model_dir=model_dir,
            processes=processes,
        )
        logger.info("Processing peptides...")
        results = ms2pip_parallelized.process_peptides(psm_list)

    return results


def predict_library():
    """Predict spectral library from protein FASTA file."""
    raise NotImplementedError


def correlate(
    psms: Union[PSMList, str, Path],
    spectrum_file: Union[str, Path],
    psm_filetype: Optional[str] = None,
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
    psm_filetype
        Filetype of the PSM file. By default, None. Should be one of the supported psm_utils
        filetypes. See https://psm-utils.readthedocs.io/en/stable/#supported-file-formats.
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
    psm_list = read_psms(psms, filetype=psm_filetype)
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    if add_retention_time:
        logger.info("Adding retention time predictions")
        rt_predictor = RetentionTime(processes=processes)
        rt_predictor.add_rt_predictions(psm_list)

    with Encoder.from_psm_list(psm_list) as encoder:
        ms2pip_parallelized = _Parallelized(
            encoder=encoder,
            model=model,
            model_dir=model_dir,
            ms2_tolerance=ms2_tolerance,
            processes=processes,
        )
        logger.info("Processing spectra and peptides...")
        results = ms2pip_parallelized.process_spectra(psm_list, spectrum_file, spectrum_id_pattern)

    # Correlations also requested
    if compute_correlations:
        logger.info("Computing correlations")
        calculate_correlations(results)
        logger.info(f"Median correlation: {np.median(list(r.correlation for r in results))}")

    return results


def get_training_data(
    psms: Union[PSMList, str, Path],
    spectrum_file: Union[str, Path],
    psm_filetype: Optional[str] = None,
    spectrum_id_pattern: Optional[str] = None,
    model: Optional[str] = "HCD",
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
    psm_filetype
        Filetype of the PSM file. By default, None. Should be one of the supported psm_utils
        filetypes. See https://psm-utils.readthedocs.io/en/stable/#supported-file-formats.
    spectrum_id_pattern
        Regular expression pattern to apply to spectrum titles before matching to
        peptide file ``spec_id`` entries.
    model
        Model to use as reference for the ion types that are extracted from the observed spectra.
        Default: "HCD", which results in the extraction of singly charged b- and y-ions.
    ms2_tolerance
        MS2 tolerance in Da for observed spectrum peak annotation. By default, 0.02 Da.
    processes
        Number of parallel processes for multiprocessing steps. By default, all available.

    Returns
    -------
    features
        :py:class:`pandas.DataFrame` with feature vectors and targets.

    """
    psm_list = read_psms(psms, filetype=psm_filetype)
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    with Encoder.from_psm_list(psm_list) as encoder:
        ms2pip_parallelized = _Parallelized(
            encoder=encoder,
            model=model,
            ms2_tolerance=ms2_tolerance,
            processes=processes,
        )
        logger.info("Processing spectra and peptides...")
        results = ms2pip_parallelized.process_spectra(
            psm_list, spectrum_file, spectrum_id_pattern, vector_file=True
        )

        logger.info("Assembling training data in DataFrame...")
        training_data = _assemble_training_data(results, model)

    return training_data


def annotate_spectra(
    psms: Union[PSMList, str, Path],
    spectrum_file: Union[str, Path],
    psm_filetype: Optional[str] = None,
    spectrum_id_pattern: Optional[str] = None,
    model: Optional[str] = "HCD",
    ms2_tolerance: float = 0.02,
    processes: Optional[int] = None,
):
    """
    Annotate observed spectra.\f

    Parameters
    ----------
    psms
        PSMList or path to PSM file that is supported by psm_utils.
    spectrum_file
        Path to spectrum file with target intensities.
    psm_filetype
        Filetype of the PSM file. By default, None. Should be one of the supported psm_utils
        filetypes. See https://psm-utils.readthedocs.io/en/stable/#supported-file-formats.
    spectrum_id_pattern
        Regular expression pattern to apply to spectrum titles before matching to
        peptide file ``spec_id`` entries.
    model
        Model to use as reference for the ion types that are extracted from the observed spectra.
        Default: "HCD", which results in the extraction of singly charged b- and y-ions.
    ms2_tolerance
        MS2 tolerance in Da for observed spectrum peak annotation. By default, 0.02 Da.
    processes
        Number of parallel processes for multiprocessing steps. By default, all available.

    Returns
    -------
    results: List[ProcessingResult]
        List of ProcessingResult objects with theoretical m/z and observed intensity values.

    """
    psm_list = read_psms(psms, filetype=psm_filetype)
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    with Encoder.from_psm_list(psm_list) as encoder:
        ms2pip_parallelized = _Parallelized(
            encoder=encoder,
            model=model,
            ms2_tolerance=ms2_tolerance,
            processes=processes,
        )
        logger.info("Processing spectra and peptides...")
        results = ms2pip_parallelized.process_spectra(
            psm_list, spectrum_file, spectrum_id_pattern, vector_file=False, annotations_only=True
        )

    return results


def download_models(
    models: Optional[List[str]] = None, model_dir: Optional[Union[str, Path]] = None
):
    """
    Download all specified models to the specified directory.

    Parameters
    ----------
    models
        List of models to download. If not specified, all models will be downloaded.
    model_dir
        Directory where XGBoost model files are to be stored. Default: ``~/.ms2pip``.

    """
    model_dir = model_dir if model_dir else Path.home() / ".ms2pip"
    model_dir = Path(model_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    if not models:
        models = list(MODELS.keys())

    for model in models:
        try:
            if "xgb_model_files" in MODELS[model].keys():
                continue
        except KeyError:
            raise exceptions.UnknownModelError(model)
        logger.debug("Downloading %s model files", model)
        validate_requested_xgb_model(
            MODELS[model]["xgboost_model_files"],
            MODELS[model]["model_hash"],
            model_dir,
        )


class _Parallelized:
    """Implementations of common multiprocessing functionality across MS²PIP usage modes."""

    def __init__(
        self,
        encoder: Encoder = None,
        model: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
        ms2_tolerance: float = 0.02,
        processes: Optional[int] = None,
    ):
        """
        Implementations of common multiprocessing functionality across MS²PIP usage modes.

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
        self.encoder = encoder
        self.model = model
        self.model_dir = model_dir if model_dir else Path.home() / ".ms2pip"
        self.ms2_tolerance = ms2_tolerance
        self.processes = processes if processes else multiprocessing.cpu_count()

        # Setup encoder if not configured
        if not self.encoder:
            self.encoder = Encoder()
            self.encoder.write_encoder_files()

        # Validate requested model
        if self.model in MODELS.keys():
            logger.debug("Using %s model", self.model)
            if "xgboost_model_files" in MODELS[self.model].keys():
                validate_requested_xgb_model(
                    MODELS[self.model]["xgboost_model_files"],
                    MODELS[self.model]["model_hash"],
                    self.model_dir,
                )
        else:
            raise exceptions.UnknownModelError(self.model)

    def _get_pool(self):
        """Get multiprocessing pool."""
        logger.debug(f"Starting workers (processes={self.processes})...")
        if multiprocessing.current_process().daemon:
            logger.warn(
                "MS²PIP is running in a daemon process. Disabling multiprocessing as daemonic "
                "processes cannot have children."
            )
            return multiprocessing.dummy.Pool(1)
        elif self.processes == 1:
            logger.debug("Using dummy multiprocessing pool.")
            return multiprocessing.dummy.Pool(1)
        else:
            return multiprocessing.get_context("spawn").Pool(self.processes)

    def _validate_output_formats(self, output_formats: List[str]) -> List[str]:
        """Validate requested output formats."""
        if not output_formats:
            self.output_formats = ["csv"]
        else:
            for output_format in output_formats:
                if output_format not in SUPPORTED_OUTPUT_FORMATS:
                    raise exceptions.UnknownOutputFormatError(output_format)
            self.output_formats = output_formats

    def _execute_in_pool(self, psm_list: PSMList, func: Callable, args: tuple):
        """Execute function in multiprocessing pool."""

        def get_chunk_size(n_items, n_processes):
            """Get optimal chunk size for multiprocessing."""
            if n_items < 5000:
                return n_items
            else:
                max_chunk_size = 50000
                n_chunks = ceil(ceil(n_items / n_processes) / max_chunk_size) * n_processes
                return ceil(n_items / n_chunks)

        def to_chunks(_list, chunk_size):
            """Split _list into chunks of size chunk_size."""

            def _generate_chunks():
                for i in range(0, len(_list), chunk_size):
                    yield _list[i : i + chunk_size]

            _list = list(_list)
            return list(_generate_chunks())

        def _enumerated_psm_list_by_spectrum_id(psm_list, spectrum_ids_chunk):
            selected_indices = np.flatnonzero(np.isin(psm_list["spectrum_id"], spectrum_ids_chunk))
            return [(i, psm_list.psm_list[i]) for i in selected_indices]

        with self._get_pool() as pool:
            if not psm_list:
                logger.warning("No PSMs to process.")
                return []

            # Split PSMList into chunks
            if func == _process_spectra:
                # Split by spectrum_id to keep PSMs for same spectrum together
                spectrum_ids = set(psm_list["spectrum_id"])
                chunk_size = get_chunk_size(len(spectrum_ids), pool._processes)
                chunks = [
                    _enumerated_psm_list_by_spectrum_id(psm_list, spectrum_ids_chunk)
                    for spectrum_ids_chunk in to_chunks(spectrum_ids, chunk_size)
                ]
            else:
                # Simple split by PSM
                chunk_size = get_chunk_size(len(psm_list), pool._processes)
                chunks = to_chunks(list(enumerate(psm_list)), chunk_size)

            logger.debug(f"Processing {len(chunks)} chunk(s) of ~{chunk_size} entries each.")

            # Add jobs to pool
            mp_results = []
            for psm_list_chunk in chunks:
                mp_results.append(pool.apply_async(func, args=(psm_list_chunk, *args)))

            # Gather results
            # results = [
            #     r.get()
            #     for r in track(
            #         mp_results,
            #         disable=len(chunks) == 1,
            #         description="Processing chunks...",
            #         transient=True,
            #         show_speed=False,
            #     )
            # ]
            results = [r.get() for r in mp_results]

        # Sort results by input order
        results = list(
            sorted(
                itertools.chain.from_iterable(results),
                key=lambda result: result.psm_index,
            )
        )

        return results

    def process_peptides(self, psm_list: PSMList) -> List[ProcessingResult]:
        """Process peptides in parallel."""
        results = self._execute_in_pool(
            psm_list,
            _process_peptides,
            (self.encoder, self.model),
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
        annotations_only: bool = False,
    ) -> List[ProcessingResult]:
        """
        Process PSMs and observed spectra in parallel

        Parameters
        ----------
        psm_list
            psm_utils.PSMList instance with PSMs to process
        spectrum_file
            Filename of spectrum file
        spectrum_id_pattern
            Regular expression pattern to apply to spectrum titles before matching to
            peptide file entries
        vector_file
            If feature vectors should be extracted instead of predictions
        annotations_only
            If only peak annotations should be extracted from the spectrum file

        """
        args = (
            spectrum_file,
            vector_file,
            self.encoder,
            self.model,
            self.ms2_tolerance,
            spectrum_id_pattern,
            annotations_only,
        )
        results = self._execute_in_pool(psm_list, _process_spectra, args)

        # Validate number of results
        if not results:
            raise exceptions.NoMatchingSpectraFound(
                "No spectra matching spectrum IDs from PSM list could be found in provided file."
            )
        logger.debug(f"Gathered data for {len(results)} PSMs.")

        # Add XGBoost predictions if required
        if (
            not (vector_file or annotations_only)
            and "xgboost_model_files" in MODELS[self.model].keys()
        ):
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

        if "xgboost_model_files" not in MODELS[self.model].keys():
            raise ValueError("XGBoost model files not found in MODELS dictionary.")

        logger.debug("Converting feature vectors to XGBoost DMatrix...")
        import xgboost as xgb

        results_to_predict = [r for r in results if r.feature_vectors is not None]

        if not results_to_predict:
            return results

        num_ions = [len(r.psm.peptidoform.parsed_sequence) - 1 for r in results_to_predict]
        xgb_vector = xgb.DMatrix(np.vstack(list(r.feature_vectors for r in results_to_predict)))

        predictions = get_predictions_xgb(
            xgb_vector,
            num_ions,
            MODELS[self.model],
            self.model_dir,
            processes=self.processes,
        )

        logger.debug("Adding XGBoost predictions to results...")
        for result, preds in zip(results_to_predict, predictions):
            result.predicted_intensity = preds
            result.feature_vectors = None

        return results

    # TODO IMPLEMENT
    def write_predictions(
        self, all_preds: pd.DataFrame, peptides: pd.DataFrame, output_filename: str
    ):
        raise NotImplementedError
        spec_out = spectrum_output.SpectrumOutput(
            all_preds,
            peptides,
            self.params["ms2pip"],
            output_filename=output_filename,
        )
        spec_out.write_results(self.output_formats)


def _process_peptidoform(
    psm_index: int,
    psm: PSM,
    model: str,
    encoder: Encoder,
    ion_types: Optional[List[str]] = None,
) -> ProcessingResult:
    """
    Process a single peptidoform from a PSM.

    Get theoretical m/z and predicted intensities (from C model) or feature vectors (for XGBoost
    model) for a single peptidoform from a PSM.

    Notes
    -----
    - ``ms2pip_pyx.init()`` must be called before this function is called.
    - Optionally, lowercase version of ``ion_types`` from the model configuration can be provided
    to save computational time.

    """
    peptidoform = psm.peptidoform
    if not ion_types:
        ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    enc_peptide = encoder.encode_peptide(peptidoform)
    enc_peptidoform = encoder.encode_peptidoform(peptidoform)

    # Get ion mzs and map to ion types
    mz = ms2pip_pyx.get_mzs(enc_peptidoform, MODELS[model]["peaks_version"])
    mz = {i: np.array(mz, dtype=np.float32) for i, mz in zip(ion_types, mz)}

    # Get predictions from XGBoost models.
    if "xgboost_model_files" in MODELS[model].keys():
        predictions = None
        feature_vectors = np.array(
            ms2pip_pyx.get_vector(enc_peptide, enc_peptidoform, peptidoform.precursor_charge),
            dtype=np.uint16,
        )
    # Or get predictions from C models.
    else:
        predictions = ms2pip_pyx.get_predictions(
            enc_peptide,
            enc_peptidoform,
            peptidoform.precursor_charge,
            MODELS[model]["id"],
            MODELS[model]["peaks_version"],
            30.0,  # TODO: Remove CE feature
        )
        predictions = {i: np.array(p, dtype=np.float32) for i, p in zip(ion_types, predictions)}
        feature_vectors = None

    return ProcessingResult(
        psm_index=psm_index,
        psm=psm,
        theoretical_mz=mz,
        predicted_intensity=predictions,
        observed_intensity=None,
        feature_vectors=feature_vectors,
    )


def _process_peptides(
    enumerated_psm_list: List[Tuple[int, PSM]],
    encoder: Encoder,
    model: str,
) -> List[ProcessingResult]:
    """
    Predict spectrum for each entry in PeptideRecord DataFrame.

    Parameters
    ----------
    enumerated_psm_list
        List of tuples of (index, PSM) for each PSM in the input file.
    encoder
        Configured encoder to use for peptide and peptidoform encoding
    model
        Name of prediction model to be used

    """
    ms2pip_pyx.ms2pip_init(*encoder.encoder_files)
    results = []
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    for psm_index, psm in enumerated_psm_list:
        try:
            result = _process_peptidoform(psm_index, psm, model, encoder, ion_types)
        except (
            exceptions.InvalidPeptidoformError,
            exceptions.InvalidAminoAcidError,
        ):
            result = ProcessingResult(psm_index=psm_index, psm=psm)
        results.append(result)

    return results


def _process_spectra(
    enumerated_psm_list: List[Tuple[int, PSM]],
    spec_file: str,
    vector_file: bool,
    encoder: Encoder,
    model: str,
    ms2_tolerance: float,
    spectrum_id_pattern: str,
    annotations_only: bool = False,
) -> List[ProcessingResult, None]:
    """
    Perform requested tasks for each spectrum in spectrum file.

    Parameters
    ----------
    enumerated_psm_list
        List of tuples of (index, PSM) for each PSM in the input file.
    spec_file
        Filename of spectrum file
    vector_file
        If feature vectors should be extracted instead of predictions
    encoder: Encoder
        Configured encoder to use for peptide and peptidoform encoding
    model
        Name of prediction model to be used
    ms2_tolerance
        Fragmentation spectrum m/z error tolerance in Dalton
    spectrum_id_pattern
        Regular expression pattern to apply to spectrum titles before matching to
        peptide file entries
    annotations_only
        If only peak annotations should be extracted from the spectrum file

    """
    ms2pip_pyx.ms2pip_init(*encoder.encoder_files)
    results = []
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    try:
        spectrum_id_regex = re.compile(spectrum_id_pattern)
    except TypeError:
        spectrum_id_regex = re.compile(r"(.*)")

    # Restructure PeptideRecord entries as spec_id -> [(id, psm_1), (id, psm_2), ...]
    psms_by_specid = defaultdict(list)
    for psm_index, psm in enumerated_psm_list:
        psms_by_specid[str(psm.spectrum_id)].append((psm_index, psm))

    for spectrum in read_spectrum_file(spec_file):
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

        for psm_index, psm in psms_by_specid[spectrum_id]:
            try:
                enc_peptidoform = encoder.encode_peptidoform(psm.peptidoform)
            except exceptions.InvalidAminoAcidError:
                result = ProcessingResult(psm_index=psm_index, psm=psm)
                results.append(result)
                continue

            targets = ms2pip_pyx.get_targets(
                enc_peptidoform,
                spectrum.mz.astype(np.float32),
                spectrum.intensity.astype(np.float32),
                float(ms2_tolerance),
                MODELS[model]["peaks_version"],
            )
            targets = {i: np.array(t, dtype=np.float32) for i, t in zip(ion_types, targets)}

            if not psm.peptidoform.precursor_charge:
                psm.peptidoform.precursor_charge = spectrum.precursor_charge

            if vector_file:
                enc_peptide = encoder.encode_peptide(psm.peptidoform)
                feature_vectors = np.array(
                    ms2pip_pyx.get_vector(
                        enc_peptide, enc_peptidoform, psm.peptidoform.precursor_charge
                    ),
                    dtype=np.uint16,
                )
                result = ProcessingResult(
                    psm_index=psm_index,
                    psm=psm,
                    theoretical_mz=None,
                    predicted_intensity=None,
                    observed_intensity=targets,
                    correlation=None,
                    feature_vectors=feature_vectors,
                )

            elif annotations_only:
                # Only return mz and targets
                mz = ms2pip_pyx.get_mzs(enc_peptidoform, MODELS[model]["peaks_version"])
                mz = {i: np.array(mz, dtype=np.float32) for i, mz in zip(ion_types, mz)}

                result = ProcessingResult(
                    psm_index=psm_index,
                    psm=psm,
                    theoretical_mz=mz,
                    predicted_intensity=None,
                    observed_intensity=targets,
                    correlation=None,
                    feature_vectors=None,
                )

            else:
                # Predict with C model or get feature vectors for XGBoost
                try:
                    result = _process_peptidoform(psm_index, psm, model, encoder, ion_types)
                except (
                    exceptions.InvalidPeptidoformError,
                    exceptions.InvalidAminoAcidError,
                ):
                    result = ProcessingResult(psm_index=psm_index, psm=psm)
                else:
                    result.observed_intensity = targets

            results.append(result)

    return results


def _assemble_training_data(results: List[ProcessingResult], model: str) -> pd.DataFrame:
    """Assemble training data from results list to single pandas DataFrame."""
    # Get ion types
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    # Assemble feature vectors, PSM indices, and targets
    training_data = pd.DataFrame(
        np.vstack([r.feature_vectors for r in results if r.feature_vectors is not None]),
        columns=get_feature_names(),
    )
    training_data["psm_index"] = np.concatenate(
        [
            np.repeat(r.psm_index, r.feature_vectors.shape[0])
            for r in results
            if r.feature_vectors is not None
        ]
    )
    for ion_type in ion_types:
        if ion_type in ["a", "b", "b2", "c"]:
            training_data[f"target_{ion_type}"] = np.concatenate(
                [r.observed_intensity[ion_type] for r in results if r.feature_vectors is not None]
            )
        elif ion_type in ["x", "y", "y2", "z"]:
            training_data[f"target_{ion_type}"] = np.concatenate(
                [
                    r.observed_intensity[ion_type][::-1]
                    for r in results
                    if r.feature_vectors is not None
                ]
            )

    # Reorder columns
    training_data = training_data[
        ["psm_index"] + get_feature_names() + [f"target_{it}" for it in ion_types]
    ]

    return training_data
