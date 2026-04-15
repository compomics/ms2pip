#!/usr/bin/env python
from __future__ import annotations

import logging
import os
from collections.abc import Generator
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from ms2rescore_rs import (
    ms2pip_compute_features,  # type: ignore[ty:unresolved-import]
    ms2pip_compute_theoretical_mz,  # type: ignore[ty:unresolved-import]
    ms2pip_extract_targets,  # type: ignore[ty:unresolved-import]
)
from psm_utils import PSM, Peptidoform, PSMList
from rich.progress import track

import ms2pip.exceptions as exceptions
from ms2pip._spectrum_processing import (
    MatchedSpectrum,
    annotate_spectrum,
    proforma_to_mass_shift,
    resolve_spectra,
)
from ms2pip._utils.psm_input import filter_valid_psms, read_psms
from ms2pip._utils.xgb_models import load_xgb_models, predict_intensities, validate_model
from ms2pip.constants import MODELS
from ms2pip.result import ProcessingResult, calculate_correlations
from ms2pip.search_space import ProteomeSearchSpace
from ms2pip.spectrum import ObservedSpectrum

logger = logging.getLogger(__name__)

NUM_FEATURES = 139


def _set_rayon_threads(processes: int | None) -> None:
    """Set RAYON_NUM_THREADS if processes is specified and not already set."""
    if processes is None:
        return
    if "RAYON_NUM_THREADS" in os.environ:
        logger.debug(
            "RAYON_NUM_THREADS already set to %s; not overriding with processes=%d",
            os.environ["RAYON_NUM_THREADS"],
            processes,
        )
        return
    os.environ["RAYON_NUM_THREADS"] = str(processes)


def _predict_batch_internal(
    psm_list: PSMList,
    model: str,
    model_dir: str | Path | None = None,
    processes: int | None = None,
    xgb_models: dict | None = None,
) -> list[ProcessingResult]:
    """
    Batch predict features, m/z, and intensities for all PSMs.

    Uses ms2rescore-rs for feature computation and theoretical m/z calculation
    (both internally parallelized via Rayon), then XGBoost for intensity prediction.
    """
    model_dir = validate_model(model, model_dir)
    if not psm_list:
        return []

    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]
    frag_model = MODELS[model]["fragmentation"]

    # Validate and filter PSMs
    valid_indices, _ = filter_valid_psms(psm_list)
    results: list[ProcessingResult] = [
        ProcessingResult(psm_index=i, psm=psm) for i, psm in enumerate(psm_list)
    ]

    if not valid_indices:
        return results

    valid_psms = [psm_list[i] for i in valid_indices]
    proformas = [proforma_to_mass_shift(psm.peptidoform) for psm in valid_psms]
    num_ions = [len(psm.peptidoform.parsed_sequence) - 1 for psm in valid_psms]

    _set_rayon_threads(processes)

    # Batch compute theoretical m/z
    logger.debug("Computing theoretical m/z for %d peptides...", len(proformas))
    all_mz = ms2pip_compute_theoretical_mz(proformas, ion_types, frag_model, "monoisotopic")

    # Batch compute features
    logger.debug("Computing features for %d peptides...", len(proformas))
    all_features = ms2pip_compute_features(proformas)

    # Predict intensities with XGBoost
    logger.debug("Predicting intensities with XGBoost...")
    predictions = predict_intensities(
        np.concatenate([f.reshape(-1, NUM_FEATURES) for f in all_features]),
        num_ions,
        MODELS[model],
        model_dir,
        processes=processes,
        xgb_models=xgb_models,
    )

    # Fill in results for valid PSMs
    for j, i in enumerate(valid_indices):
        results[i] = ProcessingResult(
            psm_index=i,
            psm=psm_list[i],
            theoretical_mz=all_mz[j],
            predicted_intensity=predictions[j],
        )
    return results


def _validate_and_extract_targets(
    psm_spectrum_annotations: list[MatchedSpectrum],
    model: str,
    processes: int | None = None,
) -> tuple[
    list[MatchedSpectrum], list[ProcessingResult], list[dict], list[str], list[int]
]:
    """
    Filter valid PSMs, extract observed targets, and prepare proformas/num_ions.

    Returns ``(valid_matches, skipped_results, all_targets, proformas, num_ions)``.
    """
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

    _set_rayon_threads(processes)

    if not psm_spectrum_annotations:
        return [], [], [], [], []

    # Validate and filter
    psms_for_validation = PSMList(psm_list=[m.psm for m in psm_spectrum_annotations])
    valid_indices, _ = filter_valid_psms(psms_for_validation)
    valid_index_set = set(valid_indices)

    valid_matches = [m for i, m in enumerate(psm_spectrum_annotations) if i in valid_index_set]
    skipped_results = [
        ProcessingResult(psm_index=m.psm_index, psm=m.psm)
        for i, m in enumerate(psm_spectrum_annotations) if i not in valid_index_set
    ]

    # Fill in missing precursor charges from spectra
    for m in valid_matches:
        if not m.psm.peptidoform.precursor_charge:
            m.psm.peptidoform.precursor_charge = m.spectrum.precursor_charge  # type: ignore[ty:invalid-assignment]

    # Extract targets from annotations (single batch Rust call)
    all_targets = ms2pip_extract_targets(
        annotated_spectra=[m.annotated_spectrum for m in valid_matches],
        intensities=[m.spectrum.intensity.astype(np.float32) for m in valid_matches],
        ion_types=ion_types,
        seq_lens=[len(m.psm.peptidoform.parsed_sequence) for m in valid_matches],
    )

    proformas = [proforma_to_mass_shift(m.psm.peptidoform) for m in valid_matches]
    num_ions = [len(m.psm.peptidoform.parsed_sequence) - 1 for m in valid_matches]

    return valid_matches, skipped_results, all_targets, proformas, num_ions


def _predict_with_observed(
    psm_spectrum_annotations: list[MatchedSpectrum],
    model: str,
    model_dir: str | Path | None = None,
    processes: int | None = None,
) -> list[ProcessingResult]:
    """Compute features, m/z, XGBoost predictions, and observed targets for matched spectra."""
    model_dir = validate_model(model, model_dir)
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]
    frag_model = MODELS[model]["fragmentation"]

    valid_matches, skipped_results, all_targets, proformas, num_ions = (
        _validate_and_extract_targets(psm_spectrum_annotations, model, processes)
    )
    if not valid_matches:
        return skipped_results

    logger.debug("Computing features for %d peptides...", len(proformas))
    all_features = ms2pip_compute_features(proformas)

    logger.debug("Computing theoretical m/z for %d peptides...", len(proformas))
    all_mz = ms2pip_compute_theoretical_mz(proformas, ion_types, frag_model, "monoisotopic")

    logger.debug("Predicting intensities with XGBoost...")
    predictions = predict_intensities(
        np.concatenate([f.reshape(-1, NUM_FEATURES) for f in all_features]),
        num_ions,
        MODELS[model],
        model_dir,
        processes=processes,
    )

    results = [
        ProcessingResult(
            psm_index=m.psm_index,
            psm=m.psm,
            theoretical_mz=all_mz[i],
            predicted_intensity=predictions[i],
            observed_intensity=all_targets[i],
        )
        for i, m in enumerate(valid_matches)
    ]
    results.extend(skipped_results)
    return results


def _extract_observations(
    psm_spectrum_annotations: list[MatchedSpectrum],
    model: str,
    processes: int | None = None,
) -> list[ProcessingResult]:
    """Compute theoretical m/z and extract observed targets (no predictions)."""
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]
    frag_model = MODELS[model]["fragmentation"]

    valid_matches, skipped_results, all_targets, proformas, _ = (
        _validate_and_extract_targets(psm_spectrum_annotations, model, processes)
    )
    if not valid_matches:
        return skipped_results

    logger.debug("Computing theoretical m/z for %d peptides...", len(proformas))
    all_mz = ms2pip_compute_theoretical_mz(proformas, ion_types, frag_model, "monoisotopic")

    results = [
        ProcessingResult(
            psm_index=m.psm_index,
            psm=m.psm,
            theoretical_mz=all_mz[i],
            observed_intensity=all_targets[i],
        )
        for i, m in enumerate(valid_matches)
    ]
    results.extend(skipped_results)
    return results


def _extract_training_data(
    psm_spectrum_annotations: list[MatchedSpectrum],
    model: str,
    processes: int | None = None,
) -> list[ProcessingResult]:
    """Compute features and extract observed targets for model training (no m/z or predictions)."""
    valid_matches, skipped_results, all_targets, proformas, _ = (
        _validate_and_extract_targets(psm_spectrum_annotations, model, processes)
    )
    if not valid_matches:
        return skipped_results

    logger.debug("Computing features for %d peptides...", len(proformas))
    all_features = ms2pip_compute_features(proformas)

    results = [
        ProcessingResult(
            psm_index=m.psm_index,
            psm=m.psm,
            observed_intensity=all_targets[i],
            feature_vectors=all_features[i],
        )
        for i, m in enumerate(valid_matches)
    ]
    results.extend(skipped_results)
    return results


def _into_batches(iterable, batch_size: int) -> Generator[list, None, None]:
    """Accumulate iterator elements into batches of a given size."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _assemble_training_data(results: list[ProcessingResult], model: str) -> pd.DataFrame:
    """Assemble training data from results list to single pandas DataFrame."""
    from ms2pip._utils.feature_names import get_feature_names

    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]

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
                [
                    r.observed_intensity[ion_type]
                    for r in results
                    if r.feature_vectors is not None and r.observed_intensity is not None
                ]
            )
        elif ion_type in ["x", "y", "y2", "z"]:
            training_data[f"target_{ion_type}"] = np.concatenate(
                [
                    r.observed_intensity[ion_type][::-1]
                    for r in results
                    if r.feature_vectors is not None and r.observed_intensity is not None
                ]
            )

    training_data = training_data[
        ["psm_index"] + get_feature_names() + [f"target_{it}" for it in ion_types]
    ]

    return training_data


def _add_im_rt(
    psm_list: PSMList,
    add_retention_time: bool,
    add_ion_mobility: bool,
    processes: int | None = None,
) -> None:
    """Add retention time and ion mobility predictions to PSMList if requested."""
    if add_retention_time:
        from deeplc import predict as _predict_rt

        logger.info("Adding retention time predictions with DeepLC")
        psm_list["retention_time"] = np.array(
            _predict_rt(psm_list, predict_kwargs={"num_threads": processes}), dtype=np.float32
        )  # type: ignore[ty:invalid-assignment]
    if add_ion_mobility:
        from im2deep import predict as _predict_im

        logger.info("Adding ion mobility predictions with IM2Deep...")
        psm_list["ion_mobility"] = np.array(
            _predict_im(psm_list, predict_kwargs={"num_threads": processes}), dtype=np.float32
        )  # type: ignore[ty:invalid-assignment]


def predict_single(
    peptidoform: Peptidoform | str,
    model: str = "HCD",
    model_dir: str | Path | None = None,
) -> ProcessingResult:
    """
    Predict fragmentation spectrum for a single peptide.\f
    """
    if isinstance(peptidoform, str):
        peptidoform = Peptidoform(peptidoform)
    psm = PSM(peptidoform=peptidoform, spectrum_id=0)
    psm_list = PSMList(psm_list=[psm])
    results = _predict_batch_internal(psm_list, model, model_dir)
    return results[0]


def predict_batch(
    psms: PSMList | str | Path,
    add_retention_time: bool = False,
    add_ion_mobility: bool = False,
    psm_filetype: str | None = None,
    model: str = "HCD",
    model_dir: str | Path | None = None,
    processes: int | None = None,
) -> list[ProcessingResult]:
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
    add_ion_mobility
        Add ion mobility predictions with IM2Deep (Requires optional IM2Deep dependency).
    model
        Model to use for prediction. Default: "HCD".
    model_dir
        Directory where XGBoost model files are stored. Default: `~/.ms2pip`.
    processes
        Number of threads for Rayon (Rust) and XGBoost parallelism. By default,
        all available.

    Returns
    -------
    predictions: list[ProcessingResult]
        Predicted spectra with theoretical m/z and predicted intensity values.

    """
    psm_list = read_psms(psms, filetype=psm_filetype)

    _add_im_rt(psm_list, add_retention_time, add_ion_mobility, processes=processes)

    logger.info("Processing peptides...")
    return _predict_batch_internal(psm_list, model, model_dir, processes=processes)


def predict_library(
    fasta_file: str | Path | None = None,
    config: ProteomeSearchSpace | dict | str | Path | None = None,
    add_retention_time: bool = False,
    add_ion_mobility: bool = False,
    model: str = "HCD",
    model_dir: str | Path | None = None,
    batch_size: int = 100000,
    processes: int | None = None,
) -> Generator[list[ProcessingResult], None, None]:
    """
    Predict spectral library from protein FASTA file.\f

    Parameters
    ----------
    fasta_file
        Path to FASTA file with protein sequences. Required if `search-space-config` is not
        provided.
    config
        ProteomeSearchSpace, or a dictionary or path to JSON file with proteome search space
        parameters. Required if `fasta_file` is not provided.
    add_retention_time
        Add retention time predictions with DeepLC (Requires optional DeepLC dependency).
    add_ion_mobility
        Add ion mobility predictions with IM2Deep (Requires optional IM2Deep dependency).
    model
        Model to use for prediction. Default: "HCD".
    model_dir
        Directory where XGBoost model files are stored. Default: `~/.ms2pip`.
    batch_size
        Number of peptides to process in each batch.
    processes
        Number of threads for Rayon (Rust) and XGBoost parallelism. By default,
        all available.

    Yields
    ------
    predictions: list[ProcessingResult]
        Predicted spectra with theoretical m/z and predicted intensity values.

    """
    if fasta_file and config:
        search_space = ProteomeSearchSpace.from_any(config)
        search_space.fasta_file = Path(fasta_file)
    elif fasta_file and not config:
        search_space = ProteomeSearchSpace(fasta_file=fasta_file)
    elif not fasta_file and config:
        search_space = ProteomeSearchSpace.from_any(config)
    else:
        raise ValueError("Either `fasta_file` or `config` must be provided.")

    search_space.build(processes=processes)

    # Convert to PSMList and filter by precursor m/z range
    psm_list = PSMList(psm_list=list(search_space))
    psm_list = search_space.filter_psms_by_mz(psm_list)

    _add_im_rt(psm_list, add_retention_time, add_ion_mobility, processes=processes)

    # Pre-load XGBoost models once for all batches
    model_dir = validate_model(model, model_dir)
    xgb_models = load_xgb_models(MODELS[model], model_dir, processes)

    for batch in track(
        _into_batches(psm_list, batch_size=batch_size),
        description="Predicting spectra...",
        total=ceil(len(psm_list) / batch_size),
    ):
        yield _predict_batch_internal(
            PSMList(psm_list=list(batch)),
            model,
            model_dir,
            processes=processes,
            xgb_models=xgb_models,
        )


def correlate(
    psms: PSMList | list[PSM] | str | Path,
    spectrum_file: str | Path | None = None,
    psm_filetype: str | None = None,
    spectrum_id_pattern: str | None = None,
    compute_correlations: bool = False,
    add_retention_time: bool = False,
    add_ion_mobility: bool = False,
    model: str = "HCD",
    model_dir: str | Path | None = None,
    ms2_tolerance: float = 0.02,
    ms2_tolerance_mode: str = "Da",
    processes: int | None = None,
) -> list[ProcessingResult]:
    """
    Compare predicted and observed intensities and optionally compute correlations.\f

    Spectra can be provided in two ways:

    - **From file**: pass ``spectrum_file`` with a path to a spectrum file. PSMs are matched
      to spectra by spectrum ID.
    - **Preloaded**: each PSM already has an :py:class:`ms2rescore_rs.MS2Spectrum` or
      :py:class:`ms2rescore_rs.AnnotatedMS2Spectrum` in its ``spectrum`` attribute.
      In this case, ``spectrum_file`` should not be provided.

    Parameters
    ----------
    psms
        PSMList, list of PSM objects, or path to a PSM file supported by psm_utils.
    spectrum_file
        Path to spectrum file with target intensities. Required when PSMs do not have
        preloaded spectra; must not be provided when they do.
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
    add_ion_mobility
        Add ion mobility predictions with IM2Deep (Requires optional IM2Deep dependency).
    model
        Model to use for prediction. Default: "HCD".
    model_dir
        Directory where XGBoost model files are stored. Default: `~/.ms2pip`.
    ms2_tolerance
        MS2 tolerance for observed spectrum peak annotation. By default, 0.02.
    ms2_tolerance_mode
        Unit of the MS2 tolerance: ``"Da"`` or ``"ppm"``. By default, ``"Da"``.
    processes
        Number of threads for Rayon (Rust) and XGBoost parallelism. By default,
        all available.

    Returns
    -------
    results: list[ProcessingResult]
        Predicted spectra with theoretical m/z and predicted intensity values, and optionally,
        correlations.

    """
    psm_list = read_psms(psms, filetype=psm_filetype)

    _add_im_rt(psm_list, add_retention_time, add_ion_mobility, processes=processes)

    matched = resolve_spectra(
        psm_list, spectrum_file, spectrum_id_pattern, model, ms2_tolerance, ms2_tolerance_mode
    )

    logger.info("Processing spectra and peptides...")
    results = _predict_with_observed(matched, model, model_dir, processes=processes)

    if compute_correlations:
        logger.info("Computing correlations")
        calculate_correlations(results)
        logger.info(
            f"Median correlation: "
            f"{np.median([r.correlation for r in results if r.correlation is not None])}"
        )

    return results


def correlate_single(
    observed_spectrum: ObservedSpectrum,
    ms2_tolerance: float = 0.02,
    ms2_tolerance_mode: str = "Da",
    model: str = "HCD",
) -> ProcessingResult:
    """
    Correlate single observed spectrum with predicted intensities.\f

    Parameters
    ----------
    observed_spectrum
        ObservedSpectrum instance with observed m/z and intensity values and peptidoform.
    ms2_tolerance
        MS2 tolerance for observed spectrum peak annotation. By default, 0.02.
    ms2_tolerance_mode
        Unit of the MS2 tolerance: ``"Da"`` or ``"ppm"``. By default, ``"Da"``.
    model
        Model to use for prediction. Default: "HCD".

    Returns
    -------
    result: ProcessingResult
        Result with theoretical m/z, predicted intensity, observed intensity, and correlation.

    """
    if not isinstance(observed_spectrum.peptidoform, Peptidoform):
        raise ValueError("Peptidoform must be set in observed spectrum to correlate.")

    # Preprocess a copy of the spectrum (TIC normalization + log2 transform)
    preprocessed = observed_spectrum.model_copy(deep=True)
    for label_type in ["iTRAQ", "TMT"]:
        if label_type in model:
            preprocessed.remove_reporter_ions(label_type)
    preprocessed.tic_norm()
    preprocessed.log2_transform()

    psm = PSM(peptidoform=observed_spectrum.peptidoform, spectrum_id=0)
    annotated = annotate_spectrum(preprocessed, psm, model, ms2_tolerance, ms2_tolerance_mode)
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]
    seq_len = len(observed_spectrum.peptidoform.parsed_sequence)
    observed_intensity = ms2pip_extract_targets(
        annotated_spectra=[annotated],
        intensities=[preprocessed.intensity.astype(np.float32)],
        ion_types=ion_types,
        seq_lens=[seq_len],
    )[0]

    result = predict_single(observed_spectrum.peptidoform, model=model)
    result.observed_intensity = observed_intensity

    calculate_correlations([result])
    return result


def get_training_data(
    psms: PSMList | str | Path,
    spectrum_file: str | Path,
    psm_filetype: str | None = None,
    spectrum_id_pattern: str | None = None,
    model: str = "HCD",
    ms2_tolerance: float = 0.02,
    ms2_tolerance_mode: str = "Da",
    processes: int | None = None,
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
        MS2 tolerance for observed spectrum peak annotation. By default, 0.02.
    ms2_tolerance_mode
        Unit of the MS2 tolerance: ``"Da"`` or ``"ppm"``. By default, ``"Da"``.
    processes
        Number of threads for Rayon (Rust) and XGBoost parallelism. By default,
        all available.

    Returns
    -------
    features
        :py:class:`pandas.DataFrame` with feature vectors and targets.

    """
    psm_list = read_psms(psms, filetype=psm_filetype)

    logger.info("Processing spectra and peptides...")
    matched = resolve_spectra(
        psm_list, spectrum_file, spectrum_id_pattern, model, ms2_tolerance, ms2_tolerance_mode
    )

    results = _extract_training_data(matched, model, processes=processes)

    logger.info("Assembling training data in DataFrame...")
    return _assemble_training_data(results, model)


def annotate_spectra(
    psms: PSMList | str | Path,
    spectrum_file: str | Path,
    psm_filetype: str | None = None,
    spectrum_id_pattern: str | None = None,
    model: str = "HCD",
    ms2_tolerance: float = 0.02,
    ms2_tolerance_mode: str = "Da",
    processes: int | None = None,
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
        MS2 tolerance for observed spectrum peak annotation. By default, 0.02.
    ms2_tolerance_mode
        Unit of the MS2 tolerance: ``"Da"`` or ``"ppm"``. By default, ``"Da"``.
    processes
        Number of threads for Rayon (Rust) and XGBoost parallelism. By default,
        all available.

    Returns
    -------
    results: list[ProcessingResult]
        List of ProcessingResult objects with theoretical m/z and observed intensity values.

    """
    psm_list = read_psms(psms, filetype=psm_filetype)

    logger.info("Processing spectra and peptides...")
    matched = resolve_spectra(
        psm_list, spectrum_file, spectrum_id_pattern, model, ms2_tolerance, ms2_tolerance_mode
    )

    return _extract_observations(matched, model, processes=processes)


def download_models(models: list[str] | None = None, model_dir: str | Path | None = None):
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
        if model not in MODELS:
            raise exceptions.UnknownModelError(model)
        if "xgboost_model_files" not in MODELS[model]:
            continue
        logger.debug("Downloading %s model files", model)
        validate_model(model, model_dir)
