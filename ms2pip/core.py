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
    AnnotatedMS2Spectrum,  # type: ignore[ty:unresolved-import]
    MS2Spectrum,  # type: ignore[ty:unresolved-import]
    Precursor,  # type: ignore[ty:unresolved-import]
    annotate_ms2_spectra,  # type: ignore[ty:unresolved-import]
    ms2pip_compute_features,  # type: ignore[ty:unresolved-import]
    ms2pip_compute_theoretical_mz,  # type: ignore[ty:unresolved-import]
)
from psm_utils import PSM, Peptidoform, PSMList
from rich.progress import track

import ms2pip.exceptions as exceptions
from ms2pip._spectrum_processing import (
    annotate_spectrum,
    load_and_match_spectra,
    proforma_to_mass_shift,
    targets_from_annotations,
)
from ms2pip._utils.psm_input import read_psms
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

    proformas = [proforma_to_mass_shift(psm.peptidoform) for psm in psm_list]
    num_ions = [len(psm.peptidoform.parsed_sequence) - 1 for psm in psm_list]

    _set_rayon_threads(processes)

    # Batch compute theoretical m/z (single Rust call, Rayon-parallelized)
    logger.debug("Computing theoretical m/z for %d peptides...", len(proformas))
    all_mz = ms2pip_compute_theoretical_mz(proformas, ion_types, frag_model, "monoisotopic")

    # Batch compute features (single Rust call, Rayon-parallelized)
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

    # Assemble results
    results = []
    for i, psm in enumerate(psm_list):
        results.append(
            ProcessingResult(
                psm_index=i,
                psm=psm,
                theoretical_mz={k: np.array(v, dtype=np.float32) for k, v in all_mz[i].items()},
                predicted_intensity=predictions[i],
            )
        )
    return results


def _correlate_internal(
    psm_spectrum_annotations: list[tuple[int, PSM, ObservedSpectrum, list]],
    model: str,
    model_dir: str | Path | None = None,
    vector_file: bool = False,
    annotations_only: bool = False,
    processes: int | None = None,
) -> list[ProcessingResult]:
    """
    Core correlation logic: extract targets, compute features/predictions, assemble results.

    Parameters
    ----------
    psm_spectrum_annotations
        List of (psm_index, psm, preprocessed_spectrum, peak_annotations) tuples.
        Annotations are per-peak lists of ``(series, position, charge)`` tuples.
    model
        Name of prediction model.
    model_dir
        Directory for XGBoost model files.
    vector_file
        If True, return feature vectors instead of predictions (for training).
    annotations_only
        If True, return only m/z and observed intensities (no predictions).

    """
    if not annotations_only and not vector_file:
        model_dir = validate_model(model, model_dir)
    ion_types = [it.lower() for it in MODELS[model]["ion_types"]]
    frag_model = MODELS[model]["fragmentation"]

    _set_rayon_threads(processes)

    if not psm_spectrum_annotations:
        return []

    # Step 1: Extract targets from pre-computed annotations
    all_targets = []
    for psm_index, psm, spectrum, peak_annotations in psm_spectrum_annotations:
        if not psm.peptidoform.precursor_charge:
            psm.peptidoform.precursor_charge = spectrum.precursor_charge  # type: ignore[ty:invalid-assignment]

        seq_len = len(psm.peptidoform.parsed_sequence)
        targets = targets_from_annotations(
            peak_annotations, spectrum.intensity.astype(np.float32), ion_types, seq_len
        )
        all_targets.append(targets)

    proformas = [
        proforma_to_mass_shift(psm.peptidoform) for _, psm, _, _ in psm_spectrum_annotations
    ]
    num_ions = [
        len(psm.peptidoform.parsed_sequence) - 1 for _, psm, _, _ in psm_spectrum_annotations
    ]

    # Step 2: Compute features (needed for training and prediction, not annotation-only)
    all_features = None
    if not annotations_only:
        logger.debug("Computing features for %d peptides...", len(proformas))
        all_features = ms2pip_compute_features(proformas)

    # Step 3: Compute theoretical m/z (needed for annotation and prediction, not training)
    all_mz = None
    if not vector_file:
        logger.debug("Computing theoretical m/z for %d peptides...", len(proformas))
        all_mz = ms2pip_compute_theoretical_mz(proformas, ion_types, frag_model, "monoisotopic")

    # Step 4: Assemble results based on mode
    results = []

    if vector_file:
        # Training mode: return feature vectors + observed targets
        assert all_features is not None
        for i, (psm_index, psm, _spectrum, _ann) in enumerate(psm_spectrum_annotations):
            results.append(
                ProcessingResult(
                    psm_index=psm_index,
                    psm=psm,
                    theoretical_mz=None,
                    predicted_intensity=None,
                    observed_intensity=all_targets[i],
                    correlation=None,
                    feature_vectors=all_features[i],
                )
            )

    elif annotations_only:
        # Annotation mode: return m/z + observed targets
        assert all_mz is not None
        for i, (psm_index, psm, _spectrum, _ann) in enumerate(psm_spectrum_annotations):
            mz = {k: np.array(v, dtype=np.float32) for k, v in all_mz[i].items()}
            results.append(
                ProcessingResult(
                    psm_index=psm_index,
                    psm=psm,
                    theoretical_mz=mz,
                    predicted_intensity=None,
                    observed_intensity=all_targets[i],
                    correlation=None,
                    feature_vectors=None,
                )
            )

    else:
        # Prediction mode: compute XGBoost predictions
        assert all_features is not None and all_mz is not None
        logger.debug("Predicting intensities with XGBoost...")
        predictions = predict_intensities(
            np.concatenate([f.reshape(-1, NUM_FEATURES) for f in all_features]),
            num_ions,
            MODELS[model],
            model_dir,
            processes=processes,
        )

        for i, (psm_index, psm, _spectrum, _ann) in enumerate(psm_spectrum_annotations):
            mz = {k: np.array(v, dtype=np.float32) for k, v in all_mz[i].items()}
            results.append(
                ProcessingResult(
                    psm_index=psm_index,
                    psm=psm,
                    theoretical_mz=mz,
                    predicted_intensity=predictions[i],
                    observed_intensity=all_targets[i],
                )
            )

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
    if isinstance(psms, list):
        psms = PSMList(psm_list=psms)
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
    psms: PSMList | str | Path,
    spectrum_file: str | Path,
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
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    _add_im_rt(psm_list, add_retention_time, add_ion_mobility, processes=processes)

    # Validate runs and collections
    if len(psm_list.collections) != 1 or len(psm_list.runs) != 1:
        raise exceptions.InvalidInputError("PSMs should be for a single run and collection.")

    logger.info("Processing spectra and peptides...")
    matched = load_and_match_spectra(
        psm_list, spectrum_file, spectrum_id_pattern, model, ms2_tolerance, ms2_tolerance_mode
    )

    if not matched:
        raise exceptions.NoMatchingSpectraFound(
            "No spectra matching spectrum IDs from PSM list could be found in provided file."
        )

    results = _correlate_internal(matched, model, model_dir, processes=processes)

    if compute_correlations:
        logger.info("Computing correlations")
        calculate_correlations(results)
        logger.info(f"Median correlation: {np.median(list(r.correlation for r in results))}")

    return results


def correlate_preloaded(
    psms: PSMList | list[PSM],
    compute_correlations: bool = False,
    model: str = "HCD",
    model_dir: str | Path | None = None,
    ms2_tolerance: float = 0.02,
    ms2_tolerance_mode: str = "Da",
    processes: int | None = None,
) -> list[ProcessingResult]:
    """
    Compare predicted and observed intensities for PSMs with preloaded spectra.\f

    Processes PSMs that already have :py:class:`ms2rescore_rs.MS2Spectrum` or
    :py:class:`ms2rescore_rs.AnnotatedMS2Spectrum` objects in their ``spectrum``
    attribute.

    Parameters
    ----------
    psms
        PSMList or list of PSM objects. Each PSM must have an
        :py:class:`ms2rescore_rs.MS2Spectrum` or
        :py:class:`ms2rescore_rs.AnnotatedMS2Spectrum` object in its ``spectrum``
        attribute.
    compute_correlations
        Compute correlations between predictions and targets. Default: False.
    model
        Model to use for prediction. Default: "HCD".
    model_dir
        Directory where XGBoost model files are stored. Default: `~/.ms2pip`.
    ms2_tolerance
        MS2 tolerance for observed spectrum peak annotation. By default, 0.02.
        Only used when spectra are not already annotated.
    ms2_tolerance_mode
        Unit of the MS2 tolerance: ``"Da"`` or ``"ppm"``. By default, ``"Da"``.
        Only used when spectra are not already annotated.
    processes
        Number of threads for Rayon (Rust) and XGBoost parallelism. By default,
        all available.

    Returns
    -------
    results: list[ProcessingResult]
        ProcessingResult objects with theoretical m/z, predicted intensity, and observed
        intensity values, and optionally, correlations.

    Raises
    ------
    ValueError
        If PSMs do not contain spectrum objects in the ``spectrum`` attribute.

    """
    if isinstance(psms, list):
        psm_list = PSMList(psm_list=psms)
    else:
        psm_list = psms

    first_spectrum = psm_list["spectrum"][0]
    if not all(psm_list["spectrum"]) or not isinstance(
        first_spectrum, (MS2Spectrum, AnnotatedMS2Spectrum)
    ):
        raise ValueError(
            "PSMs must contain MS2Spectrum or AnnotatedMS2Spectrum objects "
            "in the 'spectrum' attribute."
        )

    spectra_are_annotated = isinstance(first_spectrum, AnnotatedMS2Spectrum)

    # Convert to ObservedSpectrum and preprocess; store annotations if present
    preloaded_spectra: dict[str, ObservedSpectrum] = {}
    preloaded_annotations: dict[str, list] | None = {} if spectra_are_annotated else None
    for psm in psm_list:
        spec_id = str(psm.spectrum_id)
        if spec_id in preloaded_spectra:
            continue
        spectrum = psm.spectrum
        assert spectrum is not None
        obs = ObservedSpectrum(
            mz=np.array(spectrum.mz, dtype=np.float32),
            intensity=np.array(spectrum.intensity, dtype=np.float32),
            identifier=str(spectrum.identifier),
            precursor_mz=float(spectrum.precursor.mz),
            precursor_charge=int(spectrum.precursor.charge),
            retention_time=float(spectrum.precursor.rt),
        )
        for label_type in ["iTRAQ", "TMT"]:
            if label_type in model:
                obs.remove_reporter_ions(label_type)
        obs.tic_norm()
        obs.log2_transform()
        preloaded_spectra[spec_id] = obs
        if spectra_are_annotated:
            assert isinstance(spectrum, AnnotatedMS2Spectrum)
            assert preloaded_annotations is not None
            preloaded_annotations[spec_id] = [
                [(a.series, a.position, a.charge) for a in peak_anns]
                for peak_anns in spectrum.peak_annotations
            ]

    # Build PSM-spectrum-annotation tuples
    # For unannotated spectra, batch annotate using ms2rescore-rs
    psm_spectrum_annotations = []
    needs_annotation = []  # indices into psm_spectrum_annotations that need annotation

    for i, psm in enumerate(psm_list):
        spec_id = str(psm.spectrum_id)
        spectrum = preloaded_spectra.get(spec_id)
        if spectrum is None:
            continue
        if preloaded_annotations is not None and spec_id in preloaded_annotations:
            psm_spectrum_annotations.append((i, psm, spectrum, preloaded_annotations[spec_id]))
        else:
            psm_spectrum_annotations.append((i, psm, spectrum, None))
            needs_annotation.append(len(psm_spectrum_annotations) - 1)

    if not psm_spectrum_annotations:
        raise exceptions.NoMatchingSpectraFound(
            "No spectra matching spectrum IDs from PSM list could be found."
        )

    # Batch annotate any unannotated spectra
    if needs_annotation:
        frag_model = MODELS[model]["fragmentation"]
        batch_spectra = []
        batch_proformas = []
        batch_seq_lens = []
        for idx in needs_annotation:
            _, psm, spectrum, _ = psm_spectrum_annotations[idx]
            batch_spectra.append(
                MS2Spectrum(
                    identifier=spectrum.identifier or "",
                    mz=list(spectrum.mz),
                    intensity=list(spectrum.intensity),
                    precursor=Precursor(
                        mz=float(spectrum.precursor_mz) if spectrum.precursor_mz else 0.0,
                        charge=int(spectrum.precursor_charge) if spectrum.precursor_charge else 0,
                        rt=float(spectrum.retention_time) if spectrum.retention_time else 0.0,
                    ),
                )
            )
            batch_proformas.append(proforma_to_mass_shift(psm.peptidoform))
            batch_seq_lens.append(len(psm.peptidoform.parsed_sequence))

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
            psm_index, psm, spectrum, _ = psm_spectrum_annotations[idx]
            peak_annotations = [
                [(a.series, a.position, a.charge) for a in peak_anns]
                for peak_anns in annotated[j].peak_annotations
            ]
            psm_spectrum_annotations[idx] = (psm_index, psm, spectrum, peak_annotations)

    logger.info("Processing spectra and peptides...")
    results = _correlate_internal(psm_spectrum_annotations, model, model_dir, processes=processes)

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
    observed_intensity = targets_from_annotations(
        annotated, preprocessed.intensity.astype(np.float32), ion_types, seq_len
    )

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
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    if len(psm_list.collections) != 1 or len(psm_list.runs) != 1:
        raise exceptions.InvalidInputError("PSMs should be for a single run and collection.")

    logger.info("Processing spectra and peptides...")
    matched = load_and_match_spectra(
        psm_list, spectrum_file, spectrum_id_pattern, model, ms2_tolerance, ms2_tolerance_mode
    )

    if not matched:
        raise exceptions.NoMatchingSpectraFound(
            "No spectra matching spectrum IDs from PSM list could be found in provided file."
        )

    results = _correlate_internal(
        matched,
        model,
        model_dir=None,
        vector_file=True,
        processes=processes,
    )

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
    spectrum_id_pattern = spectrum_id_pattern if spectrum_id_pattern else "(.*)"

    if len(psm_list.collections) != 1 or len(psm_list.runs) != 1:
        raise exceptions.InvalidInputError("PSMs should be for a single run and collection.")

    logger.info("Processing spectra and peptides...")
    matched = load_and_match_spectra(
        psm_list, spectrum_file, spectrum_id_pattern, model, ms2_tolerance, ms2_tolerance_mode
    )

    if not matched:
        raise exceptions.NoMatchingSpectraFound(
            "No spectra matching spectrum IDs from PSM list could be found in provided file."
        )

    return _correlate_internal(
        matched,
        model,
        model_dir=None,
        annotations_only=True,
        processes=processes,
    )


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
