"""Utilities for handling XGBoost model files within the MS²PIP prediction framework."""

import hashlib
import logging
import os
import urllib.request
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xgboost as xgb

import ms2pip.exceptions as exceptions
from ms2pip.constants import MODELS

logger = logging.getLogger(__name__)

_MAX_PREDICTION_THREADS = 16


def validate_model(model: str, model_dir: Union[str, Path, None] = None) -> Path:
    """
    Validate model name and ensure XGBoost model files are available.

    Downloads missing model files if necessary.

    Parameters
    ----------
    model
        Model name (must be a key in :data:`ms2pip.constants.MODELS`).
    model_dir
        Directory for XGBoost model files. Default: ``~/.ms2pip``.

    Returns
    -------
    model_dir
        Resolved model directory as Path.

    """
    model_dir = Path(model_dir) if model_dir else Path.home() / ".ms2pip"
    if model not in MODELS:
        raise exceptions.UnknownModelError(model)
    logger.debug("Using %s model", model)
    _validate_requested_xgb_model(
        MODELS[model]["xgboost_model_files"],
        MODELS[model]["model_hash"],
        model_dir,
    )
    return model_dir


def _validate_requested_xgb_model(xgboost_model_files, xgboost_model_hashes, model_dir):
    """Validate requested XGBoost models, and download if necessary."""
    for model_file in xgboost_model_files.values():
        if not _check_model_presence(model_file, xgboost_model_hashes[model_file], model_dir):
            _download_model(model_file, xgboost_model_hashes[model_file], model_dir)


def load_xgb_models(
    model_params: dict,
    model_dir,
    processes: Optional[int] = None,
) -> dict:
    """
    Load XGBoost models from disk.

    Returns a dict of ion_type -> xgb.Booster that can be passed to
    :func:`predict_intensities` to avoid re-loading on every call.

    Parameters
    ----------
    model_params
        Model configuration dict (must contain ``xgboost_model_files``).
    model_dir
        Directory where model files are stored.
    processes
        Number of threads for XGBoost prediction. Capped internally.

    """
    nthread = min(
        processes if processes is not None else (os.cpu_count() or 1),
        _MAX_PREDICTION_THREADS,
    )
    return _initialize_xgb_models(model_params["xgboost_model_files"], model_dir, nthread)


def predict_intensities(
    features: np.ndarray,
    num_ions: List[int],
    model_params: dict,
    model_dir,
    processes: Optional[int] = None,
    xgb_models: Optional[dict] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Predict intensities from feature vectors using XGBoost models.

    Parameters
    ----------
    features
        Feature vectors as numpy array. Will be converted to DMatrix internally.
    num_ions
        Number of ions (per series) for each peptide, i.e. peptide length - 1.
    model_params
        Model configuration dict (must contain ``xgboost_model_files``).
    model_dir
        Directory where model files are stored.
    processes
        Number of threads for XGBoost prediction. Capped internally to avoid
        diminishing returns. By default, uses all available cores (up to cap).
    xgb_models
        Pre-loaded XGBoost models (from :func:`load_xgb_models`). When provided,
        ``model_params``, ``model_dir``, and ``processes`` are ignored for model
        loading.

    Returns
    -------
    predictions
        List of dicts mapping ion type to predicted intensity array, one per peptide.

    """
    if xgb_models is None:
        xgb_models = load_xgb_models(model_params, model_dir, processes)
    dmatrix = xgb.DMatrix(features)

    logger.debug("Predicting intensities from XGBoost model files...")
    prediction_dict = {}
    for ion_type, xgb_model in xgb_models.items():
        preds = xgb_model.predict(dmatrix)
        preds = preds.clip(min=np.log2(0.001))

        if ion_type.lower() in ["x", "y", "y2", "z"]:
            preds = _split_list_by_lengths(preds, num_ions, reverse=True)
        elif ion_type.lower() in ["a", "b", "b2", "c"]:
            preds = _split_list_by_lengths(preds, num_ions, reverse=False)
        else:
            raise ValueError(f"Unsupported ion_type: {ion_type}")
        prediction_dict[ion_type] = preds

    num_peptides = len(list(prediction_dict.values())[0])
    return [{k: v[i] for k, v in prediction_dict.items()} for i in range(num_peptides)]


def _split_list_by_lengths(list_in, lengths, reverse=False):
    """Split list of predictions into sublists per peptide given their lengths."""
    list_in = iter(list_in)
    if reverse:
        list_out = [np.array(list(islice(list_in, e)), dtype=np.float32)[::-1] for e in lengths]
    else:
        list_out = [np.array(list(islice(list_in, e)), dtype=np.float32) for e in lengths]
    return list_out


def _check_model_presence(model, model_hash, model_dir):
    """Check whether XGBoost model file is downloaded."""
    filename = os.path.join(model_dir, model)
    if not os.path.isfile(filename):
        return False
    return _check_model_integrity(filename, model_hash)


def _download_model(model, model_hash, model_dir):
    """Download the xgboost model from the Genesis server."""
    os.makedirs(model_dir, exist_ok=True)
    filename = os.path.join(model_dir, model)

    logger.info(f"Downloading {model} to {filename}...")
    try:
        urllib.request.urlretrieve(
            f"https://genesis.ugent.be/uvpublicdata/ms2pip/{model}", filename
        )
    except Exception:
        logger.warning("Falling back to Zenodo for model downloads.")
        urllib.request.urlretrieve(f"https://zenodo.org/records/13270668/files/{model}", filename)
    if not _check_model_integrity(filename, model_hash):
        raise exceptions.InvalidXGBoostModelError()


def _check_model_integrity(filename, model_hash):
    """Check that models are correctly downloaded."""
    sha1_hash = hashlib.sha1()
    with open(filename, "rb") as model_file:
        while True:
            chunk = model_file.read(16 * 1024)
            if not chunk:
                break
            sha1_hash.update(chunk)
    if sha1_hash.hexdigest() == model_hash:
        return True
    else:
        logger.warning("Model hash not recognized.")
        return False


def _initialize_xgb_models(xgboost_model_files, model_dir, nthread) -> dict:
    """Initialize xgboost models and return them in a dict with ion types as keys."""
    xgb.set_config(verbosity=0)
    xgboost_models = {}
    for ion_type in xgboost_model_files.keys():
        model_file = os.path.join(model_dir, xgboost_model_files[ion_type])
        logger.debug(f"Initializing model from file: `{model_file}`")
        xgb_model = xgb.Booster({"nthread": nthread})
        xgb_model.load_model(model_file)
        xgboost_models[ion_type] = xgb_model
    return xgboost_models
