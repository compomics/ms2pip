"""Utilities for handling XGBoost model files within the MS²PIP prediction framework."""

import hashlib
import logging
import os
import urllib.request
from itertools import islice

import numpy as np
import xgboost as xgb

from ms2pip.exceptions import InvalidXGBoostModelError

logger = logging.getLogger(__name__)


def validate_requested_xgb_model(xgboost_model_files, xgboost_model_hashes, model_dir):
    """Validate requested XGBoost models, and download if necessary"""
    for _, model_file in xgboost_model_files.items():
        if not _check_model_presence(model_file, xgboost_model_hashes[model_file], model_dir):
            _download_model(model_file, xgboost_model_hashes[model_file], model_dir)


def get_predictions_xgb(features, num_ions, model_params, model_dir, processes=1):
    """
    Get predictions starting from feature vectors in DMatrix object.

    Parameters
    ----------
    features: xgb.DMatrix, np.ndarray
        Feature vectors in DMatrix object or as Numpy array.
    num_ions: list[int]
        List with number of ions (per series) for each peptide, i.e. peptide length - 1
    model_params: dict
        Model configuration as defined in ms2pip.ms2pipC.MODELS.
    model_dir: str
        Directory where model files are stored.
    processes: int
        Number of CPUs to use in multiprocessing

    """
    # Init models
    xgboost_models = _initialize_xgb_models(
        model_params["xgboost_model_files"],
        model_dir,
        processes,
    )
    if isinstance(features, np.ndarray):
        features = xgb.DMatrix(features)

    logger.debug("Predicting intensities from XGBoost model files...")
    prediction_dict = {}
    for ion_type, xgb_model in xgboost_models.items():
        # Get predictions from XGBoost model
        preds = xgb_model.predict(features)
        preds = preds.clip(min=np.log2(0.001))  # Clip negative intensities
        xgb_model.__del__()

        # Reshape into arrays for each peptide
        if ion_type.lower() in ["x", "y", "y2", "z"]:
            preds = _split_list_by_lengths(preds, num_ions, reverse=True)
        elif ion_type.lower() in ["a", "b", "b2", "c"]:
            preds = _split_list_by_lengths(preds, num_ions, reverse=False)
        else:
            raise ValueError(f"Unsupported ion_type: {ion_type}")
        prediction_dict[ion_type] = preds

    # Convert to list per peptide with dicts per ion type
    num_peptides = len(list(prediction_dict.values())[0])
    predictions = [{k: v[i] for k, v in prediction_dict.items()} for i in range(num_peptides)]
    return predictions


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
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    filename = os.path.join(model_dir, model)

    logger.info(f"Downloading {model} to {filename}...")
    urllib.request.urlretrieve(f"https://zenodo.org/records/13270668/files/{model}", filename)
    if not _check_model_integrity(filename, model_hash):
        raise InvalidXGBoostModelError()


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
        logger.warn("Model hash not recognized.")
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
