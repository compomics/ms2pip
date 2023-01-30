"""Get predictions directly from XGBoost model, within ms2pip framework."""

import hashlib
import logging
import os
import urllib.request
from itertools import islice

import numpy as np
import xgboost as xgb

from ms2pip.exceptions import InvalidXGBoostModelError

logger = logging.getLogger(__name__)


def get_predictions_xgb(features, num_ions, model_params, model_dir, num_cpu=1):
    """
    Get predictions starting from feature vectors in DMatrix object.

    Parameters
    ----------
    features: xgb.DMatrix
        Feature vectors in DMatrix object
    num_ions: list[int]
        List with number of ions (per series) for each peptide, i.e. peptide length - 1
    model_params: dict
        Model configuration as defined in ms2pip.ms2pipC.MODELS.
    model_dir: str
        Directory where model files are stored.
    num_cpu: int
        Number of CPUs to use in multiprocessing

    """
    # Init models
    xgboost_models = initialize_xgb_models(
        model_params["xgboost_model_files"],
        model_dir,
        num_cpu,
    )

    logger.debug("Predicting intensities from XGBoost model files...")
    preds_list = []
    for ion_type, xgb_model in xgboost_models.items():
        # Get predictions from XGBoost model
        preds = xgb_model.predict(features)
        xgb_model.__del__()

        # Reshape into arrays for each peptide
        preds = _split_list_by_lengths(preds, num_ions)
        if ion_type in ["x", "y", "y2", "z"]:
            preds = [np.array(x[::-1], dtype=np.float32) for x in preds]
        elif ion_type in ["a", "b", "b2", "c"]:
            preds = [np.array(x, dtype=np.float32) for x in preds]
        else:
            raise ValueError(f"Unsupported ion_type: {ion_type}")
        preds_list.append(preds)

    predictions = [list(t) for t in zip(*preds_list)]
    return predictions


def _split_list_by_lengths(list_in, lengths):
    list_in = iter(list_in)
    return [list(islice(list_in, elem)) for elem in lengths]


def check_model_presence(model, model_hash, model_dir):
    """Check whether XGBoost model file is downloaded."""
    filename = os.path.join(model_dir, model)
    if not os.path.isfile(filename):
        return False
    return check_model_integrity(filename, model_hash)


def download_model(model, model_hash, model_dir):
    """Download the xgboost model from the Genesis server."""
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    filename = os.path.join(model_dir, model)

    logger.info(f"Downloading {model} to {filename}...")
    urllib.request.urlretrieve(
        os.path.join("http://genesis.ugent.be/uvpublicdata/ms2pip/", model), filename
    )
    if not check_model_integrity(filename, model_hash):
        raise InvalidXGBoostModelError()


def check_model_integrity(filename, model_hash):
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


def validate_requested_xgb_model(xgboost_model_files, xgboost_model_hashes, model_dir):
    """Validate requested XGBoost models, and download if necessary"""
    for _, model_file in xgboost_model_files.items():
        if not check_model_presence(
            model_file, xgboost_model_hashes[model_file], model_dir
        ):
            download_model(model_file, xgboost_model_hashes[model_file], model_dir)


def initialize_xgb_models(xgboost_model_files, model_dir, nthread) -> dict:
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
