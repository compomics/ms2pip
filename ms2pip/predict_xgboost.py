"""Get predictions directly from XGBoost model, within ms2pip framework."""

import hashlib
import logging
import os
import urllib.request
from itertools import islice

import numpy as np
import xgboost as xgb

from ms2pip.exceptions import (InvalidModificationFormattingError,
                               InvalidXGBoostModelError,
                               UnknownModificationError)
from ms2pip.ms2pipC import AMINO_ACID_IDS, ms2pip_pyx

logger = logging.getLogger("ms2pip")


def process_peptides_xgb(peprec, model_params, ptm_ids, model_dir, num_cpu=1):
    """Get predictions for peptides directly from XGBoost model."""
    feature_vectors, mzs = _get_ms2pip_data_for_xgb(peprec, model_params, ptm_ids)
    feature_vectors = xgb.DMatrix(feature_vectors)

    num_ions = (peprec["peptide"].str.len() - 1).to_list()
    peptide_lengths = peprec["peptide"].str.len().to_list()
    charges = peprec["charge"].to_list()
    spec_ids = peprec["spec_id"].to_list()

    preds_list = []
    for ion_type, model_file in model_params["xgboost_model_files"].items():
        # Get predictions from XGBoost model
        bst = xgb.Booster({"nthread": num_cpu})
        bst.load_model(os.path.join(model_dir, model_file))
        preds = bst.predict(feature_vectors)

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

    # List of objects with `get` method is expected, use spoofer class
    return [
        _MultiprocessingResultSpoofer(
            (mzs, predictions, None, peptide_lengths, charges, spec_ids)
        )
    ]


def _get_ms2pip_data_for_xgb(peprec, model_params, ptm_ids):
    """Get feature vectors and mz values for all peptides in self.data."""
    peaks_version = model_params["peaks_version"]

    vector_list = []
    mz_list = []
    for row in peprec.to_dict(orient="records"):

        peptide = np.array(
            [0] + [AMINO_ACID_IDS[x] for x in row["peptide"].replace("L", "I")] + [0],
            dtype=np.uint16,
        )
        modpeptide = apply_mods(peptide, row["modifications"], ptm_ids)
        charge = row["charge"]

        vector_list.append(
            np.array(
                ms2pip_pyx.get_vector(peptide, modpeptide, charge), dtype=np.uint16
            )
        )
        mzs = ms2pip_pyx.get_mzs(modpeptide, peaks_version)
        mz_list.append([np.array(m, dtype=np.float32) for m in mzs])

    feature_vectors = np.vstack(vector_list)

    return feature_vectors, mz_list


def _split_list_by_lengths(list_in, lengths):
    list_in = iter(list_in)
    return [list(islice(list_in, elem)) for elem in lengths]


class _MultiprocessingResultSpoofer:
    """Spoof result structure of multiprocessing, for direct XGB predictions."""

    def __init__(self, contents):
        self.contents = contents

    def get(self):
        return self.contents


def apply_mods(peptide, mods, PTMmap):
    """
    Takes a peptide sequence and a set of modifications. Returns the modified
    version of the peptide sequence, c- and n-term modifications. This modified
    version are hard coded in ms2pipfeatures_c.c for now.
    """
    modpeptide = np.array(peptide[:], dtype=np.uint16)

    if mods != "-":
        l = mods.split("|")
        if len(l) % 2 != 0:
            raise InvalidModificationFormattingError(mods)
        for i in range(0, len(l), 2):
            tl = l[i + 1]
            if tl in PTMmap:
                modpeptide[int(l[i])] = PTMmap[tl]
            else:
                raise UnknownModificationError(tl)

    return modpeptide


def check_model_presence(model, model_hash, model_dir):
    """Check whether xgboost model is downloaded."""
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
        os.path.join("http://genesis.ugent.be/uvpublicdata/ms2pip/", model),
        filename
    )
    if not check_model_integrity(filename, model_hash):
        raise InvalidXGBoostModelError()


def check_model_integrity(filename, model_hash):
    """Check that models are correctly downloaded."""
    sha1_hash = hashlib.sha1()
    with open(filename, "rb") as modelfile:
        while True:
            chunk = modelfile.read(16 * 1024)
            if not chunk:
                break
            sha1_hash.update(chunk)
    if sha1_hash.hexdigest() == model_hash:
        return True
    else:
        logger.warn("Model hash not recognised")
        return False


def validate_requested_xgb_model(xgboost_model_files, xgboost_model_hashes, model_dir):
    """Validate requestes xgboost models, and download if neccessary"""

    for _, model_file in xgboost_model_files.items():
        if not check_model_presence(
            model_file, xgboost_model_hashes[model_file], model_dir
        ):
            download_model(
                model_file,
                xgboost_model_hashes[model_file],
                model_dir
            )


def initialize_xgb_models(xgboost_model_files, model_dir, nthread) -> dict:
    """Initialize xgboost models and return them in a dict wit ion types as keys"""
    xgb.set_config(verbosity=0)

    xgboost_models = {}
    for ion_type in xgboost_model_files.keys():
        xgb_model = xgb.Booster({"nthread": nthread})
        xgb_model.load_model(
            os.path.join(model_dir, xgboost_model_files[ion_type])
        )
        xgboost_models[ion_type] = xgb_model

    return xgboost_models
