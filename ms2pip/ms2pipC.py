#!/usr/bin/env python
import csv
import glob
import itertools
import logging
import multiprocessing
import multiprocessing.dummy
import os
import sys
from random import shuffle

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr

from ms2pip.cython_modules import ms2pip_pyx
from ms2pip.exceptions import (EmptySpectrumError,
                               FragmentationModelRequiredError,
                               InvalidModificationFormattingError,
                               InvalidPEPRECError, MissingConfigurationError,
                               NoValidPeptideSequencesError,
                               UnknownFragmentationMethodError,
                               UnknownModificationError,
                               UnknownOutputFormatError)
from ms2pip.feature_names import get_feature_names_new
from ms2pip.match_spectra import MatchSpectra
from ms2pip.ms2pip_tools import calc_correlations, spectrum_output
from ms2pip.peptides import (AMINO_ACID_IDS, Modifications,
                             write_amino_accid_masses)
from ms2pip.predict_xgboost import (validate_requested_xgb_model, initialize_xgb_models,
                                    process_peptides_xgb)
from ms2pip.retention_time import RetentionTime

logger = logging.getLogger("ms2pip")

# Supported output formats
SUPPORTED_OUT_FORMATS = ["csv", "mgf", "msp", "bibliospec", "spectronaut", "dlib"]

# Models and their properties
# id is passed to get_predictions to select model
# ion_types is required to write the ion types in the headers of the result files
# features_version is required to select the features version
MODELS = {
    "CID": {
        "id": 0,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_CID_train_B.xgboost",
            "y": "model_20190107_CID_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_CID_train_B.xgboost": "4398c6ebe23e2f37c0aca42b095053ecea6fb427",
            "model_20190107_CID_train_Y.xgboost": "e0a9eb37e50da35a949d75807d66fb57e44aca0f"
        }
    },
    "HCD2019": {
        "id": 1,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
    },
    "TTOF5600": {
        "id": 2,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_TTOF5600_train_B.xgboost",
            "y": "model_20190107_TTOF5600_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_TTOF5600_train_B.xgboost": "ab2e28dfbc4ee60640253b0b4c127fc272c9d0ed",
            "model_20190107_TTOF5600_train_Y.xgboost": "f8e9ddd8ca78ace06f67460a2fea0d8fa2623452"
        }
    },
    "TMT": {
        "id": 3,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
    },
    "iTRAQ": {
        "id": 4,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_iTRAQ_train_B.xgboost",
            "y": "model_20190107_iTRAQ_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_iTRAQ_train_B.xgboost": "b8d94ad329a245210c652a5b35d724d2c74d0d50",
            "model_20190107_iTRAQ_train_Y.xgboost": "56ae87d56fd434b53fcc1d291745cabb7baf463a"
        }
    },
    "iTRAQphospho": {
        "id": 5,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_iTRAQphospho_train_B.xgboost",
            "y": "model_20190107_iTRAQphospho_train_Y.xgboost",
        },
        "model_hash": {
            "model_20190107_iTRAQphospho_train_B.xgboost": "e283b158cc50e219f42f93be624d0d0ac01d6b49",
            "model_20190107_iTRAQphospho_train_Y.xgboost": "261b2e1810a299ed7ebf193ce1fb81a608c07d3b"
        }
    },
    # ETD': {'id': 6, 'ion_types': ['B', 'Y', 'C', 'Z'], 'peaks_version': 'etd', 'features_version': 'normal'},
    "HCDch2": {
        "id": 7,
        "ion_types": ["B", "Y", "B2", "Y2"],
        "peaks_version": "ch2",
        "features_version": "normal",
    },
    "CIDch2": {
        "id": 8,
        "ion_types": ["B", "Y", "B2", "Y2"],
        "peaks_version": "ch2",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20190107_CID_train_B.xgboost",
            "y": "model_20190107_CID_train_Y.xgboost",
            "b2": "model_20190107_CID_train_B2.xgboost",
            "y2": "model_20190107_CID_train_Y2.xgboost"
        },
        "model_hash": {
            "model_20190107_CID_train_B.xgboost": "4398c6ebe23e2f37c0aca42b095053ecea6fb427",
            "model_20190107_CID_train_Y.xgboost": "e0a9eb37e50da35a949d75807d66fb57e44aca0f",
            "model_20190107_CID_train_B2.xgboost": "602f2fc648890aebbbe2646252ade658af3221a3",
            "model_20190107_CID_train_Y2.xgboost": "4e4ad0f1d4606c17015aae0f74edba69f684d399"
        }
    },
    "HCD2021": {
        "id": 9,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20210416_HCD2021_B.xgboost",
            "y": "model_20210416_HCD2021_Y.xgboost",
        },
        "model_hash": {
            "model_20210416_HCD2021_B.xgboost": "c086c599f618b199bbb36e2411701fb2866b24c8",
            "model_20210416_HCD2021_Y.xgboost": "22a5a137e29e69fa6d4320ed7d701b61cbdc4fcf"
        }
    },
    "Immuno-HCD": {
        "id": 10,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20210316_Immuno_HCD_B.xgboost",
            "y": "model_20210316_Immuno_HCD_Y.xgboost",
        },
        "model_hash": {
            "model_20210316_Immuno_HCD_B.xgboost": "977466d378de2e89c6ae15b4de8f07800d17a7b7",
            "model_20210316_Immuno_HCD_Y.xgboost": "71948e1b9d6c69cb69b9baf84d361a9f80986fea"
        }
    },
    "CID-TMT": {
        "id": 11,
        "ion_types": ["B", "Y"],
        "peaks_version": "general",
        "features_version": "normal",
        "xgboost_model_files": {
            "b": "model_20220104_CID_TMT_B.xgboost",
            "y": "model_20220104_CID_TMT_Y.xgboost",
        },
        "model_hash": {
            "model_20220104_CID_TMT_B.xgboost": "fa834162761a6ae444bb6523c9c1a174b9738389",
            "model_20220104_CID_TMT_Y.xgboost": "299539179ca55d4ac82e9aed6a4e0bd134a9a41e"
        }
    },
}
MODELS["HCD"] = MODELS["HCD2021"]


def process_peptides(worker_num, data, afile, modfile, modfile2, PTMmap, model):
    """
    Function for each worker to process a list of peptides. The models are
    chosen based on model. PTMmap, Ntermmap and Ctermmap determine the
    modifications applied to each peptide sequence. Returns the predicted
    spectra for all the peptides.
    """

    ms2pip_pyx.ms2pip_init(afile, modfile, modfile2)

    pcount = 0

    # Prepare output variables
    mz_buf = []
    prediction_buf = []
    peplen_buf = []
    charge_buf = []
    pepid_buf = []

    # transform pandas dataframe into dictionary for easy access
    if "ce" in data.columns:
        specdict = (
            data[["spec_id", "peptide", "modifications", "charge", "ce"]]
            .set_index("spec_id")
            .to_dict()
        )
        ces = specdict["ce"]
    else:
        specdict = (
            data[["spec_id", "peptide", "modifications", "charge"]]
            .set_index("spec_id")
            .to_dict()
        )
    pepids = data["spec_id"].tolist()
    peptides = specdict["peptide"]
    modifications = specdict["modifications"]
    charges = specdict["charge"]
    del specdict

    for pepid in pepids:
        peptide = peptides[pepid]
        peptide = peptide.replace("L", "I")
        mods = modifications[pepid]

        # TODO: Check if 30 is good default CE!
        colen = 30
        if "ce" in data.columns:
            colen = ces[pepid]

        # Peptides longer then 101 lead to "Segmentation fault (core dumped)"
        if len(peptide) > 100:
            continue

        # convert peptide string to integer list to speed up C code
        peptide = np.array([0] + [AMINO_ACID_IDS[x] for x in peptide] + [0], dtype=np.uint16)
        modpeptide = apply_mods(peptide, mods, PTMmap)

        pepid_buf.append(pepid)
        peplen = len(peptide) - 2
        peplen_buf.append(peplen)

        ch = charges[pepid]
        charge_buf.append(ch)

        model_id = MODELS[model]["id"]
        peaks_version = MODELS[model]["peaks_version"]

        # get ion mzs
        mzs = ms2pip_pyx.get_mzs(modpeptide, peaks_version)

        mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])

        # Predict the b- and y-ion intensities from the peptide
        # For C-term ion types (y, y++, z), flip the order of predictions,
        # because get_predictions follows order from vector file
        # enumerate works for variable number (and all) ion types
        predictions = ms2pip_pyx.get_predictions(
            peptide, modpeptide, ch, model_id, peaks_version, colen
        )  # SD: added colen
        prediction_buf.append([np.array(p, dtype=np.float32) for p in predictions])

        pcount += 1
        if (pcount % 500) == 0:
            sys.stdout.write("(%i)%i " % (worker_num, pcount))
            sys.stdout.flush()

    return mz_buf, prediction_buf, None, peplen_buf, charge_buf, pepid_buf


def process_spectra(
    worker_num,
    data,
    spec_file,
    vector_file,
    afile,
    modfile,
    modfile2,
    PTMmap,
    model,
    fragerror,
    tableau,
    model_dir,
):
    """
    Function for each worker to process a list of spectra. Each peptide's
    sequence is extracted from the mgf file. Then models are chosen based on
    model. PTMmap, Ntermmap and Ctermmap determine the modifications
    applied to each peptide sequence and the spectrum is predicted. Then either
    the feature vectors are returned, or a DataFrame with the predicted and
    empirical intensities.
    """
    ms2pip_pyx.ms2pip_init(afile, modfile, modfile2)

    if "xgboost_model_files" in MODELS[model].keys():
        xgboost_models = initialize_xgb_models(
            MODELS[model]["xgboost_model_files"],
            model_dir,
            1,
        )

    # transform pandas datastructure into dictionary for easy access
    if "ce" in data.columns:
        specdict = (
            data[["spec_id", "peptide", "modifications", "ce"]]
            .set_index("spec_id")
            .to_dict()
        )
        ces = specdict["ce"]
    else:
        specdict = (
            data[["spec_id", "peptide", "modifications"]].set_index("spec_id").to_dict()
        )
    peptides = specdict["peptide"]
    modifications = specdict["modifications"]

    # cols contains the names of the computed features
    cols_n = get_feature_names_new()
    if "ce" in data.columns:
        cols_n.append("ce")
    # cols_n = get_feature_names_catboost()

    # SD
    dvectors = []
    dtargets = dict()
    psmids = []

    mz_buf = []
    target_buf = []
    prediction_buf = []
    peplen_buf = []
    charge_buf = []
    pepid_buf = []

    if tableau:
        ft = open("ms2pip_tableau.%i" % worker_num, "w")
        ft2 = open("stats_tableau.%i" % worker_num, "w")

    title = ""
    charge = 0
    msms = []
    peaks = []
    pepmass = 0
    f = open(spec_file)
    skip = False
    pcount = 0
    while 1:
        rows = f.readlines(50000)
        if not rows:
            break
        for row in rows:
            row = row.rstrip()
            if row == "":
                continue
            if skip:
                if row[0] == "B":
                    if row[:10] == "BEGIN IONS":
                        skip = False
                else:
                    continue
            if row == "":
                continue
            if row[0] == "T":
                if row[:5] == "TITLE":
                    title = row[6:]
                    if title not in peptides:
                        skip = True
                        continue
            elif row[0].isdigit():
                tmp = row.split()
                msms.append(float(tmp[0]))
                peaks.append(float(tmp[1]))
            elif row[0] == "B":
                if row[:10] == "BEGIN IONS":
                    msms = []
                    peaks = []
            elif row[0] == "C":
                if row[:6] == "CHARGE":
                    charge = int(row[7:9].replace("+", ""))
            elif row[0] == "P":
                if row[:7] == "PEPMASS":
                    pepmass = float(row.split("=")[1].split(" ")[0])
            elif row[:8] == "END IONS":
                # process current spectrum
                if title not in peptides:
                    continue

                peptide = peptides[title]
                peptide = peptide.replace("L", "I")
                mods = modifications[title]

                if "mut" in mods:
                    continue

                # Peptides longer then 101 lead to "Segmentation fault (core dumped)"
                if len(peptide) > 100:
                    continue

                # convert peptide string to integer list to speed up C code
                peptide = np.array(
                    [0] + [AMINO_ACID_IDS[x] for x in peptide] + [0], dtype=np.uint16
                )

                try:
                    modpeptide = apply_mods(peptide, mods, PTMmap)
                except UnknownModificationError as e:
                    logger.warn("Unknown modification: %s", e)
                    continue

                # remove reporter ions
                if "iTRAQ" in model:
                    for mi, mp in enumerate(msms):
                        if (mp >= 113) & (mp <= 118):
                            peaks[mi] = 0

                # TMT6plex: 126.1277, 127.1311, 128.1344, 129.1378, 130.1411, 131.1382
                if "TMT" in model:
                    for mi, mp in enumerate(msms):
                        if (mp >= 125) & (mp <= 132):
                            peaks[mi] = 0

                # remove percursor peak
                # for mi, mp in enumerate(msms):
                #   if (mp >= pepmass-0.02) & (mp <= pepmass+0.02):
                #       peaks[mi] = 0

                # normalize and convert MS2 peaks
                msms = np.array(msms, dtype=np.float32)
                tic = np.sum(peaks)
                peaks = peaks / tic
                peaks = np.log2(np.array(peaks) + 0.001)
                peaks = peaks.astype(np.float32)

                if (len(peaks) == 0) or (len(msms) == 0):
                    raise EmptySpectrumError()

                model_id = MODELS[model]["id"]
                peaks_version = MODELS[model]["peaks_version"]

                # TODO: Check if 30 is good default CE!
                # RG: removed `if ce == 0` in get_vector, split up into two functions
                colen = 30
                if "ce" in data.columns:
                    try:
                        colen = int(float(ces[title]))
                    except:
                        logger.warn("Could not parse collision energy!")
                        continue

                if vector_file:
                    # get targets
                    targets = ms2pip_pyx.get_targets(
                        modpeptide, msms, peaks, float(fragerror), peaks_version
                    )
                    psmids.extend([title] * (len(targets[0])))
                    if "ce" in data.columns:
                        dvectors.append(
                            np.array(
                                ms2pip_pyx.get_vector_ce(
                                    peptide, modpeptide, charge, colen
                                ),
                                dtype=np.uint16,
                            )
                        )  # SD: added collision energy
                    else:
                        dvectors.append(
                            np.array(
                                ms2pip_pyx.get_vector(peptide, modpeptide, charge),
                                dtype=np.uint16,
                            )
                        )

                    # Collecting targets to dict; works for variable number of ion types
                    # For C-term ion types (y, y++, z), flip the order of targets,
                    # for correct order in vectors DataFrame
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
                elif tableau:
                    numby = 0
                    numall = 0
                    explainedby = 0
                    explainedall = 0
                    ts = []
                    ps = []
                    predictions = ms2pip_pyx.get_predictions(
                        peptide, modpeptide, charge, model_id, peaks_version, colen
                    )
                    for m, p in zip(msms, peaks):
                        ft.write("%s;%f;%f;;;0\n" % (title, m, 2 ** p))
                    # get targets
                    mzs, targets = ms2pip_pyx.get_targets_all(
                        modpeptide, msms, peaks, float(fragerror), "all"
                    )
                    # get mean by intensity values to normalize!; WRONG !!!
                    maxt = 0.0
                    maxp = 0.0
                    it = 0
                    for cion in [1, 2]:
                        for ionnumber in range(len(modpeptide) - 3):
                            for lion in ["a", "b-h2o", "b-nh3", "b", "c"]:
                                if (lion == "b") & (cion == 1):
                                    if maxt < (2 ** targets[it]) - 0.001:
                                        maxt = (2 ** targets[it]) - 0.001
                                    if maxp < (2 ** predictions[0][ionnumber]) - 0.001:
                                        maxp = (2 ** predictions[0][ionnumber]) - 0.001
                                it += 1
                    for cion in [1, 2]:
                        for ionnumber in range(len(modpeptide) - 3):
                            for lion in ["y-h2o", "z", "y", "x"]:
                                if (lion == "y") & (cion == 1):
                                    if maxt < (2 ** targets[it]) - 0.001:
                                        maxt = (2 ** targets[it]) - 0.001
                                    if maxp < (2 ** predictions[1][ionnumber]) - 0.001:
                                        maxp = (2 ** predictions[1][ionnumber]) - 0.001
                                it += 1
                    # b
                    it = 0
                    for cion in [1, 2]:
                        for ionnumber in range(len(modpeptide) - 3):
                            for lion in ["a", "b-h2o", "b-nh3", "b", "c"]:
                                if mzs[it] > 0:
                                    numall += 1
                                    explainedall += (2 ** targets[it]) - 0.001
                                ft.write(
                                    "%s;%f;%f;%s;%i;%i;1\n"
                                    % (
                                        title,
                                        mzs[it],
                                        (2 ** targets[it]) / maxt,
                                        lion,
                                        cion,
                                        ionnumber,
                                    )
                                )
                                if (lion == "b") & (cion == 1):
                                    ts.append(targets[it])
                                    ps.append(predictions[0][ionnumber])
                                    if mzs[it] > 0:
                                        numby += 1
                                        explainedby += (2 ** targets[it]) - 0.001
                                    ft.write(
                                        "%s;%f;%f;%s;%i;%i;2\n"
                                        % (
                                            title,
                                            mzs[it],
                                            (2 ** (predictions[0][ionnumber])) / maxp,
                                            lion,
                                            cion,
                                            ionnumber,
                                        )
                                    )
                                it += 1
                    # y
                    for cion in [1, 2]:
                        for ionnumber in range(len(modpeptide) - 3):
                            for lion in ["y-h2o", "z", "y", "x"]:
                                if mzs[it] > 0:
                                    numall += 1
                                    explainedall += (2 ** targets[it]) - 0.001
                                ft.write(
                                    "%s;%f;%f;%s;%i;%i;1\n"
                                    % (
                                        title,
                                        mzs[it],
                                        (2 ** targets[it]) / maxt,
                                        lion,
                                        cion,
                                        ionnumber,
                                    )
                                )
                                if (lion == "y") & (cion == 1):
                                    ts.append(targets[it])
                                    ps.append(predictions[1][ionnumber])
                                    if mzs[it] > 0:
                                        numby += 1
                                        explainedby += (2 ** targets[it]) - 0.001
                                    ft.write(
                                        "%s;%f;%f;%s;%i;%i;2\n"
                                        % (
                                            title,
                                            mzs[it],
                                            (2 ** (predictions[1][ionnumber])) / maxp,
                                            lion,
                                            cion,
                                            ionnumber,
                                        )
                                    )
                                it += 1
                    ft2.write(
                        "%s;%i;%i;%f;%f;%i;%i;%f;%f;%f;%f\n"
                        % (
                            title,
                            len(modpeptide) - 2,
                            len(msms),
                            tic,
                            pearsonr(ts, ps)[0],
                            numby,
                            numall,
                            explainedby,
                            explainedall,
                            float(numby) / (2 * (len(peptide) - 3)),
                            float(numall) / (18 * (len(peptide) - 3)),
                        )
                    )
                else:
                    # Predict the b- and y-ion intensities from the peptide
                    pepid_buf.append(title)
                    peplen_buf.append(len(peptide) - 2)
                    charge_buf.append(charge)

                    # get/append ion mzs, targets and predictions
                    targets = ms2pip_pyx.get_targets(
                        modpeptide, msms, peaks, float(fragerror), peaks_version
                    )
                    target_buf.append([np.array(t, dtype=np.float32) for t in targets])
                    mzs = ms2pip_pyx.get_mzs(modpeptide, peaks_version)
                    mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])
                    if "xgboost_model_files" in MODELS[model].keys():
                        xgb_vector = np.array(
                            ms2pip_pyx.get_vector(peptide, modpeptide, charge),
                            dtype=np.uint16
                        )
                        xgb_vector = xgb.DMatrix(xgb_vector)
                        predictions = []
                        for ion_type, model_file in xgboost_models.items():
                            preds = model_file.predict(xgb_vector)
                            if ion_type in ["x", "y", "y2", "z"]:
                                preds = list(np.array(preds[::-1], dtype=np.float32))
                            elif ion_type in ["a", "b", "b2", "c"]:
                                preds = list(np.array(preds, dtype=np.float32))
                            else:
                                raise ValueError(f"Unsupported ion_type: {ion_type}")
                            predictions.append(preds)
                    else:
                        predictions = ms2pip_pyx.get_predictions(
                            peptide, modpeptide, charge, model_id, peaks_version, colen
                        )
                    prediction_buf.append(
                        [np.array(p, dtype=np.float32) for p in predictions]
                    )

                pcount += 1
                if (pcount % 500) == 0:
                    sys.stdout.write("(%i)%i " % (worker_num, pcount))
                    sys.stdout.flush()

    f.close()
    if tableau:
        ft.close()
        ft2.close()

    if vector_file:
        # If num_cpu > number of spectra, dvectors can be empty
        if dvectors:
            # Concatenating dvectors into a 2D ndarray before making DataFrame saves lots of memory!
            if len(dvectors) >= 1:
                dvectors = np.concatenate(dvectors)
            df = pd.DataFrame(dvectors, dtype=np.uint16, copy=False)
            df.columns = df.columns.astype(str)
        else:
            df = pd.DataFrame()
        return psmids, df, dtargets

    return mz_buf, prediction_buf, target_buf, peplen_buf, charge_buf, pepid_buf


def scan_spectrum_file(filename):
    """
    go over mgf file and return list with all spectrum titles
    """
    titles = []
    f = open(filename)
    while 1:
        rows = f.readlines(10000)
        if not rows:
            break
        for row in rows:
            if row[0] == "T":
                if row[:5] == "TITLE":
                    titles.append(
                        row.rstrip()[6:]
                    )  # .replace(" ", "") # unnecessary? creates issues when PEPREC spec_id has spaces
    f.close()
    return titles


def prepare_titles(titles, num_cpu):
    """
    Take a list and return a list containing num_cpu smaller lists with the
    spectrum titles/peptides that will be split across the workers
    """
    # titles might be ordered from small to large peptides,
    # shuffling improves parallel speeds
    shuffle(titles)

    split_titles = [
        titles[i * len(titles) // num_cpu : (i + 1) * len(titles) // num_cpu]
        for i in range(num_cpu)
    ]
    logger.debug(
        "{} spectra (~{:.0f} per cpu)\n".format(
            len(titles), np.mean([len(a) for a in split_titles])
        )
    )

    return split_titles


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


def peakcount(x):
    c = 0.0
    for i in x:
        if i > -9.95:
            c += 1.0
    return c / len(x)


class MS2PIP:
    def __init__(
        self,
        pep_file,
        spec_file=None,
        vector_file=None,
        num_cpu=1,
        params=None,
        output_filename=None,
        return_results=False,
        limit=None,
        add_retention_time=False,
        compute_correlations=False,
        match_spectra=False,
        sqldb_uri=None,
        tableau=False,
        model_dir=None,
    ):
        """
        MS²PIP peak intensity predictor.

        Parameters
        ----------
        pep_file : str or pandas.DataFrame
            Path to file or DataFrame with peptide information (see
            https://github.com/compomics/ms2pip_c#peprec-file)
        spec_file : str, optional
            Path to spectrum file with target intensities. Provide for
            prediction evaluation, or in combination with `vector_file` for
            target extraction.
        vector_file : str, optional
            Output filepath for training feature vectors. Provide this to
            extract feature vectors from `spec_file`. Requires `spec_file`.
        num_cpu : int, default: 1
            Number of parallel processes for multiprocessing steps.
        params : dict
            Dictonary with `model`, `frag_error` and `modifications`.
        output_filename : str, optional
            Filepath prefix for output files
        return_results : bool, default: False
            Return results after prediction (`MS2PIP.run()`) instead of writing
            to output files.
        limit : int, optional
            Limit to first N peptides in PEPREC file.
        add_retention_time : bool, default: False
            Add retention time predictions with DeepLC.
        compute_correlations : bool, default: False
            Compute correlations between predictions and targets. Requires
            `spec_file`.
        match_spectra : bool, default: False
            Match spectra in `spec_file` or `sqldb_uri` to peptides in
            `pep_file` based on predicted intensities (experimental).
        sqldb_uri : str, optional
            URI to SQL database for `match_spectra` feature.
        tableau : bool, default: False
            Write results to Tableau file.
        model_dir : str, optional
            Custom directory for downloaded XGBoost model files. By
            default, `~/.ms2pip` is used.

        Examples
        --------
        >>> from ms2pip.ms2pipC import MS2PIP
        >>> params = {
        ...     "ms2pip": {
        ...         "ptm": [
        ...             "Oxidation,15.994915,opt,M",
        ...             "Carbamidomethyl,57.021464,opt,C",
        ...             "Acetyl,42.010565,opt,N-term",
        ...         ],
        ...         "frag_method": "HCD",
        ...         "frag_error": 0.02,
        ...         "out": "csv",
        ...         "sptm": [], "gptm": [],
        ...     }
        ... }
        >>> ms2pip = MS2PIP("test.peprec", params=params)
        >>> ms2pip.run()

        """
        self.pep_file = pep_file
        self.vector_file = vector_file
        self.num_cpu = num_cpu
        self.params = params
        self.return_results = return_results
        self.limit = limit
        self.add_retention_time = add_retention_time
        self.compute_correlations = compute_correlations
        self.match_spectra = match_spectra
        self.sqldb_uri = sqldb_uri
        self.tableau = tableau
        self.model_dir = model_dir

        self.afile = None
        self.modfile = None
        self.modfile2 = None

        if params is None:
            raise MissingConfigurationError()

        if "model" in self.params["ms2pip"]:
            self.model = self.params["ms2pip"]["model"]
        elif "frag_method" in self.params["ms2pip"]:
            self.model = self.params["ms2pip"]["frag_method"]
        else:
            raise FragmentationModelRequiredError()
        self.fragerror = self.params["ms2pip"]["frag_error"]

        # Validate requested output formats
        if "out" in self.params["ms2pip"]:
            self.out_formats = [
                o.lower().strip() for o in self.params["ms2pip"]["out"].split(",")
            ]
            for o in self.out_formats:
                if o not in SUPPORTED_OUT_FORMATS:
                    raise UnknownOutputFormatError(o)
        else:
            if not return_results:
                logger.debug("No output format specified; defaulting to csv")
                self.out_formats = ["csv"]
            else:
                self.out_formats = []

        # Validate model_dir
        if not self.model_dir:
            self.model_dir = os.path.join(os.path.expanduser("~"), ".ms2pip")

        # Validate requested model
        if self.model in MODELS.keys():
            logger.info("using %s models", self.model)
            if "xgboost_model_files" in MODELS[self.model].keys():
                validate_requested_xgb_model(
                    MODELS[self.model]["xgboost_model_files"],
                    MODELS[self.model]["model_hash"],
                    self.model_dir
                )
        else:
            raise UnknownFragmentationMethodError(self.model)

        if output_filename is None and not return_results:
            self.output_filename = "{}_{}".format(
                ".".join(pep_file.split(".")[:-1]), self.model
            )
        else:
            self.output_filename = output_filename

        logger.debug(
            "starting workers (num_cpu=%d) ...",
            self.num_cpu,
        )
        if multiprocessing.current_process().daemon:
            logger.warn("MS2PIP is running in a daemon process. Disabling multiprocessing as daemonic processes can't have children.")
            self.myPool = multiprocessing.dummy.Pool(1)
        else:
            self.myPool = multiprocessing.Pool(self.num_cpu)

        if self.match_spectra:
            self.spec_file = None
            if self.sqldb_uri:
                self.spec_files = None
            elif os.path.isdir(spec_file):
                self.spec_files = glob.glob("{}/*.mgf".format(spec_file))
            else:
                self.spec_files = [spec_file]
            logger.debug("use spec files %s", self.spec_files)
        else:
            self.spec_file = spec_file
            self.spec_files = None

        self.mods = Modifications()
        for mod_type in ('sptm', 'ptm'):
            self.mods.add_from_ms2pip_modstrings(self.params["ms2pip"][mod_type], mod_type=mod_type)

    def run(self):
        """Run initiated MS2PIP based on class configuration."""
        self.afile = write_amino_accid_masses()
        self.modfile = self.mods.write_modifications_file(mod_type='ptm')
        self.modfile2 = self.mods.write_modifications_file(mod_type='sptm')

        self._read_peptide_information()

        if self.spec_file:
            results = self._process_spectra()

            logger.debug("Merging results")
            if self.vector_file:
                self._write_vector_file(results)
            else:
                all_preds = self._predict_spec(results)

                logger.info("writing file %s_pred_and_emp.csv...", self.output_filename)
                all_preds.to_csv(
                    "{}_pred_and_emp.csv".format(self.output_filename), index=False
                )

                if self.compute_correlations:
                    logger.info("computing correlations")
                    correlations = calc_correlations.calc_correlations(all_preds)
                    correlations.to_csv(
                        "{}_correlations.csv".format(self.output_filename), index=True
                    )
                    logger.info(
                        "median correlations: \n%s",
                        str(correlations.groupby("ion")["pearsonr"].median()),
                    )
        elif self.match_spectra:
            results = self._process_peptides()
            matched_spectra = self._match_spectra(results)
            self._write_matched_spectra(matched_spectra)
        else:
            if "xgboost_model_files" in MODELS[self.model]:
                results = self._process_peptides_xgb()
            else:
                results = self._process_peptides()

            if self.add_retention_time:
                self._predict_retention_times()

            logger.info("merging results ...")
            all_preds = self._predict_spec(results)

            if not self.return_results:
                self._write_predictions(all_preds)
            else:
                return all_preds

    def cleanup(self):
        """Cleanup temporary files."""
        if self.afile:
            os.remove(self.afile)
        if self.modfile:
            os.remove(self.modfile)
        if self.modfile2:
            os.remove(self.modfile2)

    def _read_peptide_information(self):
        # read peptide information
        # the file contains the columns: spec_id, modifications, peptide and charge
        if type(self.pep_file) == str:
            with open(self.pep_file, "rt") as f:
                line = f.readline()
                if line[:7] != "spec_id":
                    raise InvalidPEPRECError()
                sep = line[7]
            data = pd.read_csv(
                self.pep_file,
                sep=sep,
                index_col=False,
                dtype={"spec_id": str, "modifications": str},
                nrows=self.limit,
            )
        else:
            data = self.pep_file
        # for some reason the missing values are converted to float otherwise
        data = data.fillna("-")

        # Filter PEPREC for unsupported peptides
        num_pep = len(data)
        data = data[
            ~(data["peptide"].str.contains("B|J|O|U|X|Z"))
            & ~(data["peptide"].str.len() < 3)
            & ~(data["peptide"].str.len() > 99)
        ].copy()
        num_pep_filtered = num_pep - len(data)
        if num_pep_filtered > 0:
            logger.info(
                "Removed %i unsupported peptide sequences (< 3, > 99 amino "
                "acids, or containing B, J, O, U, X or Z).",
                num_pep_filtered,
            )

        if len(data) == 0:
            raise NoValidPeptideSequencesError()

        self.data = data

    def _execute_in_pool(self, titles, func, args):
        split_titles = prepare_titles(titles, self.num_cpu)
        results = []
        for i in range(self.num_cpu):
            tmp = split_titles[i]
            results.append(
                self.myPool.apply_async(
                    func, args=(i, self.data[self.data.spec_id.isin(tmp)], *args),
                )
            )
            # """
        self.myPool.close()
        self.myPool.join()
        sys.stdout.write("\n")
        return results

    def _process_spectra(self):
        """
        When an mgf file is provided, MS2PIP either saves the feature vectors to
        train models with or writes a file with the predicted spectra next to
        the empirical one.
        """
        logger.info("scanning spectrum file...")
        titles = scan_spectrum_file(self.spec_file)
        return self._execute_in_pool(
            titles,
            process_spectra,
            (
                self.spec_file,
                self.vector_file,
                self.afile,
                self.modfile,
                self.modfile2,
                self.mods.ptm_ids,
                self.model,
                self.fragerror,
                self.tableau,
                self.model_dir,
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

        # Only concat DataFrames with content (we get empty ones if more cpu's than peptides)
        all_results = pd.concat([df for df in all_results if len(df) != 0])

        logger.info("writing vector file %s...", self.vector_file)
        # write result. write format depends on extension:
        ext = self.vector_file.split(".")[-1]
        if ext == "pkl":
            all_results.to_pickle(self.vector_file + ".pkl")
        elif ext == "csv":
            all_results.to_csv(self.vector_file)
        else:
            # "table" is a tag used to read back the .h5
            all_results.to_hdf(self.vector_file, "table")

        return all_results

    def _predict_spec(self, results):
        mz_bufs = []
        prediction_bufs = []
        target_bufs = []
        peplen_bufs = []
        charge_bufs = []
        pepid_bufs = []
        for r in results:
            (
                mz_buf,
                prediction_buf,
                target_buf,
                peplen_buf,
                charge_buf,
                pepid_buf,
            ) = r.get()
            mz_bufs.extend(mz_buf)
            prediction_bufs.extend(prediction_buf)
            peplen_bufs.extend(peplen_buf)
            charge_bufs.extend(charge_buf)
            pepid_bufs.extend(pepid_buf)
            if target_buf:
                target_bufs.extend(target_buf)

        # Reconstruct DataFrame
        num_ion_types = len(MODELS[self.model]["ion_types"])
        ions = []
        ionnumbers = []
        charges = []
        pepids = []
        for pi, pl in enumerate(peplen_bufs):
            [
                ions.extend([ion_type] * (pl - 1))
                for ion_type in MODELS[self.model]["ion_types"]
            ]
            ionnumbers.extend([x + 1 for x in range(pl - 1)] * num_ion_types)
            charges.extend([charge_bufs[pi]] * (num_ion_types * (pl - 1)))
            pepids.extend([pepid_bufs[pi]] * (num_ion_types * (pl - 1)))
        all_preds = pd.DataFrame()
        all_preds["spec_id"] = pepids
        all_preds["charge"] = charges
        all_preds["ion"] = ions
        all_preds["ionnumber"] = ionnumbers
        all_preds["mz"] = np.concatenate(mz_bufs, axis=None)
        all_preds["prediction"] = np.concatenate(prediction_bufs, axis=None)
        if target_bufs:
            all_preds["target"] = np.concatenate(target_bufs, axis=None)
        if "rt" in self.data:
            # TODO: might be a good idea to index the dataframes on spec_id
            all_preds = all_preds.merge(self.data[["spec_id", "rt"]], on="spec_id", copy=False)

        return all_preds

    def _predict_retention_times(self):
        logging.info("Adding retention time predictions")
        rt_predictor = RetentionTime(config=self.params)
        rt_predictor.add_rt_predictions(self.data)

    def _process_peptides(self):
        logger.info("scanning peptide file...")
        titles = self.data.spec_id.tolist()
        return self._execute_in_pool(
            titles,
            process_peptides,
            (self.afile, self.modfile, self.modfile2, self.mods.ptm_ids, self.model),
        )

    def _process_peptides_xgb(self):
        """Process peptides and get predictions directly from XGBoost models."""
        ms2pip_pyx.ms2pip_init(self.afile, self.modfile, self.modfile2)

        return process_peptides_xgb(
            self.data,
            MODELS[self.model],
            self.mods.ptm_ids,
            self.model_dir,
            self.num_cpu
        )

    def _write_predictions(self, all_preds):
        spec_out = spectrum_output.SpectrumOutput(
            all_preds, self.data, self.params["ms2pip"], output_filename=self.output_filename,
        )
        spec_out.write_results(self.out_formats)

    def _match_spectra(self, results):
        mz_bufs, prediction_bufs, _, _, _, pepid_bufs = zip(*(r.get() for r in results))

        match_spectra = MatchSpectra(self.data,
                                     self.mods,
                                     itertools.chain.from_iterable(pepid_bufs),
                                     itertools.chain.from_iterable(mz_bufs),
                                     itertools.chain.from_iterable(prediction_bufs))
        if self.spec_files:
            return match_spectra.match_mgfs(self.spec_files)
        elif self.sqldb_uri:
            return match_spectra.match_sqldb(self.sqldb_uri)
        else:
            raise NotImplementedError

    def _write_matched_spectra(self, matched_spectra):
        filename = f"{self.output_filename}_matched_spectra.csv"
        logger.info("writing file %s...", filename)

        with open(filename, mode="w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(('spec_id', 'matched_file' 'matched_title'))
            for pep, spec_file, spec in matched_spectra:
                csv_writer.writerow((pep, spec_file, spec['params']['title']))
