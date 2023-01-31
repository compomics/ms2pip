#!/usr/bin/env python
import csv
import glob
import itertools
import logging
import multiprocessing
import multiprocessing.dummy
import os
import re
from random import shuffle

import numpy as np
import pandas as pd
import xgboost as xgb
from rich.progress import track

from ms2pip.cython_modules import ms2pip_pyx
from ms2pip.exceptions import (FragmentationModelRequiredError,
                               InvalidModificationFormattingError,
                               InvalidPEPRECError, MissingConfigurationError,
                               NoMatchingSpectraFound,
                               NoValidPeptideSequencesError, TitlePatternError,
                               UnknownFragmentationMethodError,
                               UnknownModificationError,
                               UnknownOutputFormatError)
from ms2pip.feature_names import get_feature_names_new
from ms2pip.match_spectra import MatchSpectra
from ms2pip.ms2pip_tools import calc_correlations, spectrum_output
from ms2pip.peptides import (AMINO_ACID_IDS, Modifications,
                             write_amino_acid_masses)
from ms2pip.predict_xgboost import (get_predictions_xgb,
                                    validate_requested_xgb_model)
from ms2pip.retention_time import RetentionTime
from ms2pip.spectrum import read_spectrum_file

logger = logging.getLogger(__name__)

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
            "model_20190107_CID_train_Y.xgboost": "e0a9eb37e50da35a949d75807d66fb57e44aca0f",
        },
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
            "model_20190107_TTOF5600_train_Y.xgboost": "f8e9ddd8ca78ace06f67460a2fea0d8fa2623452",
        },
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
            "model_20190107_iTRAQ_train_Y.xgboost": "56ae87d56fd434b53fcc1d291745cabb7baf463a",
        },
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
            "model_20190107_iTRAQphospho_train_Y.xgboost": "261b2e1810a299ed7ebf193ce1fb81a608c07d3b",
        },
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
            "y2": "model_20190107_CID_train_Y2.xgboost",
        },
        "model_hash": {
            "model_20190107_CID_train_B.xgboost": "4398c6ebe23e2f37c0aca42b095053ecea6fb427",
            "model_20190107_CID_train_Y.xgboost": "e0a9eb37e50da35a949d75807d66fb57e44aca0f",
            "model_20190107_CID_train_B2.xgboost": "602f2fc648890aebbbe2646252ade658af3221a3",
            "model_20190107_CID_train_Y2.xgboost": "4e4ad0f1d4606c17015aae0f74edba69f684d399",
        },
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
            "model_20210416_HCD2021_Y.xgboost": "22a5a137e29e69fa6d4320ed7d701b61cbdc4fcf",
        },
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
            "model_20210316_Immuno_HCD_Y.xgboost": "71948e1b9d6c69cb69b9baf84d361a9f80986fea",
        },
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
            "model_20220104_CID_TMT_Y.xgboost": "299539179ca55d4ac82e9aed6a4e0bd134a9a41e",
        },
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

    # Prepare output variables
    pepid_buf = []
    peplen_buf = []
    charge_buf = []
    mz_buf = []
    target_buf = None
    prediction_buf = []
    vector_buf = []

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

    # Track progress for only one worker (good approximation of all workers' progress)
    for pepid in track(
        pepids,
        total=len(pepids),
        disable=worker_num != 0,
        transient=True,
        description="",
    ):
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
        peptide = np.array(
            [0] + [AMINO_ACID_IDS[x] for x in peptide] + [0], dtype=np.uint16
        )
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

        # If using xgboost model file, get feature vectors to predict outside of MP.
        # Predictions will be added in `_merge_predictions` function.
        if "xgboost_model_files" in MODELS[model].keys():
            vector_buf.append(
                np.array(
                    ms2pip_pyx.get_vector(peptide, modpeptide, ch),
                    dtype=np.uint16,
                )
            )
        else:
            predictions = ms2pip_pyx.get_predictions(
                peptide, modpeptide, ch, model_id, peaks_version, colen
            )
            prediction_buf.append([np.array(p, dtype=np.float32) for p in predictions])

    return (
        pepid_buf,
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
    PTMmap,
    model,
    fragerror,
    spectrum_id_pattern,
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

    model_id = MODELS[model]["id"]
    peaks_version = MODELS[model]["peaks_version"]

    # transform pandas data structure into dictionary for easy access
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

    # if vector_file
    dvectors = []
    dtargets = dict()
    psmids = []

    # else
    pepid_buf = []
    peplen_buf = []
    charge_buf = []
    mz_buf = []
    target_buf = []
    prediction_buf = []
    vector_buf = []

    spectrum_id_regex = re.compile(spectrum_id_pattern)

    # Track progress for only one worker (good approximation of all workers' progress)
    for spectrum in track(
        read_spectrum_file(spec_file),
        total=len(peptides),
        disable=worker_num != 0,
        transient=True,
        description="",
    ):
        # Match title with regex
        match = spectrum_id_regex.search(spectrum.title)
        try:
            title = match[1]
        except (TypeError, IndexError):
            raise TitlePatternError(
                "Spectrum title pattern could not be matched to spectrum IDs "
                f"`{spectrum.title}`. "
                " Are you sure that the regex contains a capturing group?"
            )

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

        # Spectrum preprocessing:
        # Remove reporter ions and percursor peak, normalize, tranform
        for label_type in ["iTRAQ", "TMT"]:
            if label_type in model:
                spectrum.remove_reporter_ions("iTRAQ")
        # spectrum.remove_precursor()
        spectrum.tic_norm()
        spectrum.log2_transform()

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
                modpeptide,
                spectrum.msms,
                spectrum.peaks,
                float(fragerror),
                peaks_version,
            )
            psmids.extend([title] * (len(targets[0])))
            if "ce" in data.columns:
                dvectors.append(
                    np.array(
                        ms2pip_pyx.get_vector_ce(
                            peptide, modpeptide, spectrum.charge, colen
                        ),
                        dtype=np.uint16,
                    )
                )  # SD: added collision energy
            else:
                dvectors.append(
                    np.array(
                        ms2pip_pyx.get_vector(peptide, modpeptide, spectrum.charge),
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

        else:
            # Predict the b- and y-ion intensities from the peptide
            pepid_buf.append(title)
            peplen_buf.append(len(peptide) - 2)
            charge_buf.append(spectrum.charge)

            # get/append ion mzs, targets and predictions
            targets = ms2pip_pyx.get_targets(
                modpeptide,
                spectrum.msms,
                spectrum.peaks,
                float(fragerror),
                peaks_version,
            )
            target_buf.append([np.array(t, dtype=np.float32) for t in targets])
            mzs = ms2pip_pyx.get_mzs(modpeptide, peaks_version)
            mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])

            # If using xgboost model file, get feature vectors to predict outside of MP.
            # Predictions will be added in `_merge_predictions` function.
            if "xgboost_model_files" in MODELS[model].keys():
                vector_buf.append(
                    np.array(
                        ms2pip_pyx.get_vector(peptide, modpeptide, spectrum.charge),
                        dtype=np.uint16,
                    )
                )
            else:
                predictions = ms2pip_pyx.get_predictions(
                    peptide, modpeptide, spectrum.charge, model_id, peaks_version, colen
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
        pepid_buf,
        peplen_buf,
        charge_buf,
        mz_buf,
        target_buf,
        prediction_buf,
        vector_buf,
    )


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
        "{} spectra (~{:.0f} per cpu)".format(
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
        spectrum_id_pattern="(.*)",
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
        spectrum_id_pattern : str, optional
            Regular expression pattern to apply to spectrum titles before matching to
            peptide file entries.
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
        self.spectrum_id_pattern = spectrum_id_pattern
        self.num_cpu = num_cpu
        self.params = params
        self.return_results = return_results
        self.limit = limit
        self.add_retention_time = add_retention_time
        self.compute_correlations = compute_correlations
        self.match_spectra = match_spectra
        self.sqldb_uri = sqldb_uri
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
                logger.info("No output format specified; defaulting to csv")
                self.out_formats = ["csv"]
            else:
                self.out_formats = []

        # Validate model_dir
        if not self.model_dir:
            self.model_dir = os.path.join(os.path.expanduser("~"), ".ms2pip")

        # Validate requested model
        if self.model in MODELS.keys():
            logger.info("Using %s models", self.model)
            if "xgboost_model_files" in MODELS[self.model].keys():
                validate_requested_xgb_model(
                    MODELS[self.model]["xgboost_model_files"],
                    MODELS[self.model]["model_hash"],
                    self.model_dir,
                )
        else:
            raise UnknownFragmentationMethodError(self.model)

        if output_filename is None and not return_results:
            self.output_filename = "{}_{}".format(
                ".".join(pep_file.split(".")[:-1]), self.model
            )
        else:
            self.output_filename = output_filename

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

        if self.match_spectra:
            self.spec_file = None
            if self.sqldb_uri:
                self.spec_files = None
            elif os.path.isdir(spec_file):
                self.spec_files = glob.glob("{}/*.mgf".format(spec_file))
            else:
                self.spec_files = [spec_file]
            logger.debug("Using spectrum files %s", self.spec_files)
        else:
            self.spec_file = spec_file
            self.spec_files = None

        self.mods = Modifications()
        for mod_type in ("sptm", "ptm"):
            self.mods.add_from_ms2pip_modstrings(
                self.params["ms2pip"][mod_type], mod_type=mod_type
            )

    def run(self):
        """Run initiated MS2PIP based on class configuration."""
        self.afile = write_amino_acid_masses()
        self.modfile = self.mods.write_modifications_file(mod_type="ptm")
        self.modfile2 = self.mods.write_modifications_file(mod_type="sptm")

        self._read_peptide_information()

        if self.add_retention_time:
            logger.info("Adding retention time predictions")
            rt_predictor = RetentionTime(config=self.params)
            rt_predictor.add_rt_predictions(self.data)

        # Spectrum file mode
        if self.spec_file:
            logger.info("Processing spectra and peptides...")
            results = self._process_spectra()
            # Feature vectors requested
            if self.vector_file:
                self._write_vector_file(results)
            # Predictions (and targets) requested
            else:
                logger.debug("Merging results")
                all_preds = self._merge_predictions(results)
                # Correlations also requested
                if self.compute_correlations:
                    logger.info("Computing correlations")
                    correlations = calc_correlations.calc_correlations(all_preds)
                    logger.info(
                        "Median correlations: \n%s",
                        str(correlations.groupby("ion")["pearsonr"].median()),
                    )
                    if not self.return_results:
                        corr_filename = self.output_filename + "_correlations.csv"
                        logger.info(f"Writing file {corr_filename}")
                        correlations.to_csv(corr_filename, index=True)
                    else:
                        return correlations
                if not self.return_results:
                    pae_filename = self.output_filename + "_pred_and_emp.csv"
                    logger.info(f"Writing file {pae_filename}...")
                    all_preds.to_csv(pae_filename, index=False)
                else:
                    return all_preds

        # Match spectra mode
        elif self.match_spectra:
            results = self._process_peptides()
            matched_spectra = self._match_spectra(results)
            self._write_matched_spectra(matched_spectra)

        # Predictions-only mode
        else:
            logger.info("Processing peptides...")
            results = self._process_peptides()

            logger.debug("Merging results ...")
            all_preds = self._merge_predictions(results)

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
            logger.warning(
                f"Removed {num_pep_filtered} unsupported peptide sequences (< 3, > 99 "
                f"amino acids, or containing B, J, O, U, X or Z). Retained "
                f"{len(data)} entries."
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
                    func,
                    args=(i, self.data[self.data.spec_id.isin(tmp)], *args),
                )
            )
        self.myPool.close()
        self.myPool.join()
        return results

    def _process_spectra(self):
        """
        When an mgf file is provided, MS2PIP either saves the feature vectors to
        train models with or writes a file with the predicted spectra next to
        the empirical one.
        """
        titles = self.data["spec_id"].to_list()

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
                self.spectrum_id_pattern,
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

        logger.info("Writing vector file %s...", self.vector_file)
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

    def _merge_predictions(self, results):
        pepid_bufs = []
        peplen_bufs = []
        charge_bufs = []
        mz_bufs = []
        target_bufs = []
        prediction_bufs = []
        vector_bufs = []
        for r in results:
            (
                pepid_buf,
                peplen_buf,
                charge_buf,
                mz_buf,
                target_buf,
                prediction_buf,
                vector_buf,
            ) = r.get()
            pepid_bufs.extend(pepid_buf)
            peplen_bufs.extend(peplen_buf)
            charge_bufs.extend(charge_buf)
            mz_bufs.extend(mz_buf)
            if target_buf:
                target_bufs.extend(target_buf)
            if prediction_buf:
                prediction_bufs.extend(prediction_buf)
            if vector_buf:
                vector_bufs.extend(vector_buf)

        # Validate number of results
        if not mz_bufs:
            raise NoMatchingSpectraFound(
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
            all_preds = all_preds.merge(
                self.data[["spec_id", "rt"]], on="spec_id", copy=False
            )

        return all_preds

    def _process_peptides(self):
        titles = self.data.spec_id.tolist()
        return self._execute_in_pool(
            titles,
            process_peptides,
            (self.afile, self.modfile, self.modfile2, self.mods.ptm_ids, self.model),
        )

    def _write_predictions(self, all_preds):
        spec_out = spectrum_output.SpectrumOutput(
            all_preds,
            self.data,
            self.params["ms2pip"],
            output_filename=self.output_filename,
        )
        spec_out.write_results(self.out_formats)

    def _match_spectra(self, results):
        mz_bufs, prediction_bufs, _, _, _, pepid_bufs = zip(*(r.get() for r in results))

        match_spectra = MatchSpectra(
            self.data,
            self.mods,
            itertools.chain.from_iterable(pepid_bufs),
            itertools.chain.from_iterable(mz_bufs),
            itertools.chain.from_iterable(prediction_bufs),
        )
        if self.spec_files:
            return match_spectra.match_mgfs(self.spec_files)
        elif self.sqldb_uri:
            return match_spectra.match_sqldb(self.sqldb_uri)
        else:
            raise NotImplementedError

    def _write_matched_spectra(self, matched_spectra):
        filename = f"{self.output_filename}_matched_spectra.csv"
        logger.info("Writing file %s...", filename)

        with open(filename, mode="w") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(("spec_id", "matched_file" "matched_title"))
            for pep, spec_file, spec in matched_spectra:
                csv_writer.writerow((pep, spec_file, spec["params"]["title"]))
