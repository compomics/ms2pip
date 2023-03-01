#!/usr/bin/env python
from __future__ import annotations

import csv
import glob
import itertools
import logging
import multiprocessing
import multiprocessing.dummy
import os
import re
from collections import defaultdict
from random import shuffle

import numpy as np
import pandas as pd
import xgboost as xgb
from rich.progress import track

import ms2pip.exceptions as exceptions
import ms2pip.peptides
from ms2pip.cython_modules import ms2pip_pyx
from ms2pip.match_spectra import MatchSpectra
from ms2pip.ms2pip_tools import calc_correlations, spectrum_output
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


def process_peptides(
    worker_num: int,
    data: pd.DataFrame,
    afile: str,
    modfile: str,
    modfile2: str,
    ptm_ids: dict[str, int],
    model: str,
):
    """
    Predict spectrum for each entry in PeptideRecord DataFrame.

    Parameters
    ----------
    worker_num: int
        Index of worker if using multiprocessing
    data: pandas.DataFrame
        PeptideRecord as Pandas DataFrame
    afile: str
        Filename of tempfile with amino acids definition for C code
    modfile: str
        Filename of tempfile with modification definition for C code
    modfile2: str
        Filename of tempfile with second instance of modification definition for C code
    ptm_ids: dict[str, int]
        Mapping of modification name -> modified residue integer encoding
    model: str
        Name of prediction model to be used

    Returns
    -------
    pepid_buf: list
    peplen_buf: list
    charge_buf: list
    mz_buf: list
    target_buf: list
    prediction_buf: list
    vector_buf: list

    """
    ms2pip_pyx.ms2pip_init(afile, modfile, modfile2)

    model_id = MODELS[model]["id"]
    peaks_version = MODELS[model]["peaks_version"]

    # Prepare output variables
    psm_id_buf = []
    spec_id_buf = []
    peplen_buf = []
    charge_buf = []
    mz_buf = []
    target_buf = None
    prediction_buf = []
    vector_buf = []

    # Track progress for only one worker (good approximation of all workers' progress)
    for entry in track(
        data.itertuples(),
        total=len(data),
        disable=worker_num != 0,
        transient=True,
        description="",
    ):
        psm_id_buf.append(entry.psm_id)
        spec_id_buf.append(entry.spec_id)
        peplen_buf.append(len(entry.peptide))
        charge_buf.append(entry.charge)

        try:
            enc_peptide = ms2pip.peptides.encode_peptide(entry.peptide)
            enc_peptidoform = ms2pip.peptides.apply_modifications(
                enc_peptide, entry.modifications, ptm_ids
            )
        except (
            exceptions.InvalidPeptideError,
            exceptions.InvalidAminoAcidError,
            exceptions.InvalidModificationFormattingError,
            exceptions.UnknownModificationError,
        ):
            continue

        # Get ion mzs
        mzs = ms2pip_pyx.get_mzs(enc_peptidoform, peaks_version)
        mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])

        # If using xgboost model file, get feature vectors to predict outside of MP.
        # Predictions will be added in `_merge_predictions` function.
        if "xgboost_model_files" in MODELS[model].keys():
            vectors = np.array(
                ms2pip_pyx.get_vector(enc_peptide, enc_peptidoform, entry.charge),
                dtype=np.uint16,
            )
            vector_buf.append(vectors)
        # Else, get predictions from C models in multiprocessing.
        else:
            predictions = ms2pip_pyx.get_predictions(
                enc_peptide,
                enc_peptidoform,
                entry.charge,
                model_id,
                peaks_version,
                entry.ce,
            )
            prediction_buf.append([np.array(p, dtype=np.float32) for p in predictions])

    return (
        spec_id_buf,
        peplen_buf,
        charge_buf,
        mz_buf,
        target_buf,
        prediction_buf,
        vector_buf,
        psm_id_buf
    )


def process_spectra(
    worker_num,
    data,
    spec_file,
    vector_file,
    afile,
    modfile,
    modfile2,
    ptm_ids,
    model,
    fragerror,
    spectrum_id_pattern,
):
    """
    Perform requested tasks for each spectrum in spectrum file.

    Parameters
    ----------
    worker_num: int
        Index of worker if using multiprocessing
    data: pandas.DataFrame
        PeptideRecord as Pandas DataFrame
    spec_file: str
        Filename of spectrum file
    vector_file: str, None
        Output filename for feature vector file
    afile: str
        Filename of tempfile with amino acids definition for C code
    modfile: str
        Filename of tempfile with modification definition for C code
    modfile2: str
        Filename of tempfile with second instance of modification definition for C code
    ptm_ids: dict[str, int]
        Mapping of modification name -> modified residue integer encoding
    model: str
        Name of prediction model to be used
    fragerror: float
        Fragmentation spectrum m/z error tolerance in Dalton
    spectrum_id_pattern
        Regular expression pattern to apply to spectrum titles before matching to
        peptide file entries

    Returns
    -------
    pepid_buf: list
    peplen_buf: list
    charge_buf: list
    mz_buf: list
    target_buf: list
    prediction_buf: list
    vector_buf: list

    """
    ms2pip_pyx.ms2pip_init(afile, modfile, modfile2)

    model_id = MODELS[model]["id"]
    peaks_version = MODELS[model]["peaks_version"]

    # if vector_file
    dvectors = []
    dtargets = dict()
    psmids = []

    # 
    psm_id_buf = []
    spec_id_buf = []
    peplen_buf = []
    charge_buf = []
    mz_buf = []
    target_buf = []
    prediction_buf = []
    vector_buf = []

    try:
        spectrum_id_regex = re.compile(spectrum_id_pattern)
    except TypeError:
        spectrum_id_regex = re.compile(r"(.*)")

    # Restructure PeptideRecord entries as spec_id -> [psm_1, psm_2, ...]
    entries_by_specid = defaultdict(list)
    for entry in data.itertuples():
        entries_by_specid[entry.spec_id].append(entry)

    # Track progress for only one worker (good approximation of all workers' progress)
    for spectrum in track(
        read_spectrum_file(spec_file),
        total=len(data),
        disable=worker_num != 0,
        transient=True,
        description="",
    ):
        # Match spectrum ID with provided regex, use first match group as new ID
        match = spectrum_id_regex.search(spectrum.title)
        try:
            spectrum_id = match[1]
        except (TypeError, IndexError):
            raise exceptions.TitlePatternError(
                "Spectrum title pattern could not be matched to spectrum IDs "
                f"`{spectrum.title}`. "
                " Are you sure that the regex contains a capturing group?"
            )

        # Spectrum preprocessing:
        # Remove reporter ions and precursor peak, normalize, transform
        for label_type in ["iTRAQ", "TMT"]:
            if label_type in model:
                spectrum.remove_reporter_ions(label_type)
        # spectrum.remove_precursor()
        spectrum.tic_norm()
        spectrum.log2_transform()

        for entry in entries_by_specid[spectrum_id]:
            try:
                enc_peptide = ms2pip.peptides.encode_peptide(entry.peptide)
                enc_peptidoform = ms2pip.peptides.apply_modifications(
                    enc_peptide, entry.modifications, ptm_ids
                )
            except (
                exceptions.InvalidPeptideError,
                exceptions.InvalidAminoAcidError,
                exceptions.InvalidModificationFormattingError,
                exceptions.UnknownModificationError,
            ):
                continue

            if vector_file:
                targets = ms2pip_pyx.get_targets(
                    enc_peptidoform,
                    spectrum.msms,
                    spectrum.peaks,
                    float(fragerror),
                    peaks_version,
                )
                psmids.extend([spectrum_id] * (len(targets[0])))
                dvectors.append(
                    np.array(
                        ms2pip_pyx.get_vector(
                            enc_peptide, enc_peptidoform, entry.charge
                        ),
                        dtype=np.uint16,
                    )
                )

                # Restructure to dict with entries per ion type
                # Flip the order for C-terminal ion types
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
                psm_id_buf.append(entry.psm_id)
                spec_id_buf.append(spectrum_id)
                peplen_buf.append(len(entry.peptide))
                charge_buf.append(entry.charge)

                targets = ms2pip_pyx.get_targets(
                    enc_peptidoform,
                    spectrum.msms,
                    spectrum.peaks,
                    float(fragerror),
                    peaks_version,
                )
                target_buf.append([np.array(t, dtype=np.float32) for t in targets])

                mzs = ms2pip_pyx.get_mzs(enc_peptidoform, peaks_version)
                mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])

                # If using xgboost model file, get feature vectors to predict outside of MP.
                # Predictions will be added in `_merge_predictions` function.
                if "xgboost_model_files" in MODELS[model].keys():
                    vector_buf.append(
                        np.array(
                            ms2pip_pyx.get_vector(
                                enc_peptide, enc_peptidoform, entry.charge
                            ),
                            dtype=np.uint16,
                        )
                    )
                # Else, get predictions from C models in multiprocessing.
                else:
                    predictions = ms2pip_pyx.get_predictions(
                        enc_peptide,
                        enc_peptidoform,
                        entry.charge,
                        model_id,
                        peaks_version,
                        entry.ce,
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
        spec_id_buf,
        peplen_buf,
        charge_buf,
        mz_buf,
        target_buf,
        prediction_buf,
        vector_buf,
        psm_id_buf
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


class MS2PIP:
    def __init__(
        self,
        pep_file,
        spec_file=None,
        spectrum_id_pattern=None,
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
        self.spectrum_id_pattern = (
            spectrum_id_pattern if spectrum_id_pattern else "(.*)"
        )
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
            raise exceptions.MissingConfigurationError()

        if "model" in self.params["ms2pip"]:
            self.model = self.params["ms2pip"]["model"]
        elif "frag_method" in self.params["ms2pip"]:
            self.model = self.params["ms2pip"]["frag_method"]
        else:
            raise exceptions.FragmentationModelRequiredError()
        self.fragerror = self.params["ms2pip"]["frag_error"]

        # Validate requested output formats
        if "out" in self.params["ms2pip"]:
            self.out_formats = [
                o.lower().strip() for o in self.params["ms2pip"]["out"].split(",")
            ]
            for o in self.out_formats:
                if o not in SUPPORTED_OUT_FORMATS:
                    raise exceptions.UnknownOutputFormatError(o)
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
            logger.debug("Using %s models", self.model)
            if "xgboost_model_files" in MODELS[self.model].keys():
                validate_requested_xgb_model(
                    MODELS[self.model]["xgboost_model_files"],
                    MODELS[self.model]["model_hash"],
                    self.model_dir,
                )
        else:
            raise exceptions.UnknownFragmentationMethodError(self.model)

        if output_filename is None and not return_results and isinstance(self.pep_file, str):
            self.output_filename = "{}_{}".format(".".join(pep_file.split(".")[:-1]), self.model)
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

        # TODO: Move to run?
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
        # TODO

        self.mods = ms2pip.peptides.Modifications()
        for mod_type in ("sptm", "ptm"):
            self.mods.add_from_ms2pip_modstrings(
                self.params["ms2pip"][mod_type], mod_type=mod_type
            )

    # TODO: Pass PEPREC and SPECFILE args here?
    def run(self):
        """Run initiated MS2PIP based on class configuration."""
        self.afile = ms2pip.peptides.write_amino_acid_masses()
        self.modfile = self.mods.write_modifications_file(mod_type="ptm")
        self.modfile2 = self.mods.write_modifications_file(mod_type="sptm")
        #

        self._read_peptide_information()

        if self.add_retention_time:
            logger.info("Adding retention time predictions")
            rt_predictor = RetentionTime(config=self.params, num_cpu=self.num_cpu)
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
                        try:
                            correlations.to_csv(
                                corr_filename,
                                index=True,
                                lineterminator="\n",
                            )
                        except TypeError:  # Pandas < 1.5 (Required for Python 3.7 support)
                            correlations.to_csv(
                                corr_filename,
                                index=True,
                                line_terminator="\n",
                            )
                    else:
                        return correlations
                if not self.return_results:
                    pae_filename = self.output_filename + "_pred_and_emp.csv"
                    logger.info(f"Writing file {pae_filename}...")
                    try:
                        all_preds.to_csv(
                            pae_filename,
                            index=False,
                            lineterminator="\n",
                        )
                    except TypeError:  # Pandas < 1.5 (Required for Python 3.7 support)
                        all_preds.to_csv(
                            pae_filename,
                            index=False,
                            line_terminator="\n",
                        )
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
        """Validate and process PeptideRecord DataFrame."""
        if isinstance(self.pep_file, str):
            with open(self.pep_file, "rt") as f:
                line = f.readline()
                if line[:7] != "spec_id":
                    raise exceptions.InvalidPEPRECError()
                sep = line[7]
            data = pd.read_csv(
                self.pep_file,
                sep=sep,
                index_col=False,
                dtype={"spec_id": str, "modifications": str},
                nrows=self.limit,
            )
        elif isinstance(self.pep_file, pd.DataFrame):
            data = self.pep_file
        else:
            raise TypeError("Invalid type for peptide file")

        data = data.fillna("-")
        if not "ce" in data.columns:
            data["ce"] = 30
        else:
            data["ce"] = data["ce"].astype(int)

        data["charge"] = data["charge"].astype(int)

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
            raise exceptions.NoValidPeptideSequencesError()
        
        if not "psm_id" in data.columns:
            data.reset_index(inplace=True)
            data["psm_id"] = data["index"].astype(str)
            data.rename({"index": "psm_id"}, axis=1)            
        
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
            try:
                all_results.to_csv(self.vector_file, lineterminator="\n")
            except TypeError:  # Pandas < 1.5 (Required for Python 3.7 support)
                all_results.to_csv(self.vector_file, line_terminator="\n")
        else:
            # "table" is a tag used to read back the .h5
            all_results.to_hdf(self.vector_file, "table")

        return all_results

    def _merge_predictions(self, results):
        spec_id_bufs = []
        peplen_bufs = []
        charge_bufs = []
        mz_bufs = []
        target_bufs = []
        prediction_bufs = []
        vector_bufs = []
        psm_id_bufs = []
        for r in results:
            (
                spec_id_buf,
                peplen_buf,
                charge_buf,
                mz_buf,
                target_buf,
                prediction_buf,
                vector_buf,
                psm_id_buf
            ) = r.get()
            spec_id_bufs.extend(spec_id_buf)
            peplen_bufs.extend(peplen_buf)
            charge_bufs.extend(charge_buf)
            mz_bufs.extend(mz_buf)
            psm_id_bufs.extend(psm_id_buf)
            if target_buf:
                target_bufs.extend(target_buf)
            if prediction_buf:
                prediction_bufs.extend(prediction_buf)
            if vector_buf:
                vector_bufs.extend(vector_buf)

        # Validate number of results
        if not mz_bufs:
            raise exceptions.NoMatchingSpectraFound(
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
        psm_ids = []
        for pi, pl in enumerate(peplen_bufs):
            [
                ions.extend([ion_type] * (pl - 1))
                for ion_type in MODELS[self.model]["ion_types"]
            ]
            ionnumbers.extend([x + 1 for x in range(pl - 1)] * num_ion_types)
            charges.extend([charge_bufs[pi]] * (num_ion_types * (pl - 1)))
            pepids.extend([spec_id_bufs[pi]] * (num_ion_types * (pl - 1)))
            psm_ids.extend([psm_id_bufs[pi]] * (num_ion_types * (pl - 1)))
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

        all_preds["psm_id"] = psm_ids

        return all_preds[["psm_id", "spec_id", "charge", "ion", "ionnumber", "mz", "prediction", "target"]]

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
