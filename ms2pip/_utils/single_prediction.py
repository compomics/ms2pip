""""Run MS²PIP prediction for single peptide."""

import logging
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus
import xgboost as xgb

from ms2pip._utils import peptides
from ms2pip._utils.xgb_models import get_predictions_xgb, validate_requested_xgb_model
from ms2pip.core import MODELS
from ms2pip.cython_modules import ms2pip_pyx

logger = logging.getLogger("ms2pip")


class SinglePrediction:
    """Run MS²PIP prediction for single peptide."""

    def __init__(self, modification_strings=None, model_dir=None) -> None:
        """
        Run MS²PIP prediction for single peptide.

        Parameters
        ----------
        modification_strings: list-like
            List of MS²PIP configuration-style modification strings, e.g.
            `Carbamidomethyl,57.02146,opt,C` or `Oxidation,15.994915,opt,M`. See MS²PIP
            README.md for more info.
        model_dir : str, optional
            Custom directory for downloaded XGBoost model files. By default,
            `~/.ms2pip` is used.

        Examples
        --------
        >>> from ms2pip.single_prediction import SinglePrediction
        >>> ms2pip_sp = SinglePrediction(
        >>>     modification_strings=[
        >>>         "Carbamidomethyl,57.021464,opt,C"
        >>>     ]
        >>> )
        >>> mz, intensity, annotation = ms2pip_sp.predict(
        >>>     "GSIGECIAEEEEFELDSESNR", "6|Carbamidomethyl", 3
        >>> )

        """
        if not modification_strings:
            modification_strings = []
        self._init_ms2pip(modification_strings)

        if model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = os.path.join(os.path.expanduser("~"), ".ms2pip")

    def _init_ms2pip(self, modification_strings):
        self.mod_info = peptides.Modifications()
        self.mod_info.modifications = {"ptm": {}, "sptm": {}}
        self.mod_info.add_from_ms2pip_modstrings(modification_strings)
        afile = peptides.write_amino_acid_masses()
        modfile = self.mod_info.write_modifications_file(mod_type="ptm")
        modfile2 = self.mod_info.write_modifications_file(mod_type="sptm")
        ms2pip_pyx.ms2pip_init(afile, modfile, modfile2)

    def predict(
        self,
        peptide,
        modifications,
        charge,
        model="HCD",
    ):
        """
        Predict single peptide spectrum with MS²PIP.

        Parameters
        ----------
        peptide: string
            Unmodified peptide sequence. Only canonical amino acids are allowed, and
            peptide sequence should be of length [3, 100].
        modifications: string
            MS²PIP style-formatted modification string (e.g. `0|Acetyl|5|Oxidation`).
            See MS²PIP README.md for more info.
        charge: int
            Peptide precursor charge
        model: string (default: "HCD")
            MS²PIP model to use, identical to the option in the MS²PIP configuration
            file.

        Returns
        -------
        mz: list[float]
            List with fragment ion m/z values in Da.
        intensity: list[float]
            List with TIC-normalized predicted intensities, order matches `mz`
        annotation: list[str]
            List with fragment ion types and series, order matches `mz`

        """
        peptide_encoded = peptides.encode_peptide(peptide)
        peptidoform_encoded = peptides.apply_modifications(
            peptide_encoded, modifications, self.mod_info.ptm_ids
        )
        model_id = MODELS[model]["id"]
        peaks_version = MODELS[model]["peaks_version"]
        ce = 30

        # Get m/z values
        mz = np.array(ms2pip_pyx.get_mzs(peptidoform_encoded, peaks_version))

        # Get intensity values
        if "xgboost_model_files" in MODELS[model].keys():
            validate_requested_xgb_model(
                MODELS[model]["xgboost_model_files"],
                MODELS[model]["model_hash"],
                self.model_dir,
            )
            xgb_vector = xgb.DMatrix(
                np.array(ms2pip_pyx.get_vector(peptide_encoded, peptidoform_encoded, charge))
            )
            num_ions = [len(peptide) - 1]
            intensity = np.array(
                get_predictions_xgb(xgb_vector, num_ions, MODELS[model], self.model_dir)
            )
        else:
            intensity = np.array(
                ms2pip_pyx.get_predictions(
                    peptide_encoded, peptidoform_encoded, charge, model_id, peaks_version, ce
                )
            )

        # Get peak annotations
        annotation = []
        for ion_type in MODELS[model]["ion_types"]:
            annotation.append(
                [
                    ion_type.lower() + str(i + 1)
                    for i in range(len(mz[MODELS[model]["ion_types"].index(ion_type)]))
                ]
            )
        annotation = np.array(annotation)

        # Reshape arrays
        mz = mz.flatten()
        intensity = self._tic_normalize(self._transform(intensity.flatten()))
        annotation = annotation.flatten()

        return mz, intensity, annotation

    def plot_prediction(
        self,
        peptide,
        modifications,
        charge,
        prediction=None,
        ax=None,
        filename=None,
    ):
        """
        Plot MS²PIP-predicted spectrum with spectrum_utils.

        Parameters
        ----------
        peptide: string
            Unmodified peptide sequence. Only canonical amino acids are allowed, and
            peptide sequence should be of length [3, 100].
        modifications: string
            MS²PIP style-formatted modification string (e.g. `0|Acetyl|5|Oxidation`).
            See MS²PIP README.md for more info.
        charge: int
            Peptide precursor charge.
        prediction: tuple or None (default: None)
            Tuple with `ms2pip.single_prediction.SinglePrediction.predict()` output.
        ax: matplotlib.axes.Axes or None (default: None)
            Figure ax to plot onto.
        filename: str or None (default: None)
            Filename to save plot to. File extension defines the format. Figure will
            not be saved if None.

        """
        if not prediction:
            prediction = self.predict(peptide, modifications, charge)
        mz, intensity, annotation = prediction

        identifier = f"{peptide}/{charge}/{modifications}"
        precursor_mz = self.mod_info.calc_precursor_mz(peptide, modifications, charge)
        mod_dict = self._modifications_to_dict(modifications)
        sus_annotation = self._get_sus_annotation(mz, annotation)

        spectrum = sus.MsmsSpectrum(
            identifier,
            precursor_mz,
            charge,
            mz,
            intensity,
            annotation=sus_annotation,
            retention_time=None,
            peptide=peptide,
            modifications=mod_dict,
        )

        if not ax:
            ax = plt.gca()
        sup.spectrum(spectrum, ax=ax)
        ax.set_title("MS²PIP prediction for " + identifier)
        if filename:
            plt.savefig(filename)

    @staticmethod
    def _transform(intensity):
        """Undo MS²PIP peak intensity log transformation and pseudo-count."""
        return (2 ** np.array(intensity)) - 0.001

    @staticmethod
    def _tic_normalize(intensity):
        """TIC-normalize peak intensities."""
        intensity = np.array(intensity)
        return intensity / np.sum(intensity)

    @staticmethod
    def _modifications_to_dict(modifications):
        """Convert ms2pip modification notation to spectrum_utils dict."""

        def parse_loc(loc):
            if loc == "0":
                return "N-term"
            elif loc == "-1":
                return "C-term"
            else:
                return int(loc) - 1

        m_split = [modifications.split("|")[i::2] for i in [0, 1]]
        mods_dict = {parse_loc(loc): name for loc, name in zip(m_split[0], m_split[1])}
        return mods_dict

    @staticmethod
    def _get_sus_annotation(mz, annotation):
        """Get spectrum_utils.PeptideFragmentAnnotation objects."""
        return [
            sus.PeptideFragmentAnnotation(1, mz, annotation[0], annotation[1:])
            for mz, annotation in zip(mz, annotation)
        ]


@click.command()
@click.argument("peptide", type=str)
@click.argument("modifications", type=str)
@click.argument("charge", type=int)
@click.option("-m", "--model", type=str, default="HCD", help="")
@click.option("-c", "--configfile", type=click.Path(exists=True), default=None, help="")
@click.option("--model-dir", type=click.Path(), default=None, help="")
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False),
    default="ms2pip_prediction.png",
    help="",
)
def _main(
    peptide,
    modifications,
    charge,
    model="HCD",
    configfile=None,
    model_dir=None,
    output="ms2pip_prediction.png",
):
    """
    Generate MS²PIP-predicted spectrum and plot.

    \b
    Examples:
     - ms2pip-single-prediction PGAQANPYSR "-" 3
     - ms2pip-single-prediction -o prediction.png PGAQANPYSR "-" 3
     - ms2pip-single-prediction -c config.toml NSVPCSR "5|Carbamidomethyl" 3

    """
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    mod_strings = None
    ms2pip_sp = SinglePrediction(mod_strings, model_dir)

    _, ax = plt.subplots(figsize=(10, 5))
    prediction = ms2pip_sp.predict(peptide, modifications, charge, model=model)
    ms2pip_sp.plot_prediction(
        peptide,
        modifications,
        charge,
        prediction=prediction,
        ax=ax,
        filename=output,
    )


if __name__ == "__main__":
    _main()