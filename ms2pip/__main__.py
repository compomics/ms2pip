import logging
import sys
from pathlib import Path
from typing import Optional

import click
from psm_utils.io import READERS
from rich.console import Console
from rich.logging import RichHandler
from werkzeug.utils import secure_filename

import ms2pip.core
from ms2pip import __version__, exceptions
from ms2pip._utils.cli import build_credits, build_prediction_table
from ms2pip.constants import MODELS
from ms2pip.plot import spectrum_to_png
from ms2pip.result import write_correlations
from ms2pip.spectrum_output import SUPPORTED_FORMATS, write_spectra

console = Console()
logger = logging.getLogger(__name__)

LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

PSM_FILETYPES = list(READERS.keys())


def _infer_output_name(
    input_filename: str,
    output_name: Optional[str] = None,
) -> Path:
    """Infer output filename from input filename if output_filename was not defined."""
    if output_name:
        return Path(output_name)
    else:
        input__filename = Path(input_filename)
        return input__filename.with_name(input__filename.stem + "_predictions").with_suffix("")


@click.group()
@click.option("--logging-level", "-l", type=click.Choice(LOGGING_LEVELS.keys()), default="INFO")
@click.version_option(version=__version__)
def cli(*args, **kwargs):
    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=LOGGING_LEVELS[kwargs["logging_level"]],
        handlers=[
            RichHandler(rich_tracebacks=True, console=console, show_level=True, show_path=False)
        ],
    )
    console.print(build_credits())


@cli.command(help=ms2pip.core.predict_single.__doc__)
@click.argument("peptidoform", required=True)
@click.option("--output-name", "-o", type=str)
@click.option("--output-format", "-f", type=click.Choice(SUPPORTED_FORMATS), default="tsv")
@click.option("--model", type=click.Choice(MODELS), default="HCD")
@click.option("--model-dir")
@click.option("--plot", "-p", is_flag=True)
def predict_single(*args, **kwargs):
    # Parse arguments
    output_name = kwargs.pop("output_name")
    output_format = kwargs.pop("output_format")
    plot = kwargs.pop("plot")
    if not output_name:
        output_name = "ms2pip_prediction_" + secure_filename(kwargs["peptidoform"])

    # Predict spectrum
    result = ms2pip.core.predict_single(*args, **kwargs)
    predicted_spectrum, _ = result.as_spectra()

    # Write output
    console.print(build_prediction_table(predicted_spectrum))
    write_spectra(output_name, [result], output_format)
    if plot:
        spectrum_to_png(predicted_spectrum, output_name)


@cli.command(help=ms2pip.core.predict_batch.__doc__)
@click.argument("psms", required=True)
@click.option("--psm-filetype", "-t", type=click.Choice(PSM_FILETYPES), default=None)
@click.option("--output-name", "-o", type=str)
@click.option("--output-format", "-f", type=click.Choice(SUPPORTED_FORMATS), default="tsv")
@click.option("--add-retention-time", "-r", is_flag=True)
@click.option("--add-ion-mobility", "-i", is_flag=True)
@click.option("--model", type=click.Choice(MODELS), default="HCD")
@click.option("--model-dir")
@click.option("--processes", "-n", type=int)
def predict_batch(*args, **kwargs):
    # Parse arguments
    output_format = kwargs.pop("output_format")
    output_name = _infer_output_name(kwargs["psms"], kwargs.pop("output_name"))

    # Run
    predictions = ms2pip.core.predict_batch(*args, **kwargs)

    # Write output
    write_spectra(output_name, predictions, output_format)


@cli.command(help=ms2pip.core.predict_library.__doc__)
@click.argument("fasta-file", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-name", "-o", type=str)
@click.option("--output-format", "-f", type=click.Choice(SUPPORTED_FORMATS), default="msp")
@click.option("--add-retention-time", "-r", is_flag=True)
@click.option("--add-ion-mobility", "-i", is_flag=True)
@click.option("--model", type=click.Choice(MODELS), default="HCD")
@click.option("--model-dir")
@click.option("--batch-size", type=int, default=100000)
@click.option("--processes", "-n", type=int)
def predict_library(*args, **kwargs):
    # Parse arguments
    if not kwargs["fasta_file"] and not kwargs["config"]:
        raise click.UsageError("Either `fasta_file` or `config` must be provided.")
    output_format = kwargs.pop("output_format")
    output_name = _infer_output_name(
        kwargs["fasta_file"] or kwargs["config"], kwargs.pop("output_name")
    )

    # Run and write output for each batch
    for i, result_batch in enumerate(ms2pip.core.predict_library(*args, **kwargs)):
        write_spectra(output_name, result_batch, output_format, write_mode="w" if i == 0 else "a")


@cli.command(help=ms2pip.core.correlate.__doc__)
@click.argument("psms", required=True)
@click.argument("spectrum_file", required=True)
@click.option("--psm-filetype", "-t", type=click.Choice(PSM_FILETYPES), default=None)
@click.option("--output-name", "-o", type=str)
@click.option("--spectrum-id-pattern", "-p")
@click.option("--compute-correlations", "-x", is_flag=True)
@click.option("--add-retention-time", "-r", is_flag=True)
@click.option("--add-ion-mobility", "-i", is_flag=True)
@click.option("--model", type=click.Choice(MODELS), default="HCD")
@click.option("--model-dir")
@click.option("--ms2-tolerance", type=float, default=0.02)
@click.option("--processes", "-n", type=int)
def correlate(*args, **kwargs):
    # Parse arguments
    output_name = _infer_output_name(kwargs["psms"], kwargs.pop("output_name"))

    # Run
    results = ms2pip.core.correlate(*args, **kwargs)

    # Write intensities
    logger.info(f"Writing intensities to {output_name.with_suffix('.tsv')}")
    write_spectra(output_name, results, "tsv")

    # Write correlations
    if kwargs["compute_correlations"]:
        output_name_corr = output_name.with_name(output_name.stem + "_correlations").with_suffix(
            ".tsv"
        )
        logger.info(f"Writing correlations to {output_name_corr}")
        write_correlations(results, output_name_corr)


@cli.command(help=ms2pip.core.get_training_data.__doc__)
@click.argument("psms", required=True)
@click.argument("spectrum_file", required=True)
@click.option("--psm-filetype", "-t", type=click.Choice(PSM_FILETYPES), default=None)
@click.option("--output-name", "-o", type=str)
@click.option("--spectrum-id-pattern", "-p")
@click.option("--model", type=click.Choice(MODELS), default="HCD")
@click.option("--ms2-tolerance", type=float, default=0.02)
@click.option("--processes", "-n", type=int)
def get_training_data(*args, **kwargs):
    # Parse arguments
    output_name = kwargs.pop("output_name")
    output_name = _infer_output_name(kwargs["psms"], output_name).with_suffix(".feather")

    # Run
    training_data = ms2pip.core.get_training_data(*args, **kwargs)

    # Write output
    logger.info(f"Writing training data to {output_name}")
    training_data.to_feather(output_name)


@cli.command(help=ms2pip.core.annotate_spectra.__doc__)
@click.argument("psms", required=True)
@click.argument("spectrum_file", required=True)
@click.option("--psm-filetype", "-t", type=click.Choice(PSM_FILETYPES), default=None)
@click.option("--output-name", "-o", type=str)
@click.option("--spectrum-id-pattern", "-p")
@click.option("--model", type=click.Choice(MODELS), default="HCD")
@click.option("--ms2-tolerance", type=float, default=0.02)
@click.option("--processes", "-n", type=int)
def annotate_spectra(*args, **kwargs):
    # Parse arguments
    output_name = kwargs.pop("output_name")
    output_name = _infer_output_name(kwargs["psms"], output_name)

    # Run
    results = ms2pip.core.annotate_spectra(*args, **kwargs)

    # Write intensities
    output_name_int = output_name.with_name(output_name.stem + "_observations").with_suffix()
    logger.info(f"Writing intensities to {output_name_int.with_suffix('.tsv')}")
    write_spectra(output_name, results, "tsv")


def main():
    try:
        cli()
    except exceptions.UnresolvableModificationError as e:
        logger.critical(
            "Unresolvable modification: `%s`. See "
            "https://ms2pip.readthedocs.io/en/stable/usage/#amino-acid-modifications "
            "for more info.",
            e,
        )
        sys.exit(1)
    except exceptions.UnknownOutputFormatError as o:
        logger.critical(f"Unknown output format: `{o}` (supported formats: `{SUPPORTED_FORMATS}`)")
        sys.exit(1)
    except exceptions.UnknownModelError as f:
        logger.critical(f"Unknown model: `{f}` (supported models: {set(MODELS.keys())})")
        sys.exit(1)
    except exceptions.InvalidXGBoostModelError:
        logger.critical("Could not correctly download XGBoost model\nTry a manual download.")
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected error occurred in MS²PIP.")
        sys.exit(1)


if __name__ == "__main__":
    main()
