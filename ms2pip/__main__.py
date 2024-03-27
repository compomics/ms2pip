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
from ms2pip import __version__
from ms2pip._utils.cli import build_credits, build_prediction_table
from ms2pip.constants import MODELS, SUPPORTED_OUTPUT_FORMATS
from ms2pip.exceptions import (
    InvalidXGBoostModelError,
    UnknownModelError,
    UnknownOutputFormatError,
    UnresolvableModificationError,
)
from ms2pip.result import correlations_to_csv, results_to_csv
from ms2pip.spectrum_output import write_single_spectrum_csv, write_single_spectrum_png

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
        return Path(input_filename).with_suffix("")


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
@click.option("--model", type=click.Choice(MODELS), default="HCD")
@click.option("--model-dir")
@click.option("--plot", "-p", is_flag=True)
def predict_single(*args, **kwargs):
    # Parse arguments
    output_name = kwargs.pop("output_name")
    plot = kwargs.pop("plot")
    if not output_name:
        output_name = "ms2pip_prediction_" + secure_filename(kwargs["peptidoform"]) + ".csv"

    # Predict spectrum
    result = ms2pip.core.predict_single(*args, **kwargs)
    predicted_spectrum, _ = result.as_spectra()

    # Write output
    console.print(build_prediction_table(predicted_spectrum))
    write_single_spectrum_csv(predicted_spectrum, output_name)
    if plot:
        write_single_spectrum_png(predicted_spectrum, output_name)


@cli.command(help=ms2pip.core.predict_batch.__doc__)
@click.argument("psms", required=True)
@click.option("--output-name", "-o", type=str)
@click.option("--output-format", "-f", type=click.Choice(SUPPORTED_OUTPUT_FORMATS))
@click.option("--add-retention-time", "-r", is_flag=True)
@click.option("--model", type=click.Choice(MODELS), default="HCD")
@click.option("--model-dir")
@click.option("--processes", "-n", type=int)
def predict_batch(*args, **kwargs):
    # Parse arguments
    output_name = kwargs.pop("output_name")
    output_format = kwargs.pop("output_format")
    output_name = _infer_output_name(kwargs["psms"], output_name)

    # Run
    predictions = ms2pip.core.predict_batch(*args, **kwargs)

    # Write output
    output_name_csv = output_name.with_name(output_name.stem + "_predictions").with_suffix(".csv")
    logger.info(f"Writing output to {output_name_csv}")
    results_to_csv(predictions, output_name_csv)
    # TODO: add support for other output formats


@cli.command(help=ms2pip.core.predict_library.__doc__)
def predict_library(*args, **kwargs):
    ms2pip.core.predict_library(*args, **kwargs)


@cli.command(help=ms2pip.core.correlate.__doc__)
@click.argument("psms", required=True)
@click.argument("spectrum_file", required=True)
@click.option("--psm-filetype", "-t", type=click.Choice(PSM_FILETYPES), default=None)
@click.option("--output-name", "-o", type=str)
@click.option("--spectrum-id-pattern", "-p")
@click.option("--compute-correlations", "-x", is_flag=True)
@click.option("--add-retention-time", "-r", is_flag=True)
@click.option("--model", type=click.Choice(MODELS), default="HCD")
@click.option("--model-dir")
@click.option("--ms2-tolerance", type=float, default=0.02)
@click.option("--processes", "-n", type=int)
def correlate(*args, **kwargs):
    # Parse arguments
    output_name = kwargs.pop("output_name")
    output_name = _infer_output_name(kwargs["psms"], output_name)

    # Run
    results = ms2pip.core.correlate(*args, **kwargs)

    # Write output
    output_name_int = output_name.with_name(output_name.stem + "_predictions").with_suffix(".csv")
    logger.info(f"Writing intensities to {output_name_int}")
    results_to_csv(results, output_name_int)
    # TODO: add support for other output formats

    # Write correlations
    if kwargs["compute_correlations"]:
        output_name_corr = output_name.with_name(output_name.stem + "_correlations")
        output_name_corr = output_name_corr.with_suffix(".csv")
        logger.info(f"Writing correlations to {output_name_corr}")
        correlations_to_csv(results, output_name_corr)


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

    # Write output
    output_name_int = output_name.with_name(output_name.stem + "_observations").with_suffix(".csv")
    logger.info(f"Writing intensities to {output_name_int}")
    results_to_csv(results, output_name_int)


def main():
    try:
        cli()
    except UnresolvableModificationError as e:
        logger.critical(
            "Unresolvable modification: `%s`. See "
            "https://ms2pip.readthedocs.io/en/stable/usage/#amino-acid-modifications "
            "for more info.",
            e,
        )
        sys.exit(1)
    except UnknownOutputFormatError as o:
        logger.critical(
            f"Unknown output format: `{o}` (supported formats: `{SUPPORTED_OUTPUT_FORMATS}`)"
        )
        sys.exit(1)
    except UnknownModelError as f:
        logger.critical(f"Unknown model: `{f}` (supported models: {set(MODELS.keys())})")
        sys.exit(1)
    except InvalidXGBoostModelError:
        logger.critical("Could not correctly download XGBoost model\nTry a manual download.")
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected error occurred in MS²PIP.")
        sys.exit(1)


if __name__ == "__main__":
    main()
