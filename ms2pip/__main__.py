import logging
import sys
from pathlib import Path
from typing import Optional, Union

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

import click
from rich.console import Console
from rich.logging import RichHandler

import ms2pip.core
from ms2pip.constants import MODELS, SUPPORTED_OUTPUT_FORMATS
from ms2pip.exceptions import (
    FragmentationModelRequiredError,
    InvalidModificationFormattingError,
    InvalidPEPRECError,
    InvalidXGBoostModelError,
    NoValidPeptideSequencesError,
    UnknownModelError,
    UnknownModificationError,
    UnknownOutputFormatError,
)

__version__ = importlib_metadata.version("ms2pip")

logger = logging.getLogger(__name__)


def print_logo():
    print(
        f"\nMS²PIP v{__version__}\n"
        "CompOmics, VIB / Ghent University, Belgium\n"
        "https://github.com/compomics/ms2pip\n"
    )


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
@click.version_option(version=__version__)
def cli(*args, **kwargs):
    pass


@cli.command(help=ms2pip.core.predict_single.__doc__)
def predict_single(*args, **kwargs):
    ms2pip.core.predict_single(*args, **kwargs)


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
    # TODO: Support other output formats (Requires access to PEPREC? Or redesign output object?)
    output_filename = output_name.with_suffix(".csv")
    logger.info(f"Writing output to {output_filename}")
    predictions.to_csv(output_filename, index=False)


@cli.command(help=ms2pip.core.predict_library.__doc__)
def predict_library(*args, **kwargs):
    ms2pip.core.predict_library(*args, **kwargs)


@cli.command(help=ms2pip.core.correlate.__doc__)
@click.argument("psms", required=True)
@click.argument("spectrum_file", required=True)
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
    intensities, correlations = ms2pip.core.correlate(*args, **kwargs)

    # Write output
    output_filenam_int = output_name.with_suffix("_intensities.csv")
    logger.info(f"Writing intensities to {output_filenam_int}")
    intensities.to_csv(output_filenam_int, index=False)
    if correlations:
        output_filename_corr = output_name.with_suffix("_correlations.csv")
        logger.info(f"Writing correlations to {output_filename_corr}")
        correlations.to_csv(output_filename_corr, index=False)


@cli.command(help=ms2pip.core.get_training_data.__doc__)
def get_training_data(*args, **kwargs):
    ms2pip.core.get_training_data(*args, **kwargs)


@cli.command(help=ms2pip.core.match_spectra.__doc__)
def match_spectra(*args, **kwargs):
    ms2pip.core.match_spectra(*args, **kwargs)


def main():
    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            RichHandler(rich_tracebacks=True, console=Console(), show_level=True, show_path=False)
        ],
    )
    logger = logging.getLogger(__name__)

    # print_logo()

    try:
        cli()

    except InvalidPEPRECError:
        logger.critical("PEPREC file should start with header column")
        sys.exit(1)
    except NoValidPeptideSequencesError:
        logger.critical(
            "No peptides for which to predict intensities. \
            please provide at least one valid peptide sequence."
        )
        sys.exit(1)
    except UnknownModificationError as e:
        logger.critical("Unknown modification: %s", e)
        sys.exit(1)
    except InvalidModificationFormattingError as e:
        logger.critical("Invalid formatting of modifications: %s", e)
        sys.exit(1)
    except UnknownOutputFormatError as o:
        logger.critical(
            f"Unknown output format: `{o}` (supported formats: `{SUPPORTED_OUTPUT_FORMATS}`)"
        )
        sys.exit(1)
    except UnknownModelError as f:
        logger.critical(f"Unknown model: `{f}` (supported models: {set(MODELS.keys())})")
        sys.exit(1)
    except FragmentationModelRequiredError:
        logger.critical("Please specify model in config file.")
        sys.exit(1)
    except InvalidXGBoostModelError:
        logger.critical(f"Could not download XGBoost model properly\nTry manual download")
        sys.exit(1)


if __name__ == "__main__":
    main()
