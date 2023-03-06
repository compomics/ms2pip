import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

from ms2pip.constants import MODELS, SUPPORTED_OUTPUT_FORMATS
from ms2pip.exceptions import (EmptySpectrumError,
                               FragmentationModelRequiredError,
                               InvalidModificationFormattingError,
                               InvalidPEPRECError, InvalidXGBoostModelError,
                               NoValidPeptideSequencesError,
                               UnknownFragmentationMethodError,
                               UnknownModificationError,
                               UnknownOutputFormatError)
from ms2pip.core import MS2PIP


def print_logo():
    logo = r"""
 __  __ ___  __ ___ ___ ___
|  \/  / __||_ ) _ \_ _| _ \
| |\/| \__ \/__|  _/| ||  _/
|_|  |_|___/   |_| |___|_|

by CompOmics, VIB / Ghent University, Belgium

http://compomics.github.io/projects/ms2pip_c.html
    """
    print(logo)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("peptide-file", required=True)
@click.option("--output-filename", "-o")
@click.option("--add-retention-time", "-r", is_flag=True)
@click.option("--config-file", "-c")
@click.option("--output-format", "-f", type=click.Choice(SUPPORTED_OUTPUT_FORMATS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--model-dir")
@click.option("--num-cpu", "-n", type=int)
def predict(*args, **kwargs):
    config = ConfigParser(filepath=kwargs["config_file"]).config
    ms2pip = MS2PIP(
        params=config,
        model=kwargs["model"],
        model_dir=kwargs["model_dir"],
        output_formats=kwargs["output_format"],
        processes=kwargs["processes"],
    )
    try:
        ms2pip.predict(
            peptides=kwargs["peptide_file"],
            output_filename=kwargs["output_filename"],
            add_retention_time=kwargs["add_retention_time"],
        )
    finally:
        ms2pip.cleanup()


@cli.command()
@click.argument("peptide-file", required=True)
@click.argument("spectrum-file", required=True)
@click.option("--spectrum-id-pattern")
@click.option("--output-filename", "-o")
@click.option("--output-format", "-f", type=click.Choice(SUPPORTED_OUTPUT_FORMATS))
@click.option("--add-retention-time", "-r", is_flag=True)
@click.option("--compute-correlations", "-x", is_flag=True)
@click.option("--config-file", "-c")
@click.option("--model", type=click.Choice(MODELS))
@click.option("--model-dir")
@click.option("--num-cpu", "-n", type=int)
def correlate(*args, **kwargs):
    config = ConfigParser(filepath=kwargs["config_file"]).config
    ms2pip = MS2PIP(
        params=config,
        model=kwargs["model"],
        model_dir=kwargs["model_dir"],
        output_formats=kwargs["output_format"],
        processes=kwargs["processes"],
    )
    try:
        ms2pip.correlate(
            peptides=kwargs["peptide_file"],
            spectrum_file=kwargs["spectrum_file"],
            spectrum_id_pattern=kwargs["spectrum_id_pattern"],
            output_filename=kwargs["output_filename"],
            compute_correlations=kwargs["compute_correlations"],
            add_retention_time=kwargs["add_retention_time"],
        )
    finally:
        ms2pip.cleanup()

@cli.command()
@click.argument("peptide-file", required=True)
@click.argument("spectrum-file", required=True)
@click.option("--spectrum-id-pattern")
@click.option("--output-filename", "-o")
@click.option("--config-file", "-c")
@click.option("--num-cpu", "-n", type=int)
def get_features(*args, **kwargs):
    config = ConfigParser(filepath=kwargs["config_file"]).config
    ms2pip = MS2PIP(
        params=config,
        output_formats=kwargs["output_format"],
        processes=kwargs["processes"],
    )
    try:
        ms2pip.get_features(
            peptides=kwargs["peptide_file"],
            spectrum_file=kwargs["spectrum_file"],
            spectrum_id_pattern=kwargs["spectrum_id_pattern"],
            output_filename=kwargs["output_filename"],
        )
    finally:
        ms2pip.cleanup()

@cli.command()
@click.argument("peptide-file", required=True)
@click.option("--spectrum-file")
@click.option("--sqldb-uri")
@click.option("--config-file", "-c")
@click.option("--num-cpu", "-n", type=int)
def match_spectra(*args, **kwargs):
    config = ConfigParser(filepath=kwargs["config_file"]).config
    ms2pip = MS2PIP(
        params=config,
        output_formats=kwargs["output_format"],
        processes=kwargs["processes"],
    )
    try:
        ms2pip.get_features(
            peptides=kwargs["peptide_file"],
            spectrum_file=kwargs["spectrum_file"],
            sqldb_uri=kwargs["sqldb_uri"],
        )
    finally:
        ms2pip.cleanup()


def main():
    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[RichHandler(
            rich_tracebacks=True, console=Console(), show_level=True, show_path=False
        )],
    )
    logger = logging.getLogger(__name__)

    print_logo()

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
    except UnknownFragmentationMethodError as f:
        logger.critical(
            f"Unknown model: `{f}` (supported models: {MODELS.keys()})"
        )
        sys.exit(1)
    except FragmentationModelRequiredError:
        logger.critical("Please specify model in config file.")
        sys.exit(1)
    except InvalidXGBoostModelError:
        logger.critical(
            f"Could not download XGBoost model properly\nTry manual download"
        )
        sys.exit(1)
    except EmptySpectrumError:
        logger.critical("Provided MGF file cannot contain empty spectra")
        sys.exit(1)


if __name__ == "__main__":
    main()
