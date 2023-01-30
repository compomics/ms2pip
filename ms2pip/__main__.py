import argparse
import logging
import multiprocessing
import sys

from rich.console import Console
from rich.logging import RichHandler

from ms2pip.config_parser import ConfigParser
from ms2pip.exceptions import (EmptySpectrumError,
                               FragmentationModelRequiredError,
                               InvalidModificationFormattingError,
                               InvalidPEPRECError, InvalidXGBoostModelError,
                               NoValidPeptideSequencesError,
                               UnknownFragmentationMethodError,
                               UnknownModificationError,
                               UnknownOutputFormatError)
from ms2pip.ms2pipC import MODELS, MS2PIP, SUPPORTED_OUT_FORMATS


def print_logo():
    logo = r"""
 __  __ ___  __ ___ ___ ___
|  \/  / __||_ ) _ \_ _| _ \
| |\/| \__ \/__|  _/| ||  _/
|_|  |_|___/   |_| |___|_|

by CompOmics
sven.degroeve@ugent.be
ralf.gabriels@ugent.be

http://compomics.github.io/projects/ms2pip_c.html
    """
    print(logo)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("pep_file", metavar="<PEPREC file>", help="list of peptides")
    parser.add_argument(
        "-c",
        "--config-file",
        metavar="CONFIG_FILE",
        action="store",
        required=True,
        dest="config_file",
        help="Configuration file: text-based (extensions `.txt`, `.config`, or `.ms2pip`) or TOML (extension `.toml`).",
    )
    parser.add_argument(
        "-s",
        "--spectrum-file",
        metavar="SPECTRUM_FILE",
        action="store",
        dest="spec_file",
        help="MGF or mzML spectrum file (optional)",
    )
    parser.add_argument(
        "-w",
        "--vector-file",
        metavar="FEATURE_VECTOR_OUTPUT",
        action="store",
        dest="vector_file",
        help="write feature vectors to FILE.{pkl,h5} (optional)",
    )
    parser.add_argument(
        "-r",
        "--retention-time",
        action="store_true",
        default=False,
        dest="add_retention_time",
        help="add retention time predictions (requires DeepLC python package)",
    )
    parser.add_argument(
        "-x",
        "--correlations",
        action="store_true",
        default=False,
        dest="correlations",
        help="calculate correlations (if spectrum file is given)",
    )
    parser.add_argument(
        "-m",
        "--match-spectra",
        action="store_true",
        default=False,
        dest="match_spectra",
        help="match peptides to spectra based on predicted spectra (if spectrum file is given)",
    )
    parser.add_argument(
        "-n",
        "--num-cpu",
        metavar="NUM_CPU",
        action="store",
        dest="num_cpu",
        type=int,
        help="number of CPUs to use (default: all available)",
    )
    parser.add_argument(
        "--sqldb-uri",
        action="store",
        dest="sqldb_uri",
        help="use sql database of observed spectra instead of spectrum files",
    )
    parser.add_argument(
        "--model-dir",
        action="store",
        dest="model_dir",
        help="Custom directory for downloaded XGBoost model files, default: `~/.ms2pip`",
    )
    args = parser.parse_args()

    if not args.num_cpu:
        args.num_cpu = multiprocessing.cpu_count()

    return args


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

    args = argument_parser()
    config_parser = ConfigParser(filepath=args.config_file)

    try:
        ms2pip = MS2PIP(
            args.pep_file,
            spec_file=args.spec_file,
            vector_file=args.vector_file,
            params=config_parser.config,
            num_cpu=args.num_cpu,
            add_retention_time=args.add_retention_time,
            compute_correlations=args.correlations,
            match_spectra=args.match_spectra,
            sqldb_uri=args.sqldb_uri,
            model_dir=args.model_dir,
        )
        try:
            ms2pip.run()
        finally:
            ms2pip.cleanup()
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
            f"Unknown output format: `{o}` (supported formats: `{SUPPORTED_OUT_FORMATS}`)"
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
