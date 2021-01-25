import argparse
import logging
import multiprocessing
import sys

from ms2pip.config_parser import ConfigParser
from ms2pip.exceptions import (FragmentationModelRequiredError,
                               InvalidModificationFormattingError,
                               InvalidPEPRECError,
                               NoValidPeptideSequencesError,
                               UnknownFragmentationMethodError,
                               UnknownModificationError,
                               UnknownOutputFormatError)
from ms2pip.ms2pipC import MODELS, MS2PIP, SUPPORTED_OUT_FORMATS


def print_logo():
    logo = """
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
        "-c", "--config-file",
        metavar="CONFIG_FILE",
        action="store",
        required=True,
        dest="config_file",
        help="Configuration file: text-based (extensions `.txt`, `.config`, or `.ms2pip`) or TOML (extension `.toml`).",
    )
    parser.add_argument(
        "-s", "--spectrum-file",
        metavar="MGF_FILE",
        action="store",
        dest="spec_file",
        help=".mgf MS2 spectrum file (optional)",
    )
    parser.add_argument(
        "-w", "--vector-file",
        metavar="FEATURE_VECTOR_OUTPUT",
        action="store",
        dest="vector_file",
        help="write feature vectors to FILE.{pkl,h5} (optional)",
    )
    parser.add_argument(
        "-r", "--retention-time",
        action="store_true",
        default=False,
        dest="add_retention_time",
        help="add retention time predictions (requires DeepLC python package)",
    )
    parser.add_argument(
        "-x", "--correlations",
        action="store_true",
        default=False,
        dest="correlations",
        help="calculate correlations (if MGF is given)",
    )
    parser.add_argument(
        "-m", "--match-spectra",
        action="store_true",
        default=False,
        dest="match_spectra",
        help="match peptides to spectra based on predicted spectra (if MGF is given)",
    )
    parser.add_argument(
        "-t", "--tableau",
        action="store_true",
        default=False,
        dest="tableau",
        help="create Tableau Reader file",
    )
    parser.add_argument(
        "-n", "--num-cpu",
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
        help="use sql database of observed spectra instead of MGF files",
    )
    args = parser.parse_args()

    if not args.num_cpu:
        args.num_cpu = multiprocessing.cpu_count()

    return args


def main():
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

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
            tableau=args.tableau
        )
        try:
            ms2pip.run()
        finally:
            ms2pip.cleanup()
    except InvalidPEPRECError:
        root_logger.error("PEPREC file should start with header column")
        sys.exit(1)
    except NoValidPeptideSequencesError:
        root_logger.error("No peptides for which to predict intensities. \
            please provide at least one valid peptide sequence.")
        sys.exit(1)
    except UnknownModificationError as e:
        root_logger.error("Unknown modification: %s", e)
        sys.exit(1)
    except InvalidModificationFormattingError as e:
        root_logger.error("Invalid formatting of modifications: %s", e)
        sys.exit(1)
    except UnknownOutputFormatError as o:
        root_logger.error("Unknown output format: '%s' (supported formats: %s)", o, SUPPORTED_OUT_FORMATS)
        sys.exit(1)
    except UnknownFragmentationMethodError as f:
        root_logger.error("Unknown fragmentation method: %s (supported methods: %s)", f, MODELS.keys())
        sys.exit(1)
    except FragmentationModelRequiredError:
        root_logger.error("Please specify model in config file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
