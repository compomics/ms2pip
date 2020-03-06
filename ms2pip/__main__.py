import argparse
import logging
import multiprocessing

from ms2pip.exceptions import (
    InvalidPEPRECError,
    NoValidPeptideSequencesError,
    UnknownOutputFormatError,
    UnknownFragmentationMethodError,
    FragmentationModelRequiredError)
from ms2pip.ms2pipC import load_configfile, run, SUPPORTED_OUT_FORMATS, MODELS


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
        "-c",
        metavar="CONFIG_FILE",
        action="store",
        required=True,
        dest="config_file",
        help="config file",
    )
    parser.add_argument(
        "-s",
        metavar="MGF_FILE",
        action="store",
        dest="spec_file",
        help=".mgf MS2 spectrum file (optional)",
    )
    parser.add_argument(
        "-w",
        metavar="FEATURE_VECTOR_OUTPUT",
        action="store",
        dest="vector_file",
        help="write feature vectors to FILE.{pkl,h5} (optional)",
    )
    parser.add_argument(
        "-x",
        action="store_true",
        default=False,
        dest="correlations",
        help="calculate correlations (if MGF is given)",
    )
    parser.add_argument(
        "-p",
        action="store_true",
        default=False,
        dest="match_spectra",
        help="match peptides to spectra based on predicted spectra (if MGF is given)",
    )
    parser.add_argument(
        "-t",
        action="store_true",
        default=False,
        dest="tableau",
        help="create Tableau Reader file",
    )
    parser.add_argument(
        "-m",
        metavar="NUM_CPU",
        action="store",
        dest="num_cpu",
        help="number of CPUs to use (default: all available)",
    )
    args = parser.parse_args()

    if not args.num_cpu:
        args.num_cpu = multiprocessing.cpu_count()

    return args


def main():
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    print_logo()
    args = argument_parser()
    params = load_configfile(args.config_file)
    try:
        run(args.pep_file,
            spec_file=args.spec_file,
            vector_file=args.vector_file,
            params=params,
            num_cpu=args.num_cpu,
            compute_correlations=args.correlations,
            match_spectra=args.match_spectra,
            tableau=args.tableau)
    except InvalidPEPRECError:
        root_logger.error("PEPREC file should start with header column")
    except NoValidPeptideSequencesError:
        root_logger.error("No peprides for which to predict intensities. \
            please provide at least one valid peptide sequence.")
    except UnknownOutputFormatError as o:
        root_logger.error("Unknown output format: '%s' (supported formats: %s)", o, SUPPORTED_OUT_FORMATS)
    except UnknownFragmentationMethodError as f:
        root_logger.error("Unknown fragmentation method: %s (supported methods: %s)", f, MODELS.keys())
    except FragmentationModelRequiredError:
        root_logger.error("Please specify model in config file.")


if __name__ == "__main__":
    main()
