import logging

from ms2pip.ms2pipC import (argument_parser, load_configfile, run,
                            InvalidPEPRECError, NoValidPeptideSequencesError,
                            UnknownOutputFormatError,
                            UnknownFragmentationMethodError,
                            FragmentationModelRequiredError,
                            SUPPORTED_OUT_FORMATS, MODELS)


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


def main():
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    print_logo()
    pep_file, spec_file, vector_file, config_file, num_cpu, correlations, tableau = argument_parser()
    params = load_configfile(config_file)
    try:
        run(pep_file,
            spec_file=spec_file,
            vector_file=vector_file,
            params=params,
            num_cpu=num_cpu,
            compute_correlations=correlations,
            tableau=tableau)
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
