import logging

from ms2pip.ms2pipC import (argument_parser, load_configfile, run,
                            InvalidPEPREC, NoValidPeptideSequences)


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
    except InvalidPEPREC:
        root_logger.error("PEPREC file should start with header column")
    except NoValidPeptideSequences:
        root_logger.error("No peprides for which to predict intensities. \
            please provide at least one valid peptide sequence.")


if __name__ == "__main__":
    main()
