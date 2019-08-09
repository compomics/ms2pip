from ms2pip.ms2pipC import argument_parser, load_configfile, run


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
	print_logo()
	pep_file, spec_file, vector_file, config_file, num_cpu, correlations, tableau = argument_parser()
	params = load_configfile(config_file)
	run(pep_file, spec_file=spec_file, vector_file=vector_file, params=params,
	num_cpu=num_cpu, compute_correlations=correlations, tableau=tableau)


if __name__ == "__main__":
	main()
