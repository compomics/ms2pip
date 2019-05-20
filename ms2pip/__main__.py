from ms2pip.ms2pipC import argument_parser, load_configfile, run


def print_logo():
	logo = """
 _____ _____ ___ _____ _____ _____
|     |   __|_  |  _  |     |  _  |
| | | |__   |  _|   __|-   -|   __|
|_|_|_|_____|___|__|  |_____|__|
		   """
	print(logo)
	print("by sven.degroeve@ugent.be\n")


def main():
	print_logo()
	pep_file, spec_file, vector_file, config_file, num_cpu, tableau = argument_parser()
	params = load_configfile(config_file)
	run(pep_file, spec_file=spec_file, vector_file=vector_file, params=params, num_cpu=num_cpu, tableau=tableau)


if __name__ == "__main__":
	main()
