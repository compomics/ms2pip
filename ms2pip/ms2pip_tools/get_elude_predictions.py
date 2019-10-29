"""
ELUDE RT prediction integration for MS2PIP

ELUDE needs to be installed and callable with os.system(elude). Tested with
ELUDE v3.2.

Can be run as a script or by calling the `get_elude_predictions` function.
Both the script and the function take an MS2PIP PEPREC file and an ELUDE model
file.

If the given PEPREC contains PTMs, these need to be configured in the (for now)
hardcoded dictionary `unimod_mapping`. This is a dictionary where the keys
represent the name of the PTM in the PEPREC file and the values contain the
corresponding UniMod IDs.
"""


__author__ = "Ralf Gabriels"
__credits__ = ["Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__email__ = "Ralf.Gabriels@ugent.be"


# Standard library
import os
import argparse

# Third party packages
import pandas as pd


def argument_parser():
	parser = argparse.ArgumentParser(description='Add ELUDE retention time predictions to existing PEPREC file.')
	parser.add_argument("peprec_file", metavar="<peprec file>",
						help="MS2PIP PEPREC file to add RT predictions to.")
	parser.add_argument('elude_model_file', metavar="<elude model file>",
						help='ELUDE model file.')
	args = parser.parse_args()
	return args


def elude_insert_mods(row, peptide_column='peptide', mods_column='modifications',
					  unimod_mapping=None):
	"""
	Insert PEPREC modifications into peptide sequence for ELUDE.

	Accepts normal, N-terminal and C-terminal modifications.

	Positional arguments:
	row -- pandas.DataFrame row, for use with .apply(elude_insert_mods, axis=1)

	Keyword arguments:
	peptide_column -- Column name of column with peptide sequences
	mods_column -- Column name of column with MS2PIP PEPREC encoded
	modifications
	unimod_mapping -- Dictionary that maps the MS2PIP modification names to
	UniMod accessions
	"""

	if not unimod_mapping:
		unimod_mapping = {'Oxidation': 35, 'Carbamidomethyl': 4}

	peptide = row[peptide_column]
	mods = row[mods_column]

	if type(mods) == str:
		if mods == '-':
			peptide_mods = peptide

		else:
			mods = mods.split('|')
			pos, names = [list(tup) for tup in zip(*sorted(zip([int(p) for p in mods[::2]], mods[1::2])))]

			# Add 0 to pos, if no N-term PTMs present
			if pos[0] != 0:
				pos.insert(0, 0)
				names.insert(0, '')

			# Replace "-1" index from C-term PTMs to index of last aa
			if pos[-1] == -1:
				pos[-1] = len(peptide)

			pep_split = [peptide[i:j] for i, j in zip(list(pos), list(pos)[1:] + [None])]
			peptide_mods = ''.join([''.join(tup) for tup in zip(['[unimod:{}]'.format(unimod_mapping[n])
									if n else '' for n in names], pep_split)])

	else:
		peptide_mods = peptide

	return peptide_mods


def get_elude_predictions(peprec, elude_model_file, **kwargs):
	"""
	Return ELUDE retention time predictions for peptide sequences in MS2PIP PEPREC.

	ELUDE needs to be installed and callable with os.system(elude). Tested with ELUDE v3.2

	Positional arguments:
	peprec -- MS2PIP PEPREC in pandas.DataFrame()
	elude_model_file -- filename of ELUDE model to apply

	kwargs -- keyword arguments are passed to elude_insert_mods
	"""

	filename_in = '{}_Test.txt'.format(elude_model_file)
	filename_out = '{}_Preds.txt'.format(elude_model_file)
	filename_model = elude_model_file

	peprec.apply(elude_insert_mods, **kwargs, axis=1)\
		  .to_csv(filename_in, sep=' ', index=False, header=False)
	os.system('elude -l "{}" -e "{}" -o "{}" -p'.format(filename_model, filename_in, filename_out))
	preds = pd.read_csv(filename_out, sep='\t', comment='#')
	os.system('rm {}; rm {}'.format(filename_in, filename_out))

	return preds['Predicted_RT']


def main():
	args = argument_parser()

	# HARD CODED, for now
	unimod_mapping = {
		# PEPREC PTM name: UniMod ID,
		'Oxidation': 35,
		'Carbamidomethyl': 4
	}

	print("Adding ELUDE RT predictions to {}...".format(args.peprec_file))

	peprec = pd.read_csv(
		args.peprec_file, sep=" ", index_col=False,
		dtype={"spec_id": str, "modifications": str}
	)

	# Remove `* 60` to get RTs in seconds
	peprec['rt'] = get_elude_predictions(peprec, args.elude_model_file, unimod_mapping=unimod_mapping) # * 60

	output_filename = '.'.join(args.peprec_file.split('.')[:-1]) + '_with_rt.peprec'
	peprec.to_csv(output_filename, sep=' ', index=False)

	print("Ready!")


if __name__ == '__main__':
	main()
