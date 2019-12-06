#!/usr/bin/env python
# Native library
import sys
import argparse
import multiprocessing
from random import shuffle
import tempfile

# Third party
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# From project
from ms2pip.ms2pip_tools import spectrum_output, calc_correlations
from ms2pip.feature_names import get_feature_names_new
from ms2pip.cython_modules import ms2pip_pyx


# Supported output formats
SUPPORTED_OUT_FORMATS = ['csv', 'mgf', 'msp', 'bibliospec', 'spectronaut']

# Models and their properties
# id is passed to get_predictions to select model
# ion_types is required to write the ion types in the headers of the result files
# features_version is required to select the features version
MODELS = {
	'CID': {'id': 0, 'ion_types': ['B', 'Y'], 'peaks_version': 'general', 'features_version': 'normal'},
	'HCD': {'id': 1, 'ion_types': ['B', 'Y'], 'peaks_version': 'general', 'features_version': 'normal'},
	'TTOF5600': {'id': 2, 'ion_types': ['B', 'Y'], 'peaks_version': 'general', 'features_version': 'normal'},
	'TMT': {'id': 3, 'ion_types': ['B', 'Y'], 'peaks_version': 'general', 'features_version': 'normal'},
	'iTRAQ': {'id': 4, 'ion_types': ['B', 'Y'], 'peaks_version': 'general', 'features_version': 'normal'},
	'iTRAQphospho': {'id': 5, 'ion_types': ['B', 'Y'], 'peaks_version': 'general', 'features_version': 'normal'},
	#ETD': {'id': 6, 'ion_types': ['B', 'Y', 'C', 'Z'], 'peaks_version': 'etd', 'features_version': 'normal'},
	'HCDch2': {'id': 7, 'ion_types': ['B', 'Y', 'B2', 'Y2'], 'peaks_version': 'ch2', 'features_version': 'normal'},
	'CIDch2': {'id': 8, 'ion_types': ['B', 'Y', 'B2', 'Y2'], 'peaks_version': 'ch2', 'features_version': 'normal'},
}

# Create A_MAP:
# A_MAP converts the peptide amino acids to integers, note how "L" is removed
AMINOS = [
	"A", "C", "D", "E", "F", "G", "H", "I", "K", "M",
	"N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]
MASSES = [
	71.037114, 103.00919, 115.026943, 129.042593, 147.068414,
	57.021464, 137.058912, 113.084064, 128.094963, 131.040485,
	114.042927, 97.052764, 128.058578, 156.101111, 87.032028,
	101.047679, 99.068414, 186.079313, 163.063329,
	# 147.0354  # iTRAQ fixed N-term modification (gets written to amino acid masses file)
]
A_MAP = {a: i for i, a in enumerate(AMINOS)}


def process_peptides(worker_num, data, afile, modfile, modfile2, PTMmap, model):
	"""
	Function for each worker to process a list of peptides. The models are
	chosen based on model. PTMmap, Ntermmap and Ctermmap determine the
	modifications applied to each peptide sequence. Returns the predicted
	spectra for all the peptides.
	"""

	ms2pip_pyx.ms2pip_init(bytearray(afile.encode()), bytearray(modfile.encode()), bytearray(modfile2.encode()))

	pcount = 0

	# Prepare output variables
	mz_buf = []
	prediction_buf = []
	peplen_buf = []
	charge_buf = []
	pepid_buf = []

	# transform pandas dataframe into dictionary for easy access
	if "ce" in data.columns:
		specdict = data[["spec_id", "peptide", "modifications", "charge", "ce"]].set_index("spec_id").to_dict()
		ces = specdict["ce"]
	else:
		specdict = data[["spec_id", "peptide", "modifications", "charge"]].set_index("spec_id").to_dict()
	pepids = data['spec_id'].tolist()
	peptides = specdict["peptide"]
	modifications = specdict["modifications"]
	charges = specdict["charge"]
	del specdict

	for pepid in pepids:
		peptide = peptides[pepid]
		peptide = peptide.replace("L", "I")
		mods = modifications[pepid]

		# TODO: Check if 30 is good default CE!
		colen = 30
		if "ce" in data.columns:
			colen = ces[pepid]

		# Peptides longer then 101 lead to "Segmentation fault (core dumped)"
		if len(peptide) > 100:
			continue

		# convert peptide string to integer list to speed up C code
		peptide = np.array([0] + [A_MAP[x] for x in peptide] + [0], dtype=np.uint16)

		modpeptide = apply_mods(peptide, mods, PTMmap)
		if type(modpeptide) == str:
			if modpeptide == "Unknown modification":
				continue

		pepid_buf.append(pepid)
		peplen = len(peptide) -2
		peplen_buf.append(peplen)

		ch = charges[pepid]
		charge_buf.append(ch)

		model_id = MODELS[model]['id']
		peaks_version = MODELS[model]['peaks_version']

		# get ion mzs
		mzs = ms2pip_pyx.get_mzs(modpeptide, peaks_version)
		mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])

		# Predict the b- and y-ion intensities from the peptide
		# For C-term ion types (y, y++, z), flip the order of predictions,
		# because get_predictions follows order from vector file
		# enumerate works for variable number (and all) ion types
		predictions = ms2pip_pyx.get_predictions(peptide, modpeptide, ch, model_id, peaks_version, colen) #SD: added colen
		prediction_buf.append([np.array(p, dtype=np.float32) for p in predictions])

		pcount += 1
		if (pcount % 500) == 0:
			sys.stdout.write("(%i)%i "%(worker_num, pcount))
			sys.stdout.flush()

	return mz_buf, prediction_buf, peplen_buf, charge_buf, pepid_buf


def process_spectra(worker_num, spec_file, vector_file, data, afile, modfile, modfile2, PTMmap, model, fragerror, tableau):
	"""
	Function for each worker to process a list of spectra. Each peptide's
	sequence is extracted from the mgf file. Then models are chosen based on
	model. PTMmap, Ntermmap and Ctermmap determine the modifications
	applied to each peptide sequence and the spectrum is predicted. Then either
	the feature vectors are returned, or a DataFrame with the predicted and
	empirical intensities.
	"""

	ms2pip_pyx.ms2pip_init(bytearray(afile.encode()), bytearray(modfile.encode()), bytearray(modfile2.encode()))

	# transform pandas datastructure into dictionary for easy access
	if "ce" in data.columns:
		specdict = data[["spec_id", "peptide", "modifications", "ce"]].set_index("spec_id").to_dict()
		ces = specdict["ce"]
	else:
		specdict = data[["spec_id", "peptide", "modifications"]].set_index("spec_id").to_dict()
	peptides = specdict["peptide"]
	modifications = specdict["modifications"]

	# cols contains the names of the computed features
	cols_n = get_feature_names_new()
	if "ce" in data.columns:
		cols_n.append("ce")
	#cols_n = get_feature_names_catboost()

	#SD
	dvectors = []
	dtargets = dict()
	psmids = []

	mz_buf = []
	target_buf = []
	prediction_buf = []
	peplen_buf = []
	charge_buf = []
	pepid_buf = []

	if tableau:
		ft = open("ms2pip_tableau.%i"%worker_num, "w")
		ft2 = open("stats_tableau.%i"%worker_num, "w")

	title = ""
	charge = 0
	msms = []
	peaks = []
	pepmass = 0
	f = open(spec_file)
	skip = False
	pcount = 0
	while 1:
		rows = f.readlines(50000)
		if not rows:
			break
		for row in rows:
			row = row.rstrip()
			if row == "":
				continue
			if skip:
				if row[0] == "B":
					if row[:10] == "BEGIN IONS":
						skip = False
				else:
					continue
			if row == "":
				continue
			if row[0] == "T":
				if row[:5] == "TITLE":
					title = row[6:]
					if title not in peptides:
						skip = True
						continue
			elif row[0].isdigit():
				tmp = row.split()
				msms.append(float(tmp[0]))
				peaks.append(float(tmp[1]))
			elif row[0] == "B":
				if row[:10] == "BEGIN IONS":
					msms = []
					peaks = []
			elif row[0] == "C":
				if row[:6] == "CHARGE":
					charge = int(row[7:9].replace("+", ""))
			elif row[0] == "P":
				if row[:7] == "PEPMASS":
					pepmass = float(row.split("=")[1].split(" ")[0])
			elif row[:8] == "END IONS":
				# process current spectrum
				if title not in peptides:
					continue

				peptide = peptides[title]
				peptide = peptide.replace("L", "I")
				mods = modifications[title]

				if "mut" in mods:
					continue

				# Peptides longer then 101 lead to "Segmentation fault (core dumped)"
				if len(peptide) > 100:
					continue

				# convert peptide string to integer list to speed up C code
				peptide = np.array([0] + [A_MAP[x] for x in peptide] + [0], dtype=np.uint16)

				modpeptide = apply_mods(peptide, mods, PTMmap)
				if type(modpeptide) == str:
					if modpeptide == "Unknown modification":
						continue

				# remove reporter ions
				if 'iTRAQ' in model:
					for mi, mp in enumerate(msms):
						if (mp >= 113) & (mp <= 118):
							peaks[mi] = 0

				# TMT6plex: 126.1277, 127.1311, 128.1344, 129.1378, 130.1411, 131.1382
				if 'TMT' in model:
					for mi, mp in enumerate(msms):
						if (mp >= 125) & (mp <= 132):
							peaks[mi] = 0

				#remove percursor peak
				#for mi, mp in enumerate(msms):
				#   if (mp >= pepmass-0.02) & (mp <= pepmass+0.02):
				#       peaks[mi] = 0

				# normalize and convert MS2 peaks
				msms = np.array(msms, dtype=np.float32)
				tic = np.sum(peaks)
				peaks = peaks / tic
				peaks = np.log2(np.array(peaks) + 0.001)
				peaks = peaks.astype(np.float32)

				model_id = MODELS[model]['id']
				peaks_version = MODELS[model]['peaks_version']

				# TODO: Check if 30 is good default CE!
				# RG: removed `if ce == 0` in get_vector, split up into two functions 
				colen = 30
				if "ce" in data.columns:
					try:
						colen = int(float(ces[title]))
					except:
						sys.stderr.write("Could not parse collision energy!\n")
						continue

				if vector_file:
					# get targetsd
					targets = ms2pip_pyx.get_targets(modpeptide, msms, peaks, float(fragerror), peaks_version)
					psmids.extend([title]*(len(targets[0])))
					if "ce" in data.columns:
						dvectors.append(np.array(ms2pip_pyx.get_vector_ce(peptide, modpeptide, charge, colen), dtype=np.uint16)) #SD: added collision energy
					else:
						dvectors.append(np.array(ms2pip_pyx.get_vector(peptide, modpeptide, charge), dtype=np.uint16))

					# Collecting targets to dict; works for variable number of ion types
					# For C-term ion types (y, y++, z), flip the order of targets,
					# for correct order in vectors DataFrame
					for i, t in enumerate(targets):
						if i in dtargets.keys():
							if i % 2 == 0:
								dtargets[i].extend(t)
							else:
								dtargets[i].extend(t[::-1])
						else:
							if i % 2 == 0:
								dtargets[i] = [t]
							else:
								dtargets[i] = [t[::-1]]
				elif tableau:
					numby = 0
					numall = 0
					explainedby = 0
					explainedall = 0
					ts = []
					ps = []
					predictions = ms2pip_pyx.get_predictions(peptide, modpeptide, charge, model_id, peaks_version, colen)
					for m, p in zip(msms, peaks):
						ft.write("%s;%f;%f;;;0\n"%(title, m, 2**p))
					# get targets
					mzs, targets = ms2pip_pyx.get_targets_all(modpeptide, msms, peaks, float(fragerror), "all")
					# get mean by intensity values to normalize!; WRONG !!!
					maxt = 0.
					maxp = 0.
					it = 0
					for cion in [1, 2]:
						for ionnumber in range(len(modpeptide) - 3):
							for lion in ['a', 'b-h2o', 'b-nh3', 'b', 'c']:
								if (lion == "b") & (cion == 1):
									if maxt < (2 ** targets[it]) - 0.001:
										maxt = (2 ** targets[it]) - 0.001
									if maxp < (2 ** predictions[0][ionnumber]) - 0.001:
										maxp = (2 ** predictions[0][ionnumber]) - 0.001
								it += 1
					for cion in [1, 2]:
						for ionnumber in range(len(modpeptide) - 3):
							for lion in ['y-h2o', 'z', 'y', 'x']:
								if (lion == "y") & (cion == 1):
									if maxt < (2 ** targets[it]) - 0.001:
										maxt = (2 ** targets[it]) - 0.001
									if maxp < (2 ** predictions[1][ionnumber]) - 0.001:
										maxp = (2 ** predictions[1][ionnumber]) - 0.001
								it += 1
					#b
					it = 0
					for cion in [1, 2]:
						for ionnumber in range(len(modpeptide)-3):
							for lion in ['a', 'b-h2o', 'b-nh3', 'b', 'c']:
								if mzs[it] > 0:
									numall += 1
									explainedall += (2 ** targets[it]) - 0.001
								ft.write("%s;%f;%f;%s;%i;%i;1\n"%(title, mzs[it], (2 ** targets[it]) / maxt, lion, cion, ionnumber))
								if (lion == "b") & (cion == 1):
									ts.append(targets[it])
									ps.append(predictions[0][ionnumber])
									if mzs[it] > 0:
										numby += 1
										explainedby += (2 ** targets[it]) - 0.001
									ft.write("%s;%f;%f;%s;%i;%i;2\n"%(title, mzs[it], (2 ** (predictions[0][ionnumber])) / maxp, lion, cion, ionnumber))
								it += 1
					#y
					for cion in [1, 2]:
						for ionnumber in range(len(modpeptide) - 3):
							for lion in ['y-h2o', 'z', 'y', 'x']:
								if mzs[it] > 0:
									numall += 1
									explainedall += (2 ** targets[it]) - 0.001
								ft.write("%s;%f;%f;%s;%i;%i;1\n"%(title, mzs[it], (2 ** targets[it]) / maxt, lion, cion, ionnumber))
								if (lion == "y") & (cion == 1):
									ts.append(targets[it])
									ps.append(predictions[1][ionnumber])
									if mzs[it] > 0:
										numby += 1
										explainedby += (2 ** targets[it]) - 0.001
									ft.write("%s;%f;%f;%s;%i;%i;2\n"%(title, mzs[it], (2 ** (predictions[1][ionnumber])) / maxp, lion, cion, ionnumber))
								it += 1
					ft2.write("%s;%i;%i;%f;%f;%i;%i;%f;%f;%f;%f\n"%(
						title, len(modpeptide) - 2, len(msms), tic,
						pearsonr(ts, ps)[0], numby, numall, explainedby,
						explainedall, float(numby) / (2 * (len(peptide) - 3)),
						float(numall) / (18 * (len(peptide) - 3)))
					)
				else:
					# Predict the b- and y-ion intensities from the peptide
					pepid_buf.append(title)
					peplen_buf.append(len(peptide) - 2)
					charge_buf.append(charge)

					# get/append ion mzs, targets and predictions
					targets = ms2pip_pyx.get_targets(modpeptide, msms, peaks, float(fragerror), peaks_version)
					target_buf.append([np.array(t, dtype=np.float32) for t in targets])
					mzs = ms2pip_pyx.get_mzs(modpeptide, peaks_version)
					mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])
					predictions = ms2pip_pyx.get_predictions(peptide, modpeptide, charge, model_id, peaks_version, colen) #SD: added colen
					prediction_buf.append([np.array(p, dtype=np.float32) for p in predictions])

				pcount += 1
				if (pcount % 500) == 0:
					sys.stdout.write("(%i)%i "%(worker_num, pcount))
					sys.stdout.flush()

	f.close()
	if tableau:
		ft.close()
		ft2.close()

	if vector_file:
		# If num_cpu > number of spectra, dvectors can be empty
		if dvectors:
			# Concatenating dvectors into a 2D ndarray before making DataFrame saves lots of memory!
			if len(dvectors) > 1:
				dvectors = np.concatenate(dvectors)
			df = pd.DataFrame(dvectors, dtype=np.uint16, copy=False)
			df.columns = df.columns.astype(str)
		else:
			df = pd.DataFrame()
		return psmids, df, dtargets

	return mz_buf, prediction_buf, target_buf, peplen_buf, charge_buf, pepid_buf

def scan_spectrum_file(filename):
	"""
	go over mgf file and return list with all spectrum titles
	"""
	titles = []
	f = open(filename)
	while 1:
		rows = f.readlines(10000)
		if not rows:
			break
		for row in rows:
			if row[0] == "T":
				if row[:5] == "TITLE":
					titles.append(row.rstrip()[6:])#.replace(" ", "") # unnecessary? creates issues when PEPREC spec_id has spaces
	f.close()
	return titles


def prepare_titles(titles, num_cpu):
	"""
	Take a list and return a list containing num_cpu smaller lists with the
	spectrum titles/peptides that will be split across the workers
	"""
	# titles might be ordered from small to large peptides,
	# shuffling improves parallel speeds
	shuffle(titles)

	split_titles = [titles[i * len(titles) // num_cpu: (i + 1) * len(titles) // num_cpu] for i in range(num_cpu)]
	sys.stdout.write("{} spectra (~{:.0f} per cpu)\n".format(len(titles), np.mean([len(a) for a in split_titles])))

	return split_titles


def apply_mods(peptide, mods, PTMmap):
	"""
	Takes a peptide sequence and a set of modifications. Returns the modified
	version of the peptide sequence, c- and n-term modifications. This modified
	version are hard coded in ms2pipfeatures_c.c for now.
	"""
	modpeptide = np.array(peptide[:], dtype=np.uint16)

	if mods != "-":
		l = mods.split("|")
		for i in range(0, len(l), 2):
			tl = l[i + 1]
			if tl in PTMmap:
				modpeptide[int(l[i])] = PTMmap[tl]
			else:
				sys.stderr.write("Unknown modification: {}\n".format(tl))
				return "Unknown modification"

	return modpeptide


def load_configfile(filepath):
	params = {}
	params['ptm'] = []
	params['sptm'] = []
	params['gptm'] = []
	with open(filepath) as f:
		for line in f:
			line = line.strip()
			if not line or line[0] == '#':
				continue
			(par, val) = line.split('=')
			if par == "ptm":
				params["ptm"].append(val)
			elif par == "sptm":
				params["sptm"].append(val)
			elif par == "gptm":
				params["gptm"].append(val)
			else:
				params[par] = val
	return params


def generate_modifications_file(params, MASSES, A_MAP):
	PTMmap = {}

	ptmnum = 38  # Omega compatibility (mutations)
	spbuffer = []
	for v in params["sptm"]:
		l = v.split(',')
		tmpf = float(l[1])
		if l[2] == 'opt':
			if l[3] == "N-term":
				spbuffer.append([tmpf, -1, ptmnum])
				PTMmap[l[0]] = ptmnum
				ptmnum += 1
				continue
			if l[3] == "C-term":
				spbuffer.append([tmpf, -2, ptmnum])
				PTMmap[l[0]] = ptmnum
				ptmnum += 1
				continue
			if not l[3] in A_MAP:
				continue
			spbuffer.append([tmpf, A_MAP[l[3]], ptmnum])
			PTMmap[l[0]] = ptmnum
			ptmnum += 1
	pbuffer = []
	for v in params["ptm"]:
		l = v.split(',')
		tmpf = float(l[1])
		if l[2] == 'opt':
			if l[3] == "N-term":
				pbuffer.append([tmpf, -1, ptmnum])
				#PTMmap[l[0].lower()] = ptmnum
				PTMmap[l[0]] = ptmnum
				ptmnum += 1
				continue
			if l[3] == "C-term":
				pbuffer.append([tmpf, -2, ptmnum])
				#PTMmap[l[0].lower()] = ptmnum
				PTMmap[l[0]] = ptmnum
				ptmnum += 1
				continue
			if not l[3] in A_MAP:
				continue
			pbuffer.append([tmpf, A_MAP[l[3]], ptmnum])
			#PTMmap[l[0].lower()] = ptmnum
			#print("%i %s"%(ptmnum,l[0]))
			PTMmap[l[0]] = ptmnum
			ptmnum += 1

	f = tempfile.NamedTemporaryFile(delete=False, mode='wb')
	f.write(str.encode("{}\n".format(len(pbuffer))))
	for i, _ in enumerate(pbuffer):
		f.write(str.encode("{},1,{},{}\n".format(pbuffer[i][0], pbuffer[i][1], pbuffer[i][2])))
	f.close()

	f2 = tempfile.NamedTemporaryFile(delete=False, mode='wb')
	f2.write(str.encode("{}\n".format(len(spbuffer))))
	for i, _ in enumerate(spbuffer):
		f2.write(str.encode("{},1,{},{}\n".format(spbuffer[i][0], spbuffer[i][1], spbuffer[i][2])))
	f2.close()

	return f.name, f2.name, PTMmap


def peakcount(x):
	c = 0.
	for i in x:
		if i > -9.95:
			c += 1.
	return c / len(x)


def argument_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("pep_file", metavar="<PEPREC file>",
						help="list of peptides")
	parser.add_argument("-c", metavar="CONFIG_FILE", action="store", dest="config_file", default="config.txt",
						help="config file (by default config.txt)")
	parser.add_argument("-s", metavar="MGF_FILE", action="store", dest="spec_file",
						help=".mgf MS2 spectrum file (optional)")
	parser.add_argument("-w", metavar="FEATURE_VECTOR_OUTPUT", action="store", dest="vector_file",
						help="write feature vectors to FILE.{pkl,h5} (optional)")
	parser.add_argument('-x', action='store_true', default=False, dest='correlations',
						help='calculate correlations (if MGF is given)')
	parser.add_argument('-t', action='store_true', default=False, dest='tableau',
						help='create Tableau Reader file')
	parser.add_argument("-m", metavar="NUM_CPU", action="store", dest="num_cpu",
						default="23", help="number of cpu's to use")
	args = parser.parse_args()

	if not args.config_file:
		print("Please provide a configfile (-c)!")
		exit(1)

	return args.pep_file, args.spec_file, args.vector_file, args.config_file, int(args.num_cpu), args.correlations, args.tableau


def run(pep_file, spec_file=None, vector_file=None, config_file=None,
		num_cpu=23, params=None, output_filename=None, datasetname=None,
		return_results=False, limit=None, compute_correlations=False,
		tableau=False):
	# datasetname is needed for Omega compatibility. This can be set to None if a config_file is provided

	# If not specified, get parameters from config_file
	if params is None:
		if config_file is None:
			if datasetname is None:
				print("No config file specified")
				exit(1)
		else:
			params = load_configfile(config_file)

	if 'model' in params:
		model = params["model"]
	elif 'frag_method' in params:
		model = params['frag_method']
	else:
		print("Please specify model in config file or parameters.")
		exit(1)
	fragerror = params["frag_error"]

	# Validate requested output formats
	if "out" in params:
		out_formats = [o.lower().strip() for o in params["out"].split(',')]
		for o in out_formats:
			if o not in SUPPORTED_OUT_FORMATS:
				print("Unknown output format: '{}'".format(o))
				print("Should be one of the following formats: {}".format(SUPPORTED_OUT_FORMATS))
				exit(1)
	else:
		if not return_results:
			print("No output format specified; defaulting to csv")
			out_formats = ['csv']
		else:
			out_formats = []

	# Validate requested model
	if model in MODELS.keys():
		print("using {} models".format(model))
	else:
		print("Unknown fragmentation method: {}".format(model))
		print("Should be one of the following methods: {}".format(MODELS.keys()))
		exit(1)

	if output_filename is None and not return_results:
		output_filename = '{}_{}'.format('.'.join(pep_file.split('.')[:-1]), model)

	# Create amino acid MASSES file
	# to be compatible with Omega
	# that might have fixed modifications
	f = tempfile.NamedTemporaryFile(delete=False)
	for m in MASSES:
		f.write(str.encode("{}\n".format(m)))
	f.write(str.encode("0\n"))
	f.close()
	afile = f.name

	# PTMs are loaded the same as in Omega
	# This allows me to use the same C init() function in bot ms2ip and Omega
	(modfile, modfile2, PTMmap) = generate_modifications_file(params, MASSES, A_MAP)

	# read peptide information
	# the file contains the columns: spec_id, modifications, peptide and charge
	if type(pep_file) == str:
		with open(pep_file, 'rt') as f:
			line = f.readline()
			if line[:7] != 'spec_id':
				sys.stdout.write('PEPREC file should start with header column\n')
				exit(1)
			sep = line[7]
		data = pd.read_csv(pep_file,
						   sep=sep,
						   index_col=False,
						   dtype={"spec_id": str, "modifications": str},
						   nrows=limit)
	else:
		data = pep_file
	# for some reason the missing values are converted to float otherwise
	data = data.fillna("-")

	# Filter PEPREC for unsupported peptides
	num_pep = len(data)
	data = data[
		~(data['peptide'].str.contains('B|J|O|U|X|Z')) &
		~(data['peptide'].str.len() < 3) &
		~(data['peptide'].str.len() > 99)
	].copy()
	num_pep_filtered = num_pep - len(data)
	if num_pep_filtered > 0:
		sys.stdout.write("Removed {} unsupported peptide sequences (< 3, > 99 \
amino acids, or containing B, J, O, U, X or Z).\n".format(num_pep_filtered))

	if len(data) == 0:
		sys.stdout.write("No peptides for which to predict intensities. Please \
provide at least one valid peptide sequence.\n")
		exit(1)

	sys.stdout.write("starting workers...\n")
	myPool = multiprocessing.Pool(num_cpu)

	if spec_file:
		"""
		When an mgf file is provided, MS2PIP either saves the feature vectors to
		train models with or writes a file with the predicted spectra next to
		the empirical one.
		"""
		sys.stdout.write("scanning spectrum file... \n")
		titles = scan_spectrum_file(spec_file)
		split_titles = prepare_titles(titles, num_cpu)
		results = []

		for i in range(num_cpu):
			tmp = split_titles[i]
			"""
			process_spectra(
				i,
				spec_file,
				vector_file,
				data[data["spec_id"].isin(tmp)],
				afile, modfile, modfile2, PTMmap, model, fragerror, tableau)
			"""
			results.append(myPool.apply_async(process_spectra, args=(
				i,
				spec_file,
				vector_file,
				data[data["spec_id"].isin(tmp)],
				afile, modfile, modfile2, PTMmap, model, fragerror, tableau)))
			#"""
		myPool.close()
		myPool.join()

		sys.stdout.write("\nmerging results ")

		# Create vector file
		if vector_file:
			all_results = []
			for r in results:
				sys.stdout.write(".")
				psmids, df, dtargets = r.get()

				# dtargets is a dict, containing targets for every ion type (keys are int)
				for i, t in dtargets.items():
					df["targets_{}".format(MODELS[model]['ion_types'][i])] = np.concatenate(t, axis=None)
				df["psmid"] = psmids

				all_results.append(df)

			# Only concat DataFrames with content (we get empty ones if more cpu's than peptides)
			all_results = pd.concat([df for df in all_results if len(df) != 0])

			sys.stdout.write("\nwriting vector file {}... \n".format(vector_file))
			# write result. write format depends on extension:
			ext = vector_file.split(".")[-1]
			if ext == "pkl":
				all_results.to_pickle(vector_file + ".pkl")
			elif ext == "csv":
				all_results.to_csv(vector_file)
			else:
				# "table" is a tag used to read back the .h5
				all_results.to_hdf(vector_file, "table")

		# Predict and compare with MGF file
		else:
			mz_bufs = []
			prediction_bufs = []
			target_bufs = []
			peplen_bufs = []
			charge_bufs = []
			pepid_bufs = []
			for r in results:
				mz_buf, prediction_buf, target_buf, peplen_buf, charge_buf, pepid_buf = r.get()
				mz_bufs.extend(mz_buf)
				prediction_bufs.extend(prediction_buf)
				target_bufs.extend(target_buf)
				peplen_bufs.extend(peplen_buf)
				charge_bufs.extend(charge_buf)
				pepid_bufs.extend(pepid_buf)

			# Reconstruct DataFrame
			num_ion_types = len(mz_bufs[0])
			ions = []
			ionnumbers = []
			charges = []
			pepids = []
			for pi, pl in enumerate(peplen_bufs):
				[ions.extend([ion_type] * (pl - 1)) for ion_type in MODELS[model]['ion_types']]
				ionnumbers.extend([x + 1 for x in range(pl - 1)] * num_ion_types)
				charges.extend([charge_bufs[pi]] * (num_ion_types * (pl - 1)))
				pepids.extend([pepid_bufs[pi]] * (num_ion_types * (pl - 1)))
			all_preds = pd.DataFrame()
			all_preds["spec_id"] = pepids
			all_preds["charge"] = charges
			all_preds["ion"] = ions
			all_preds["ionnumber"] = ionnumbers
			all_preds["mz"] = np.concatenate(mz_bufs, axis=None)
			all_preds["target"] = np.concatenate(target_bufs, axis=None)
			all_preds["prediction"] = np.concatenate(prediction_bufs, axis=None)

			sys.stdout.write("\nwriting file {}_pred_and_emp.csv...\n".format(output_filename))
			all_preds.to_csv("{}_pred_and_emp.csv".format(output_filename), index=False)

			if compute_correlations:
				sys.stdout.write('computing correlations...\n')
				correlations = calc_correlations.calc_correlations(all_preds)
				correlations.to_csv("{}_correlations.csv".format(output_filename), index=True)
				sys.stdout.write("median correlations: \n")
				sys.stdout.write("{}\n".format(correlations.groupby('ion')['pearsonr'].median()))
			
			sys.stdout.write("done! \n")

	# Only get the predictions
	else:
		sys.stdout.write("scanning peptide file... ")

		titles = data.spec_id.tolist()
		split_titles = prepare_titles(titles, num_cpu)
		results = []

		for i in range(num_cpu):
			tmp = split_titles[i]
			"""
			process_peptides(
				i,
				data[data.spec_id.isin(tmp)],
				afile, modfile, modfile2, PTMmap, model)
			"""
			results.append(myPool.apply_async(process_peptides, args=(
				i,
				data[data.spec_id.isin(tmp)],
				afile, modfile, modfile2, PTMmap, model)))
			#"""
		myPool.close()
		myPool.join()

		sys.stdout.write("merging results...\n")

		mz_bufs = []
		prediction_bufs = []
		peplen_bufs = []
		charge_bufs = []
		pepid_bufs = []
		for r in results:
			mz_buf, prediction_buf, peplen_buf, charge_buf, pepid_buf = r.get()
			mz_bufs.extend(mz_buf)
			prediction_bufs.extend(prediction_buf)
			peplen_bufs.extend(peplen_buf)
			charge_bufs.extend(charge_buf)
			pepid_bufs.extend(pepid_buf)

		# Reconstruct DataFrame
		num_ion_types = len(MODELS[model]['ion_types'])

		ions = []
		ionnumbers = []
		charges = []
		pepids = []
		for pi, pl in enumerate(peplen_bufs):
			_ = [ions.extend([ion_type] * (pl - 1)) for ion_type in MODELS[model]['ion_types']]
			ionnumbers.extend([x + 1 for x in range(pl - 1)] * num_ion_types)
			charges.extend([charge_bufs[pi]] * (num_ion_types * (pl - 1)))
			pepids.extend([pepid_bufs[pi]] * (num_ion_types * (pl - 1)))
		all_preds = pd.DataFrame()
		all_preds["spec_id"] = pepids
		all_preds["charge"] = charges
		all_preds["ion"] = ions
		all_preds["ionnumber"] = ionnumbers
		all_preds["mz"] = np.concatenate(mz_bufs, axis=None)
		all_preds["prediction"] = np.concatenate(prediction_bufs, axis=None)


		if not return_results:
			if 'mgf' in out_formats:
				print("writing MGF file {}_predictions.mgf...".format(output_filename))
				spectrum_output.write_mgf(all_preds, peprec=data, output_filename=output_filename)

			if 'msp' in out_formats:
				print("writing MSP file {}_predictions.msp...".format(output_filename))
				spectrum_output.write_msp(all_preds, data, output_filename=output_filename)

			if 'bibliospec' in out_formats:
				print("writing SSL/MS2 files...")
				spectrum_output.write_bibliospec(all_preds, data, params, output_filename=output_filename)

			if 'spectronaut' in out_formats:
				print("writing Spectronaut CSV files...")
				spectrum_output.write_spectronaut(all_preds, data, params, output_filename=output_filename)

			if 'csv' in out_formats:
				print("writing CSV {}_predictions.csv...".format(output_filename))
				all_preds.to_csv("{}_predictions.csv".format(output_filename), index=False)
			
			sys.stdout.write("done!\n")
		else:
			return all_preds
