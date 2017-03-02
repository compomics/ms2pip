import sys
import numpy as np
import pandas as pd
import ms2pipfeatures_pyx
import pickle
import argparse
import multiprocessing
import xgboost as xgb
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('pep_file', metavar='<peptide file>',
					 help='list of peptides')
	parser.add_argument('-s', metavar='FILE',action="store", dest='spec_file',
					 help='.mgf MS2 spectrum file (optional)')
	parser.add_argument('-w', metavar='FILE',action="store", dest='vector_file',
					 help='write feature vectors to FILE.pkl (optional)')
	parser.add_argument('-c', metavar='INT',action="store", dest='num_cpu',default='23',
					 help="number of cpu's to use")

	args = parser.parse_args()

	num_cpu = int(args.num_cpu)

	# read peptide information
	# the file contains the following columns: spec_id, modifications, peptide and charge
	data = pd.read_csv(	args.pep_file,
						sep=' ',
						index_col=False,
						dtype={'spec_id':str,'modifications':str})
	data = data.fillna('-') # for some reason the missing values are converted to float otherwise

	#print data.peptide.value_counts()
	#ddd
	if args.spec_file:
		# This code runs if we give an mgf file as input. One of two things can happen after:
		# if args.vector_file, we save a file with the feature vectors and targets
		# else, we predict spectra and return a file with the target and predicted values for each ion

		# processing the mgf file:
		# this is parallelized at the spectrum TITLE level
		sys.stdout.write('scanning spectrum file...')
		titles = scan_spectrum_file(args.spec_file)
		num_spectra_per_cpu = int(len(titles)/(num_cpu))
		sys.stdout.write("%i spectra (%i per cpu)\n"%(len(titles),num_spectra_per_cpu))

		sys.stdout.write('starting workers...\n')

		myPool = multiprocessing.Pool(num_cpu)

		results = []
		i = 0
		for i in range(num_cpu-1):
			tmp = titles[i*num_spectra_per_cpu:(i+1)*num_spectra_per_cpu]
			# this commented part of code can be used for debugging by avoiding parallel processing
			"""

			process_file(
										args,
										data[data.spec_id.isin(tmp)]
										)
			"""
			results.append(myPool.apply_async(process_file,args=(
										args,
										data[data.spec_id.isin(tmp)]
										)))
			i+=1
			tmp = titles[i*num_spectra_per_cpu:]
			results.append(myPool.apply_async(process_file,args=(
									args,
									data[data.spec_id.isin(tmp)]
									)))

		myPool.close()
		myPool.join()

		# workers done...merging results
		sys.stdout.write('\nmerging results and writing files...\n')

		if args.vector_file:
			# i.e. if we want to save the features + targets:
			# read feature vectors from workers and concatenate
			all_vectors = []
			for r in results:
				all_vectors.extend(r.get())
			all_vectors = pd.concat(all_vectors)
  			# write result. write format depends on extension:
  			ext = args.vector_file.split('.')[-1]
  			if ext == 'pkl':
				# print all_vectors.head()
				all_vectors.to_pickle(args.vector_file+'.pkl')
  			elif ext == 'h5':
				all_vectors.to_hdf(args.vector_file, 'table')
    			# 'table' is a tag used to read back the .h5
  			else: # if none of the two, default to .h5
				all_vectors.to_hdf(args.vector_file, 'table')

	else:
		# For when we only give the PEPREC file and want the predictions

		result = process_peptides(None, data)
		sys.stdout.write('\nmerging results and writing CSV...\n')
		result.to_csv(args.pep_file +'_predictions.csv', index=False)
		sys.stdout.write('\nmerging results and writing MGF...\n')
		mgf_output = open(args.pep_file +'_predictions.mgf', 'w+')
		for sp in result.spec_id.unique():
			tmp = result[result.spec_id == sp]
			tmp = tmp.sort_values('mz')
			mgf_output.write('BEGIN IONS\n')
			mgf_output.write('TITLE=' + str(sp) + '\n')
			mgf_output.write('CHARGE=' + str(tmp.charge[0]) +'\n')
			for i in range(len(tmp)):
				mgf_output.write(str(tmp['mz'][i]) + ' ' + str(tmp['prediction'][i]) + '\n')
			mgf_output.write('END IONS\n')
		mgf_output.close()


#peak intensity prediction without spectrum file (under construction)
def process_peptides(args,data):
	"""
	Take the PEPREC file (loaded in the variable data) and predict spectra.
	return an .mgf file.
	"""
	sys.stdout.write('predicting spectra... \n')
	# NOTE to write out the results as an .mgf file I need an m/z for each b and
	# y ion as well as the predicted intensities. I also need the total ion m/z
	# and a TITLE

	# a_map converts the peptide amino acids to integers, note how 'L' is removed
	aminos = ['A','C','D','E','F','G','H','I','K','M','N','P','Q','R','S','T','V','W','Y']
	masses = [71.037114,160.030645,115.026943,129.042593,147.068414,57.021464,137.058912,113.084064,128.094963,131.040485,114.042927,97.052764,128.058578,156.101111,87.032028,101.047679,99.068414,186.079313,163.063329,147.0354]
	a_map = {}
	a_mass = {}
	for i,a in enumerate(aminos):
		a_map[a] = i
		a_mass[i] = masses[i]

	a_mass[19] = masses[19]
	# transform pandas datastructure into dictionary for easy access
	specdict = data[['spec_id','peptide','modifications','charge']].set_index('spec_id').to_dict()
	peptides = specdict['peptide']
	modifications = specdict['modifications']
	charges = specdict['charge']

	final_result = pd.DataFrame(columns=['peplen','charge','ion','mz', 'ionnumber', 'prediction', 'spec_id'])
	i = 0
	for (pepid,modsid) in zip(peptides,modifications):
		i += 1
		if i%100000 == 0: sys.stdout.write('.')
		ch = charges[pepid]

		peptide = peptides[pepid]
		peptide = peptide.replace('L','I')

		# convert peptide string to integer list to speed up C code
		peptide = np.array([a_map[x] for x in peptide],dtype=np.uint16)
		# modpeptide is the same as peptide but with modified amino acids
		# converted to other integers (beware: these are hard coded in ms2pipfeatures_c.c for now)
		mods = modifications[modsid]
		modpeptide = np.array(peptide[:],dtype=np.uint16)
		peplen = len(peptide)
		if mods != '-':
			l = mods.split('|')
			for i in range(0,len(l),2):
				if l[i+1] == "Oxidation":
					modpeptide[int(l[i])] = 19

		b_mz = [None] * (len(modpeptide)-1)
		y_mz = [None] * (len(modpeptide)-1)
		b_mz[0] = a_mass[modpeptide[0]] + 1.007236
		y_mz[0] = a_mass[modpeptide[len(modpeptide)-1]] + 18.0105647 + 1.007236
		for i in range(1, len(modpeptide)-1):
			b_mz[i] = b_mz[i-1] + a_mass[modpeptide[i]]
			y_mz[i] = y_mz[i-1] + a_mass[modpeptide[-(i+1)]]

		# get ion intensities
		(resultB,resultY) = ms2pipfeatures_pyx.get_predictions(peptide, modpeptide, ch)

		# return results as a DataFrame
		tmp = pd.DataFrame()
		tmp['peplen'] = [len(peptide)]*(2*len(resultB))
		tmp['charge'] = [ch]*(2*len(resultB))
		tmp['ion'] = ['b']*len(resultB)+['y']*len(resultY)
		tmp['mz'] = b_mz + y_mz
		tmp['ionnumber'] = range(1,len(resultB)+1)+range(len(resultY),0,-1)
		# tmp['target'] = b + y
		tmp['prediction'] = resultB + resultY
		tmp['spec_id'] = [pepid]*len(tmp)

		final_result = final_result.append(tmp)

	return final_result

#peak intensity prediction with spectrum file (for evaluation)
def process_file(args,data):

	# a_map converts the peptide amio acids to integers, note how 'L' is removed
	aminos = ['A','C','D','E','F','G','H','I','K','M','N','P','Q','R','S','T','V','W','Y']
	a_map = {}
	for i,a in enumerate(aminos):
		a_map[a] = i

	# transform pandas datastructure into dictionary for easy access
	specdict = data[['spec_id','peptide','modifications']].set_index('spec_id').to_dict()
	peptides = specdict['peptide']
	modifications = specdict['modifications']

	# cols contains the names of the computed features
	cols_n = get_feature_names()

	title = ""
	parent_mz = 0.
	charge = 0
	msms = []
	peaks = []
	f = open(args.spec_file)
	skip = False
	vectors = []
	result = []
	pcount = 0
	while (1):
		rows = f.readlines(3000000)
		sys.stdout.write('.')
		if not rows: break
		for row in rows:
			row = row.rstrip()
			if row == "": continue
			if skip:
				if row[0] == "B":
					if row[:10] == "BEGIN IONS":
						skip = False
				else:
					continue
			if row == "": continue
			if row[0] == "T":
				if row[:5] == "TITLE":
					title = row[6:].replace(' ','')
					if not title in peptides:
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
					charge = int(row[7:9].replace("+",""))
			elif row[0] == "P":
				if row[:7] == "PEPMASS":
					parent_mz = float(row[8:].split()[0])
			elif row[:8] == "END IONS":
				#process
				if not title in peptides: continue
				#if title != "human684921": continue


				parent_mz = (float(parent_mz) * (charge)) - ((charge)*1.007825035) #or 0.0073??

				peptide = peptides[title]
				#if not peptide == "FVFCAEAIYK":continue
				#if not charge == 2: continue
				peptide = peptide.replace('L','I')
				mods = modifications[title]

				#processing unmodified identification
				#if mods != '-': continue
				#processing charge 2 only !!!!!!!!!!!!!!!!!!!!!!!!
				#if charge != 2: continue

				# convert peptide string to integer list to speed up C code
				peptide = np.array([a_map[x] for x in peptide],dtype=np.uint16)

				# modpeptide is the same as peptide but with modified amino acids
				# converted to other integers (beware: these are hard coded in ms2pipfeatures_c.c for now)
				modpeptide = np.array(peptide[:],dtype=np.uint16)
				peplen = len(peptide)
				k = False
				if mods != '-':
					l = mods.split('|')
					for i in range(0,len(l),2):
						if l[i+1] == "Oxidation":
							modpeptide[int(l[i])-1] = 19
				if k:
					continue

				# normalize and convert MS2 peaks
				msms = np.array(msms,dtype=np.float32)
				#peaks = np.array(peaks,dtype=np.float32)
				peaks = peaks / np.sum(peaks)
				peaks = np.array(np.log2(peaks+0.001))
				peaks = peaks.astype(np.float32)

				# find the b- and y-ion peak intensities in the MS2 spectrum
				(b,y) = ms2pipfeatures_pyx.get_targets(modpeptide,msms,peaks)

				#for debugging!!!!
				#tmp = pd.DataFrame(ms2pipfeatures_pyx.get_vector(peptide,modpeptide,charge),columns=cols,dtype=np.uint32)
				#print bst.predict(xgb.DMatrix(tmp))

				if args.vector_file:
					tmp = pd.DataFrame(ms2pipfeatures_pyx.get_vector(peptide,modpeptide,charge),columns=cols_n,dtype=np.uint16)
					tmp["targetsB"] = b
					tmp["targetsY"] = y
					tmp["psmid"] = [title]*len(tmp)
					vectors.append(tmp)
				else:
					# predict the b- and y-ion intensities from the peptide
					(resultB,resultY) = ms2pipfeatures_pyx.get_predictions(peptide,modpeptide,charge)
					for ii in range(len(resultB)):
						resultB[ii] = resultB[ii]+0.5 #This still needs to be checked!!!!!!!
					for ii in range(len(resultY)):
						resultY[ii] = resultY[ii]+0.5
					resultY = resultY[::-1]

					tmp = pd.DataFrame()
					tmp['peplen'] = [peplen]*(2*len(b))
					tmp['charge'] = [charge]*(2*len(b))
					tmp['ion'] = ['b']*len(b)+['y']*len(y)
					tmp['ionnumber'] = range(len(b))+range(len(y))
					tmp['target'] = b + y
					tmp['prediction'] = resultB + resultY
					tmp['spec_id'] = [title]*len(tmp)
					pcount += 1
					result.append(tmp)

	if args.vector_file:
		return vectors
	else:
		return result

#feature names
def get_feature_names():
	aminos = ['A','C','D','E','F','G','H','I','K','M','N','P','Q','R','S','T','V','W','Y']

	names = []
	for a in aminos:
		names.append("Ib_"+a)
	for a in aminos:
		names.append("Iy_"+a)
	names += ['pmz','peplen','ionnumber','ionnumber_rel']
	for c in ['mz','bas','heli','hydro','pI']:
		names.append('mean_'+c)

	for c in ['bas','heli','hydro','pI']:
		names.append('max_'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('min_'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('max'+c+'_b')
	for c in ['bas','heli','hydro','pI']:
		names.append('min'+c+'_b')
	for c in ['bas','heli','hydro','pI']:
		names.append('max'+c+'_y')
	for c in ['bas','heli','hydro','pI']:
		names.append('min'+c+'_y')

	for c in ['mz','bas','heli','hydro','pI']:
		names.append("%s_ion"%c)
		names.append("%s_ion_other"%c)
		names.append("mean_%s_ion"%c)
		names.append("mean_%s_ion_other"%c)

	for c in ['bas','heli','hydro','pI']:
		names.append('plus_cleave'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('times_cleave'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('minus1_cleave'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('minus2_cleave'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('bsum'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('ysum'+c)

	for pos in ['0','1','-2','-1']:
		for c in ['mz','bas','heli','hydro','pI','P','D','E','K','R']:
			names.append("loc_"+pos+"_"+c)

	for pos in ['i','i+1']:
		for c in ['P','D','E','K','R']:
			names.append("loc_"+pos+"_"+c)

	for c in ['bas','heli','hydro','pI','mz']:
		for pos in ['i','i-1','i+1','i+2']:
			names.append("loc_"+pos+"_"+c)

	names.append("charge")

	return names

#feature names for the fixed peptide length feature vectors
def get_feature_names_chem(peplen):
	aminos = ['A','C','D','E','F','G','H','I','K','M','N','P','Q','R','S','T','V','W','Y']

	names = []
	names += ['pmz','peplen','ionnumber','ionnumber_rel']
	for c in ['mz','bas','heli','hydro','pI']:
		names.append('mean_'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('max_'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('min_'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('max'+c+'_b')
	for c in ['bas','heli','hydro','pI']:
		names.append('min'+c+'_b')
	for c in ['bas','heli','hydro','pI']:
		names.append('max'+c+'_y')
	for c in ['bas','heli','hydro','pI']:
		names.append('min'+c+'_y')

	for c in ['mz','bas','heli','hydro','pI']:
		names.append("%s_ion"%c)
		names.append("%s_ion_other"%c)
		names.append("mean_%s_ion"%c)
		names.append("mean_%s_ion_other"%c)

	for c in ['bas','heli','hydro','pI']:
		names.append('plus_cleave'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('times_cleave'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('minus1_cleave'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('minus2_cleave'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('bsum'+c)
	for c in ['bas','heli','hydro','pI']:
		names.append('ysum'+c)

	for i in range(peplen):
		for c in ['mz','bas','heli','hydro','pI']:
			names.append("fix_"+c+"_"+str(i))

	names.append("charge")

	return names


def scan_spectrum_file(filename):
	titles = []
	f = open(filename)
	while (1):
		rows = f.readlines(1000000)
		if not rows: break
		for row in rows:
			if row[0] == "T":
				if row[:5] == "TITLE":
					titles.append(row.rstrip()[6:].replace(" ",""))
	f.close()
	return titles

def print_logo():
	logo = """
 _____ _____ ___ _____ _____ _____
|     |   __|_  |  _  |     |  _  |
| | | |__   |  _|   __|-   -|   __|
|_|_|_|_____|___|__|  |_____|__|

           """
	print logo
	print "by sven.degroeve@ugent.be\n"

if __name__ == "__main__":
	print_logo()
	main()
