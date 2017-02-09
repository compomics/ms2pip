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
			
	# processing is parallelized at the spectrum TITLE level
	sys.stderr.write('scanning spectrum file...')
	titles = scan_spectrum_file(args.spec_file)
	num_spectra_per_cpu = int(len(titles)/(num_cpu))
	sys.stderr.write("%i spectra (%i per cpu)\n"%(len(titles),num_spectra_per_cpu))

	sys.stderr.write('starting workers...\n')
	
	#myPool = multiprocessing.Pool(num_cpu)
	
	results = []
	i = 0
	for i in range(num_cpu-1):
		tmp = titles[i*num_spectra_per_cpu:(i+1)*num_spectra_per_cpu]
		# this commented part of code can be used for debugging by avoiding parallel processing
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
		"""

	# workers done...merging results
	sys.stderr.write('\nmerging results and writing files...\n')
	if args.spec_file:
		if args.vector_file:			
			# read feature vectors from workers and concatenate
			all_vectors = []
			for r in results:
				all_vectors.extend(r.get())			
			all_vectors = pd.concat(all_vectors)
			# write result
			all_vectors.to_pickle(args.vector_file +'_vectors.pkl')
		else:
			all_result = []
			for r in results:
				all_result.extend(r.get())
			all_result = pd.concat(all_result)
			#all_result.to_pickle("all_result.pkl")
			all_result.to_csv("all_result.csv",index=False)

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
	# this should be replaced by the actual feature names
	#cols = ["F"+str(a) for a in range(63)]
	#cols.append("charge")
	
	cols = get_feature_names()

	bst = xgb.Booster({'nthread':23}) #init model
	bst.load_model('vectors_vectors.pkl.xgboost') # load data
	xgb.plot_tree(bst)
	plt.show()
	
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
		sys.stderr.write('.')
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
				
				peptide = peptides[title]
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
							modpeptide[int(l[i])] = 19
				if k: 
					continue

				# normalize and convert MS2 peaks
				msms = np.array(msms,dtype=np.float32)
				peaks = peaks / np.sum(peaks)
				peaks = np.array(np.log2(peaks+0.001))
				peaks = peaks.astype(np.float32)

				# find the b- and y-ion peak intensities in the MS2 spectrum
				(b,y) = ms2pipfeatures_pyx.get_targets(modpeptide,msms,peaks)
				
				#tmp = pd.DataFrame(ms2pipfeatures_pyx.get_vector(peptide,modpeptide,charge),columns=cols,dtype=np.uint32)
				#print bst.predict(xgb.DMatrix(tmp))

				if args.vector_file:
					tmp = pd.DataFrame(ms2pipfeatures_pyx.get_vector(peptide,modpeptide,charge),columns=cols,dtype=np.uint16)
					tmp["targetsB"] = b
					tmp["targetsY"] = y
					tmp["psmid"] = [title]*len(tmp)
					vectors.append(tmp)
				else:				
					# predict the b- and y-ion intensities from the peptide
					(resultB,resultY) = ms2pipfeatures_pyx.get_predictions(peptide,modpeptide,msms,peaks,charge)
					#v = ms2pipfeatures_pyx.get_vector(peptide,modpeptide,charge)
					#print v
					xv = xgb.DMatrix(v)
					#print
					#print resultB
					#print bst.predict(xv)
					#ddddd
					resultY = resultY[::-1]
					for ii in range(len(resultB)):
						resultB[ii] = resultB[ii]+0.5 #This still needs to be checked!!!!!!!
					for ii in range(len(resultY)):
						resultY[ii] = resultY[ii]+0.5
	
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

def get_feature_names():
	names = ['pmz','peplen','ionnumber','ionnumber_rel']
	for c in ['mz','bas','hydro','heli','pI']:
		names.append('mean_'+c)
	for c in ['mz','bas','hydro','heli','pI']:
		names.append("%s_ion"%c)
		names.append("%s_ion_other"%c)
		names.append("mean_%s_ion"%c)
		names.append("mean_%s_ion_other"%c)

	for pos in ['0','1','-2','-1']:
		for c in ['mz','bas','hydro','heli','pI','P','D','E','K','R']:
			names.append("loc_"+pos+"_"+c)

	for pos in ['i','i+1']:
		for c in ['P','D','E','K','R']:
			names.append("loc_"+pos+"_"+c)

	for c in ['bas','hydro','heli','pI','mz']:
		for pos in ['i','i-1','i+1','i+2']:
			names.append("loc_"+pos+"_"+c)
				
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
					titles.append(row.rstrip()[6:])
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

