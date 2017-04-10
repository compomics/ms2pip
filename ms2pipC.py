import sys
import numpy as np
import pandas as pd
import pickle
import argparse
import multiprocessing
from random import shuffle
import tempfile
#import xgboost as xgb

#some globals

# a_map converts the peptide amio acids to integers, note how 'L' is removed
aminos = ['A','C','D','E','F','G','H','I','K','M','N','P','Q','R','S','T','V','W','Y']
masses = [71.037114,103.00919,115.026943,129.042593,147.068414,57.021464,137.058912,
		113.084064,128.094963,131.040485,114.042927,97.052764,128.058578,156.101111,
		87.032028,101.047679,99.068414,186.079313,163.063329,147.0354]
a_map = {}
for i,a in enumerate(aminos):
	a_map[a] = i
		
def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('pep_file', metavar='<peptide file>',
					 help='list of peptides')
	parser.add_argument('-c', metavar='FILE',action="store", dest='c',
					 help='config file')
	parser.add_argument('-s', metavar='FILE',action="store", dest='spec_file',
					 help='.mgf MS2 spectrum file (optional)')
	parser.add_argument('-w', metavar='FILE',action="store", dest='vector_file',
					 help='write feature vectors to FILE.pkl (optional)')
 	parser.add_argument('-i', action="store_true", default=False, help='iTRAQ models')
 	parser.add_argument('-p', action="store_true", default = False, help='phospho models')
	parser.add_argument('-m', metavar='INT',action="store", dest='num_cpu',default='23',
					 help="number of cpu's to use")

	args = parser.parse_args()

	if not args.c:
		print "Please provide a configfile (-c)!"
		exit(1)

	num_cpu = int(args.num_cpu)
			
	PTMmap = {}
	Ntermmap = {}
	Ctermmap = {}
	fragmethod = "none" # CID or HCD
	fragerror = 0
	# reading the configfile (-c) and configure the ms2pipfeatures_pyx module's datastructures
	fa = tempfile.NamedTemporaryFile(delete=False)
	numptms = 0
	with open(args.c) as f:
		for row in f:
			if row.startswith("ptm="): numptms+=1
			if row.startswith("sptm="): numptms+=1
	fa.write("%i\n"%numptms)
	pos = 38 #modified amino acids have numbers starting at 38 (mutations -> omega)
	with open(args.c) as f:
		for row in f:
			if row.startswith("sptm="):
				l=row.rstrip().split('=')[1].split(',')
				fa.write("%f\n"%(float(l[1])+masses[a_map[l[3]]]))
				PTMmap[l[0]] = pos
				pos+=1
	with open(args.c) as f:
		for row in f:
			if row.startswith("ptm="):
				l=row.rstrip().split('=')[1].split(',')
				fa.write("%f\n"%(float(l[1])+masses[a_map[l[3]]]))
				PTMmap[l[0]] = pos
				pos+=1
			if row.startswith("nterm="):
				l=row.rstrip().split('=')[1].split(',')
				Ntermmap[l[0]] = float(l[1])
			if row.startswith("cterm="):
				l=row.rstrip().split('=')[1].split(',')
				Ctermmap[l[0]] = float(l[1])
			if row.startswith("frag_method="):
				fragmethod=row.rstrip().split('=')[1]
			if row.startswith("frag_error="):
				fragerror=float(row.rstrip().split('=')[1])

	fa.close()

	if fragmethod == "CID":
		import ms2pipfeatures_pyx_CID as ms2pipfeatures_pyx
		print "using CID models..."
	elif fragmethod == "HCD":
		if args.i:
			if args.p:
				import ms2pipfeatures_pyx_HCDiTRAQ4phospho as ms2pipfeatures_pyx
				print "using HCD iTRAQ phospho models..."
			else:
				import ms2pipfeatures_pyx_HCDiTRAQ4 as ms2pipfeatures_pyx
				print "using HCD iTRAQ pmodels..."
		else:
			import ms2pipfeatures_pyx_HCD as ms2pipfeatures_pyx
			print "using HCD..."
	else:
		print "Unknown fragmentation method in configfile: %s"%fragmethod
		exit(1)
			
	ms2pipfeatures_pyx.ms2pip_init(fa.name)

	# read peptide information
	# the file contains the following columns: spec_id, modifications, peptide and charge
	data = pd.read_csv(	args.pep_file,
						sep=' ',
						index_col=False,
						dtype={'spec_id':str,'modifications':str})
	data = data.fillna('-') # for some reason the missing values are converted to float otherwise

	if args.spec_file:
		# Process the mgf file. In process_spectra, there is a check for
		# args.vector_file that determines what is returned (feature vectors or
		# evaluation of predictions)

		# processing the mgf file:
		# this is parallelized at the spectrum TITLE level
		sys.stdout.write('scanning spectrum file... ')
		titles = scan_spectrum_file(args.spec_file)
		#titles might be ordered from small to large peptides,
		#shuffling improves parallel speeds
		shuffle(titles)
		num_spectra_per_cpu = int(len(titles)/(num_cpu))
		sys.stdout.write("%i spectra (%i per cpu)\n"%(len(titles),num_spectra_per_cpu))

		sys.stdout.write('starting workers...\n')

		myPool = multiprocessing.Pool(num_cpu)

		results = []
		i = 0
		for i in range(num_cpu-1):
			#select titles for this worker
			tmp = titles[i*num_spectra_per_cpu:(i+1)*num_spectra_per_cpu]
			# this commented part of code can be used for debugging by avoiding parallel processing
			#process_spectra(i,args, data[data.spec_id.isin(tmp)],PTMmap,Ntermmap,Ctermmap,fragmethod,fragerror)
			#send worker to myPool
			
			results.append(myPool.apply_async(process_spectra,args=(
										i,
										args,
										data[data.spec_id.isin(tmp)],
										PTMmap,Ntermmap,Ctermmap,fragmethod,fragerror
										)))
		i+=1
		#some titles might be left
		tmp = titles[i*num_spectra_per_cpu:]
		results.append(myPool.apply_async(process_spectra,args=(
								i,
								args,
								data[data.spec_id.isin(tmp)],
								PTMmap,Ntermmap,Ctermmap,fragmethod,fragerror
								)))

		myPool.close()
		myPool.join()

		# workers done...merging results

		if args.vector_file:
			sys.stdout.write('\nmerging results...\n')
			# i.e. if we want to save the features + targets:
			# read feature vectors from workers and concatenate
			all_vectors = []
			for r in results:
				all_vectors.append(r.get())
			all_vectors = pd.concat(all_vectors)

			sys.stdout.write('writing file... \n')
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
			sys.stdout.write('\nmerging results...\n')
			all_spectra = []
			for r in results:
				all_spectra.append(r.get())
			all_spectra = pd.concat(all_spectra)

			sys.stdout.write('writing file...\n')
			all_spectra.to_csv(args.pep_file + '_pred_and_emp.csv', index=False)

			#sys.stdout.write('computing correlations...\n')
			#correlations = all_spectra.groupby('spec_id')[['target', 'prediction']].corr().ix[0::2,'prediction']			
			#corr_boxplot = correlations.plot('box')
			#corr_boxplot = corr_boxplot.get_figure()
			#corr_boxplot.suptitle('Pearson corr for ' + args.spec_file + ' and predictions')
			#corr_boxplot.savefig(args.pep_file + '_correlations.png')

		sys.stdout.write('done! \n')

	else:
		# Get only predictions from a pep_file
		sys.stdout.write('scanning peptide file... ')

		#titles might be ordered from small to large peptides,
		#shuffling improves parallel speeds
		titles = data.spec_id.tolist()
		shuffle(titles)
		num_pep_per_cpu = int(len(titles)/(num_cpu))
		sys.stdout.write("%i peptides (%i per cpu)\n"%(len(titles),num_pep_per_cpu))

		sys.stdout.write('starting workers...\n')
		myPool = multiprocessing.Pool(num_cpu)

		sys.stdout.write('predicting spectra... \n')
		results = []
		i = 0
		for i in range(num_cpu-1):
			#select titles for this worker			
			tmp = titles[i*num_pep_per_cpu:(i+1)*num_pep_per_cpu]
			"""
			process_peptides(i,args,data[data.spec_id.isin(tmp)],PTMmap,Ntermmap,Ctermmap,fragmethod)
			"""
			results.append(myPool.apply_async(process_peptides,args=(
										i,
										args,
										data[data.spec_id.isin(tmp)],
										PTMmap,Ntermmap,Ctermmap,fragmethod
										)))
		#some titles might be left
		i+=1
		tmp = titles[i*num_pep_per_cpu:]
		results.append(myPool.apply_async(process_peptides,args=(
								i,
								args,
								data[data.spec_id.isin(tmp)],
								PTMmap,Ntermmap,Ctermmap,fragmethod
								)))

		myPool.close()
		myPool.join()

		sys.stdout.write('\nmerging results...\n')

		all_preds = pd.DataFrame()
		for r in results:
			all_preds = all_preds.append(r.get())

		# print all_preds
		sys.stdout.write('writing files...\n')
		all_preds.to_csv(args.pep_file +'_predictions.csv', index=False)
		mgf = False # prevent writing big mgf files
		if mgf:
			sys.stdout.write('\nwriting mgf file...\n')
			mgf_output = open(args.pep_file +'_predictions.mgf', 'w+')
			for sp in all_preds.spec_id.unique():
				tmp = all_preds[all_preds.spec_id == sp]
				tmp = tmp.sort_values('mz')
				mgf_output.write('BEGIN IONS\n')
				mgf_output.write('TITLE=' + str(sp) + '\n')
				mgf_output.write('CHARGE=' + str(tmp.charge[0]) +'\n')
				for i in range(len(tmp)):
					mgf_output.write(str(tmp['mz'][i]) + ' ' + str(tmp['prediction'][i]) + '\n')
				mgf_output.write('END IONS\n')
			mgf_output.close()
		sys.stdout.write('done!\n')


#peak intensity prediction without spectrum file (under construction)
def process_peptides(worker_num,args,data,PTMmap,Ntermmap,Ctermmap,fragmethod):
	"""
	Read the PEPREC file and predict spectra.
	"""

	if fragmethod == "CID":
		import ms2pipfeatures_pyx_CID as ms2pipfeatures_pyx
	elif fragmethod == "HCD":
		if args.i:
			if args.p:
				import ms2pipfeatures_pyx_HCDiTRAQ4phospho as ms2pipfeatures_pyx
			else:
				import ms2pipfeatures_pyx_HCDiTRAQ4 as ms2pipfeatures_pyx
		else:
			import ms2pipfeatures_pyx_HCD as ms2pipfeatures_pyx
	else:
		print "Unknown fragmentation method in configfile: %s"%fragmethod
		exit(1)

	# transform pandas datastructure into dictionary for easy access
	specdict = data[['spec_id','peptide','modifications','charge']].set_index('spec_id').to_dict()
	peptides = specdict['peptide']
	modifications = specdict['modifications']
	charges = specdict['charge']

	final_result = pd.DataFrame(columns=['peplen','charge','ion','mz', 'ionnumber', 'prediction', 'spec_id'])
	pcount = 0
	total = len(peptides)

	for pepid in peptides:

		ch = charges[pepid]

		peptide = peptides[pepid]
		peptide = peptide.replace('L','I')

		# convert peptide string to integer list to speed up C code
		peptide = np.array([a_map[x] for x in peptide],dtype=np.uint16)
		# modpeptide is the same as peptide but with modified amino acids
		# converted to other integers (beware: these are hard coded in ms2pipfeatures_c.c for now)
		mods = modifications[pepid]
		modpeptide = np.array(peptide[:],dtype=np.uint16)
		peplen = len(peptide)
		nptm = 0
		cptm = 0
		if mods != '-':
			l = mods.split('|')
			for i in range(0,len(l),2):
				tl = l[i+1]
				if int(l[i]) == 0:
					if tl in Ntermmap:
						nptm += Ntermmap[tl]
					else:
						nptm += Ntermmap[tl[:-1]]
				elif int(l[i]) == -1:
					if tl in Ctermmap:
						cptm += Ctermmap[tl]
					else:
						cptm += Ctermmap[tl[:-1]]
				else:
					if tl in PTMmap:
						modpeptide[int(l[i])-1] = PTMmap[tl]
					else:
						modpeptide[int(l[i])-1] = PTMmap[tl[:-1]]

		(b_mz,y_mz) = ms2pipfeatures_pyx.get_mzs(modpeptide,nptm,cptm)
		print b_mz

		# get ion intensities
		(resultB,resultY) = ms2pipfeatures_pyx.get_predictions(peptide, modpeptide, ch)
		for ii in range(len(resultB)):
			resultB[ii] = resultB[ii]+0.5 #This still needs to be checked!!!!!!!
		for ii in range(len(resultY)):
			resultY[ii] = resultY[ii]+0.5

		# return results as a DataFrame
		tmp = pd.DataFrame()
		tmp['peplen'] = [peplen]*(2*len(resultB))
		tmp['charge'] = [ch]*(2*len(resultB))
		tmp['ion'] = ['b']*len(resultB)+['y']*len(resultY)
		tmp['mz'] = b_mz + y_mz
		tmp['ionnumber'] = range(1,len(resultB)+1)+range(len(resultY),0,-1)
		tmp['prediction'] = resultB + resultY
		tmp['spec_id'] = [pepid]*len(tmp)
		final_result = final_result.append(tmp)
		pcount += 1
		if (pcount % 500) == 0:
			sys.stderr.write('w' + str(worker_num) + '(' + str(pcount) + ') ')
	return final_result

# peak intensity prediction with spectrum file (for evaluation) OR feature extraction
def process_spectra(worker_num,args,data, PTMmap,Ntermmap,Ctermmap,fragmethod,fragerror):

	if fragmethod == "CID":
		import ms2pipfeatures_pyx_CID as ms2pipfeatures_pyx
	elif fragmethod == "HCD":
		if args.i:
			if args.p:
				import ms2pipfeatures_pyx_HCDiTRAQ4phospho as ms2pipfeatures_pyx
			else:
				import ms2pipfeatures_pyx_HCDiTRAQ4 as ms2pipfeatures_pyx
		else:
			import ms2pipfeatures_pyx_HCD as ms2pipfeatures_pyx
	else:
		print "Unknown fragmentation method in configfile: %s"%fragmethod
		exit(1)

	# transform pandas datastructure into dictionary for easy access
	specdict = data[['spec_id','peptide','modifications']].set_index('spec_id').to_dict()
	peptides = specdict['peptide']
	modifications = specdict['modifications']

	total = len(peptides)
	
	# cols contains the names of the computed features
	cols_n = get_feature_names()
	
	dataresult = pd.DataFrame(columns=['spec_id','peplen','charge','ion','ionnumber','target','prediction'])
	dataresult['peplen'] = dataresult['peplen'].astype(np.uint8)
	dataresult['charge'] = dataresult['charge'].astype(np.uint8)
	dataresult['ion'] = dataresult['ion'].astype(np.uint8)
	dataresult['ionnumber'] = dataresult['ionnumber'].astype(np.uint8)
	dataresult['target'] = dataresult['target'].astype(np.float32)					
	dataresult['prediction'] = dataresult['prediction'].astype(np.float32)					
		
	title = ""
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
		# sys.stdout.write('.')
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
			elif row[:8] == "END IONS":
				#process
				if not title in peptides: continue

				#with counter.get_lock():
				#	counter.value += 1
				#sys.stderr.write("%i ",counter.value)

				peptide = peptides[title]
				peptide = peptide.replace('L','I')
				mods = modifications[title]

				# convert peptide string to integer list to speed up C code
				peptide = np.array([a_map[x] for x in peptide],dtype=np.uint16)

				# modpeptide is the same as peptide but with modified amino acids
				# converted to other integers (beware: these are hard coded in ms2pipfeatures_c.c for now)
				modpeptide = np.array(peptide[:],dtype=np.uint16)
				peplen = len(peptide)

				nptm = 0
				cptm = 0
				if mods != '-':
					l = mods.split('|')
					for i in range(0,len(l),2):
						tl = l[i+1]
						if int(l[i]) == 0:
							if tl in Ntermmap:
								nptm += Ntermmap[tl]
							else:
								nptm += Ntermmap[tl[:-1]]
						elif int(l[i]) == -1:
							if tl in Ctermmap:
								cptm += Ctermmap[tl]
							else:
								cptm += Ctermmap[tl[:-1]]
						else:
							if tl in PTMmap:
								modpeptide[int(l[i])-1] = PTMmap[tl]
							else:
								modpeptide[int(l[i])-1] = PTMmap[tl[:-1]]

				if args.i:
					#remove reporter ionsi
					for mi,mp in enumerate(msms):
						if (mp >= 113) & (mp <= 118):
							peaks[mi]=0

				# normalize and convert MS2 peaks
				msms = np.array(msms,dtype=np.float32)
				#peaks = np.array(peaks,dtype=np.float32)
				peaks = peaks / np.sum(peaks)
				peaks = np.array(np.log2(peaks+0.001))
				peaks = peaks.astype(np.float32)

				# find the b- and y-ion peak intensities in the MS2 spectrum
				(b,y,b2,y2) = ms2pipfeatures_pyx.get_targets(modpeptide,msms,peaks,nptm,cptm,fragerror)

				#for debugging!!!!
				#tmp = pd.DataFrame(ms2pipfeatures_pyx.get_vector(peptide,modpeptide,charge),columns=cols,dtype=np.uint32)
				#print bst.predict(xgb.DMatrix(tmp))

				if args.vector_file:
					tmp = pd.DataFrame(ms2pipfeatures_pyx.get_vector(peptide,modpeptide,charge),columns=cols_n,dtype=np.uint16)
					tmp["psmid"] = [title]*len(tmp)
					tmp["targetsB"] = b
					tmp["targetsY"] = y[::-1]
					tmp["targetsB2"] = b2
					tmp["targetsY2"] = y2[::-1]
					vectors.append(tmp)
				else:
					# predict the b- and y-ion intensities from the peptide
					(resultB,resultY) = ms2pipfeatures_pyx.get_predictions(peptide,modpeptide,charge)
					for ii in range(len(resultB)):
						resultB[ii] = resultB[ii]+0.5 #This still needs to be checked!!!!!!!
					for ii in range(len(resultY)):
						resultY[ii] = resultY[ii]+0.5

					tmp = pd.DataFrame()					
					tmp['spec_id'] = [title]*(2*len(b))
					tmp['peplen'] = [peplen]*(2*len(b))
					tmp['charge'] = [charge]*(2*len(b))
					tmp['ion'] = [0]*len(b) + [1]*len(y)
					tmp['ionnumber'] = [a+1 for a in range(len(b))+range(len(y)-1,-1,-1)]
					tmp['target'] = b + y
					tmp['prediction'] = resultB + resultY
					tmp['peplen'] = tmp['peplen'].astype(np.uint8)
					tmp['charge'] = tmp['charge'].astype(np.uint8)
					tmp['ion'] = tmp['ion'].astype(np.uint8)
					tmp['ionnumber'] = tmp['ionnumber'].astype(np.uint8)
					tmp['target'] = tmp['target'].astype(np.float32)					
					tmp['prediction'] = tmp['prediction'].astype(np.float32)					
					dataresult = dataresult.append(tmp,ignore_index=True)
								
				pcount += 1
				if (pcount % 500) == 0:
					sys.stderr.write('w' + str(worker_num) + '(' + str(pcount) + ') ')

	if args.vector_file:
		return pd.concat(vectors)
	else:
		return dataresult

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
