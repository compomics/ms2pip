import sys
import numpy as np
cimport numpy as np

cdef extern from "ms2pipfeatures_c_HCD.c":
	void init_ms2pip(char* amino_masses_fname, char* modifications_fname, char* modifications_fname_sptm)
	unsigned int* get_v_ms2pip(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	unsigned int* get_v_bof_chem(int peplen, unsigned short* peptide, int charge)
	float* ms2pip_get_mz(int peplen, unsigned short* modpeptide)
	float* get_p_ms2pip(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	float* get_t_ms2pip(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks)

def ms2pip_init(amino_masses_fname, modifications_fname,modifications_fname_sptm):
	init_ms2pip(amino_masses_fname, modifications_fname,modifications_fname_sptm)

def get_vector(np.ndarray[unsigned short, ndim=1, mode="c"] peptide,np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, charge):
	cdef unsigned int* result = get_v_ms2pip(len(peptide)-2,&peptide[0],&modpeptide[0],charge)
	r = []
	offset = 0
	for i in range(len(peptide)-1):
		v = []
		for j in range(186):
			v.append(result[j+offset])
		offset+=186
		r.append(v)
	return r

def get_targets(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, np.ndarray[float, ndim=1, mode="c"] msms, np.ndarray[float, ndim=1, mode="c"] peaks):
	cdef float* result = get_t_ms2pip(len(modpeptide)-2,&modpeptide[0],len(peaks),&msms[0],&peaks[0])
	b = []
	for i in range(len(modpeptide)-3):
		b.append(result[i])
	y = []
	for i in range(len(modpeptide)-3):
		y.append(result[len(modpeptide)-3+i])
	return(b,y)

def get_score(np.ndarray[unsigned short, ndim=1, mode="c"] peptide,np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, np.ndarray[float, ndim=1, mode="c"] msms, np.ndarray[float, ndim=1, mode="c"] peaks, charge):
	cdef float* targets = get_t_ms2pip(len(modpeptide)-2,&modpeptide[0],len(peaks),&msms[0],&peaks[0])
	cdef float* predictions = get_p_ms2pip(len(peptide)-2,&peptide[0],&modpeptide[0],charge)
	mae = 0.
	#for i in range(2*len(modpeptide)-2):
	for i in range(len(modpeptide)-1):
		sys.stdout.write("%f " % (predictions[len(modpeptide-1)+i]))
		mae += abs(targets[len(modpeptide)-1+i]-predictions[len(modpeptide)-1+i])
	sys.stdout.write("\n")
	#mae /= (2*len(modpeptide)-2)
	mae /= (len(modpeptide)-1)
	#print mae
	return mae

def get_predictions(np.ndarray[unsigned short, ndim=1, mode="c"] peptide, np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, charge):
	cdef float* predictions = get_p_ms2pip(len(peptide)-2,&peptide[0],&modpeptide[0],charge)
	resultB = []
	resultY = []
	for i in range(len(modpeptide)-3):
		resultB.append(predictions[i])
	for i in range(len(modpeptide)-3):
		resultY.append(predictions[len(modpeptide)-2+i])
	return (resultB,resultY)

def get_mzs(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide):
	cdef int pos = 0
	cdef int i
	cdef float* result = ms2pip_get_mz(len(modpeptide)-2, &modpeptide[0])
	b = []
	for i in range(len(modpeptide)-3):
		b.append(result[pos])
		pos += 1
	y = []
	for i in range(len(modpeptide)-3):
		y.append(result[pos])
		pos+=1
	return(b,y)
