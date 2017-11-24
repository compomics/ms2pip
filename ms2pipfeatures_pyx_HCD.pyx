import sys
import numpy as np
cimport numpy as np

cdef extern from "ms2pipfeatures_c_HCD.c":
	void init_ms2pip(char* amino_masses_fname, char* modifications_fname, char* modifications_fname_sptm)
	unsigned int* get_v_ms2pip(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	unsigned int* get_v_bof_chem(int peplen, unsigned short* peptide, int charge)
	float* ms2pip_get_mz(int peplen, unsigned short* modpeptide)
	float* get_p_ms2pip(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	float* get_t_ms2pip(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz)

def ms2pip_init(amino_masses_fname, modifications_fname,modifications_fname_sptm):
	init_ms2pip(amino_masses_fname, modifications_fname,modifications_fname_sptm)

def get_vector(np.ndarray[unsigned short, ndim=1, mode="c"] peptide,np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, charge):
	cdef unsigned int* results = get_v_ms2pip(len(peptide)-2,&peptide[0],&modpeptide[0],charge)
	r = []
	offset = 0
	fnum = results[0]/(len(peptide)-3)
	for i in range(len(peptide)-3):
		v = []
		for j in range(fnum):
			v.append(results[j+1+offset])
		offset+=fnum
		r.append(v)
	return r

def get_targets(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, np.ndarray[float, ndim=1, mode="c"] msms, np.ndarray[float, ndim=1, mode="c"] peaks, fragerror):
	cdef float* results = get_t_ms2pip(len(modpeptide)-2,&modpeptide[0],len(peaks),&msms[0],&peaks[0],fragerror)
	num_ions = len(modpeptide)-3
	resultB = []
	resultY = []
	for i in range(num_ions):
		resultB.append(results[0*num_ions+i])
		resultY.append(results[1*num_ions+i])
	return(resultB,resultY)

def get_predictions(np.ndarray[unsigned short, ndim=1, mode="c"] peptide, np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, charge):
	cdef float* results = get_p_ms2pip(len(peptide)-2,&peptide[0],&modpeptide[0],charge)
	num_ions = len(modpeptide)-3
	resultB = []
	resultY = []
	for i in range(num_ions):
		resultB.append(results[0*num_ions+i])
		resultY.append(results[1*num_ions+i])
	return(resultB,resultY)

def get_mzs(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide):
	cdef int pos = 0
	cdef int i
	cdef float* results = ms2pip_get_mz(len(modpeptide)-2, &modpeptide[0])
	num_ions = len(modpeptide)-3
	resultB = []
	resultY = []
	for i in range(num_ions):
		resultB.append(results[0*num_ions+i])
		resultY.append(results[1*num_ions+i])
	return(resultB,resultY)
