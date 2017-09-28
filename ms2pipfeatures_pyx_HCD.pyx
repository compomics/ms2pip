import sys
import numpy as np
cimport numpy as np

cdef extern from "ms2pipfeatures_c_HCD.c":
	#uncomment for Omega
	#void init(char* amino_masses_fname, char* modifications_fname, char* modifications_fname_sptm)
	void c_ms2pip_init(char* amino_masses_fname)
	unsigned int* c_ms2pip_get_v(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	unsigned int* c_ms2pip_get_v_bof_chem(int peplen, unsigned short* peptide, int charge)
	float* c_ms2pip_get_p(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	float* c_ms2pip_get_t(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float nptm, float cptm, float tolmz)
	float* c_ms2pip_get_mz(int peplen, unsigned short* modpeptide, float nptm, float cptm)

#uncomment for Omega
#def ms2pip_init(amino_masses_fname, modifications_fname,modifications_fname_sptm):
#	init(amino_masses_fname, modifications_fname,modifications_fname_sptm)

def ms2pip_init(amino_masses_fname):
	c_ms2pip_init(amino_masses_fname)

def get_vector(np.ndarray[unsigned short, ndim=1, mode="c"] peptide,np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, charge):
	cdef unsigned int* result = c_ms2pip_get_v(len(peptide),&peptide[0],&modpeptide[0],charge)
	cdef int i,j,offset
	r = []
	offset = 0
	for i in range(len(peptide)-1):
		v = []
		for j in range(186):
			v.append(result[j+offset])
		offset+=186
		r.append(v)
	return r

def get_vector_bof_chem(np.ndarray[unsigned short, ndim=1, mode="c"] peptide, int charge):
	cdef unsigned int* result = c_ms2pip_get_v_bof_chem(len(peptide),&peptide[0],charge)
	cdef int i,j,offset
	r = []
	offset = 0
	for i in range(len(peptide)-1):
		v = []
		for j in range(128):
			v.append(result[j+offset])
		offset+=128
		r.append(v)
	return r

def get_mzs(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,float nptm,float cptm):
	cdef int pos = 0
	cdef int i
	cdef float* result = c_ms2pip_get_mz(len(modpeptide), &modpeptide[0], nptm, cptm)
	b = []
	for i in range(len(modpeptide)-1):
		b.append(result[pos])
		pos += 1
	y = []
	for i in range(len(modpeptide)-1):
		y.append(result[pos])
		pos+=1
	return(b,y)
	
def get_targets(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, np.ndarray[float, ndim=1, mode="c"] msms, np.ndarray[float, ndim=1, mode="c"] peaks,float nptm,float cptm, float tolmz):
	cdef int plen = len(modpeptide)
	cdef float* result = c_ms2pip_get_t(plen,&modpeptide[0],len(peaks),&msms[0],&peaks[0],nptm,cptm,tolmz)
	cdef int i
	
	b = []
	for i in range(plen-1):
		b.append(result[i])
	y = []
	for i in range(plen-1):
		y.append(result[(plen-1)+i])
	b2 = []
	for i in range(plen-1):
		b2.append(result[2*(plen-1)+i])
	y2 = []
	for i in range(plen-1):
		y2.append(result[3*(plen-1)+i])
	return(b,y,b2,y2)

def get_predictions(np.ndarray[unsigned short, ndim=1, mode="c"] peptide,np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide, charge):
	cdef int plen = len(modpeptide)
	cdef float* predictions = c_ms2pip_get_p(plen,&peptide[0],&modpeptide[0],charge)
	cdef int i
	
	resultB = []
	resultY = []
	for i in range(plen-1):
		resultB.append(predictions[i])
	for i in range(plen-1):
		resultY.append(predictions[plen-1+i])
	return (resultB,resultY)
