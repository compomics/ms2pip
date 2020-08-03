#cython: language_level=3
import sys
import numpy as np
cimport numpy as np
from ms2pip.exceptions import PeptideTooLongError


NUM_ION_TYPES_MAPPING = {'general': 2, 'etd': 4, 'ch2': 4, 'all': 18}


cdef extern from "ms2pip.h":
    cdef int MAX_PEPLEN

    ctypedef struct annotations:
        float* peaks
        float* msms

    void init_ms2pip(char* amino_masses_fname, char* modifications_fname, char* modifications_fname_sptm)

    unsigned int* get_ms2pip_feature_vector(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge, int ce)

    float* get_ms2pip_predictions(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge, int model_id, int ce)

    float* get_mz_ms2pip_general(int peplen, unsigned short* modpeptide)
    float* get_mz_ms2pip_etd(int peplen, unsigned short* modpeptide)
    float* get_mz_ms2pip_ch2(int peplen, unsigned short* modpeptide)

    annotations get_t_ms2pip_all(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz)
    float* get_t_ms2pip_general(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz)
    float* get_t_ms2pip_etd(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz)
    float* get_t_ms2pip_ch2(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz)


def ms2pip_init(amino_masses_fname, modifications_fname, modifications_fname_sptm):
    if not isinstance(amino_masses_fname, bytearray):
        amino_masses_fname = bytearray(amino_masses_fname.encode())
    if not isinstance(modifications_fname, bytearray):
        modifications_fname = bytearray(modifications_fname.encode())
    if not isinstance(modifications_fname_sptm, bytearray):
        modifications_fname_sptm = bytearray(modifications_fname_sptm.encode())
    init_ms2pip(amino_masses_fname, modifications_fname, modifications_fname_sptm)


def get_vector(np.ndarray[unsigned short, ndim=1, mode="c"] peptide,
               np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
               charge):

    cdef unsigned int* results = get_ms2pip_feature_vector(len(peptide)-2, &peptide[0], &modpeptide[0], charge, -1)

    r = []
    offset = 0
    fnum = int(results[0] / (len(peptide) - 3))
    for i in range(len(peptide) - 3):
        v = []
        for j in range(fnum):
            v.append(results[j + 1 + offset])
        offset += fnum
        r.append(np.array(v, dtype=np.uint16))

    return r


def get_vector_ce(np.ndarray[unsigned short, ndim=1, mode="c"] peptide,
                  np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
                  charge, ce):

    cdef unsigned int* results = get_ms2pip_feature_vector(len(peptide)-2, &peptide[0], &modpeptide[0], charge, ce)

    r = []
    offset = 0
    fnum = int(results[0] / (len(peptide) - 3))
    for i in range(len(peptide) - 3):
        v = []
        for j in range(fnum):
            v.append(results[j + 1 + offset])
        offset += fnum
        r.append(np.array(v, dtype=np.uint16))

    return r


def get_predictions(np.ndarray[unsigned short, ndim=1, mode="c"] peptide,
                    np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
                    charge, model_id, peaks_version, ce):
    cdef float* results = get_ms2pip_predictions(len(peptide)-2, &peptide[0], &modpeptide[0], charge, model_id, ce)
    if results is NULL:
        raise NotImplementedError(model_id)
    result_parsed = []
    for i in range(NUM_ION_TYPES_MAPPING[peaks_version]):
        tmp = []
        for j in range(len(modpeptide)-3):
            tmp.append(results[(len(modpeptide)-3) * i + j])
        result_parsed.append(tmp)
    return result_parsed


def get_targets(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
                np.ndarray[float, ndim=1, mode="c"] msms,
                np.ndarray[float, ndim=1, mode="c"] peaks,
                fragerror, peaks_version):
    peplen = len(modpeptide) - 2
    if peplen > MAX_PEPLEN:
        raise PeptideTooLongError(peplen)

    if peaks_version == 'general':
        result = get_targets_general(modpeptide, msms, peaks, fragerror, peaks_version)
    elif peaks_version == 'etd':
        result = get_targets_etd(modpeptide, msms, peaks, fragerror, peaks_version)
    elif peaks_version == 'ch2':
        result = get_targets_ch2(modpeptide, msms, peaks, fragerror, peaks_version)
    elif peaks_version == 'all':
        result = get_targets_all(modpeptide, msms, peaks, fragerror, peaks_version)
    return result


def get_targets_all(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
                    np.ndarray[float, ndim=1, mode="c"] msms,
                    np.ndarray[float, ndim=1, mode="c"] peaks,
                    fragerror, peaks_version):
    cdef annotations results
    results = get_t_ms2pip_all(len(modpeptide)-2, &modpeptide[0], len(peaks), &msms[0], &peaks[0], fragerror)
    result_peaks = []
    result_mzs = []
    for i in range(NUM_ION_TYPES_MAPPING[peaks_version]*(len(modpeptide)-3)):
        result_peaks.append(results.peaks[i])
        result_mzs.append(results.msms[i])
    return (result_mzs,result_peaks)

def get_targets_general(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
                        np.ndarray[float, ndim=1, mode="c"] msms,
                        np.ndarray[float, ndim=1, mode="c"] peaks,
                        fragerror, peaks_version):
    cdef float* results = get_t_ms2pip_general(len(modpeptide)-2, &modpeptide[0], len(peaks), &msms[0], &peaks[0], fragerror)
    result_parsed = []
    for i in range(NUM_ION_TYPES_MAPPING[peaks_version]): #SD: HAd to change this
        tmp = []
        for j in range(len(modpeptide)-3):
            tmp.append(results[(len(modpeptide)-3) * i + j])
        result_parsed.append(tmp)
    return result_parsed

def get_targets_etd(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
                    np.ndarray[float, ndim=1, mode="c"] msms,
                    np.ndarray[float, ndim=1, mode="c"] peaks,
                    fragerror, peaks_version):
    cdef float* results = get_t_ms2pip_etd(len(modpeptide)-2, &modpeptide[0], len(peaks), &msms[0], &peaks[0], fragerror)
    result_parsed = []
    for i in range(NUM_ION_TYPES_MAPPING[peaks_version]):
        tmp = []
        for j in range(len(modpeptide)-3):
            tmp.append(results[(len(modpeptide)-3) * i + j])
        result_parsed.append(tmp)
    return result_parsed


def get_targets_ch2(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
                    np.ndarray[float, ndim=1, mode="c"] msms,
                    np.ndarray[float, ndim=1, mode="c"] peaks,
                    fragerror, peaks_version):
    cdef float* results = get_t_ms2pip_ch2(len(modpeptide)-2, &modpeptide[0], len(peaks), &msms[0], &peaks[0], fragerror)
    result_parsed = []
    for i in range(NUM_ION_TYPES_MAPPING[peaks_version]):
        tmp = []
        for j in range(len(modpeptide)-3):
            tmp.append(results[(len(modpeptide)-3) * i + j])
        result_parsed.append(tmp)
    return result_parsed


def get_mzs(*args):
    if args[1] == 'general':
        result = get_mzs_general(*args)
    if args[1] == 'etd':
        result = get_mzs_etd(*args)
    if args[1] == 'ch2':
        result = get_mzs_general(*args)
    return result


def get_mzs_general(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
                    peaks_version):
    cdef float* results = get_mz_ms2pip_general(len(modpeptide)-2, &modpeptide[0])
    result_parsed = []
    for i in range(NUM_ION_TYPES_MAPPING[peaks_version]):
        tmp = []
        for j in range(len(modpeptide)-3):
            tmp.append(results[(len(modpeptide)-3) * i + j])
        result_parsed.append(tmp)
    return result_parsed


def get_mzs_etd(np.ndarray[unsigned short, ndim=1, mode="c"] modpeptide,
                peaks_version):
    cdef float* results = get_mz_ms2pip_etd(len(modpeptide)-2, &modpeptide[0])
    result_parsed = []
    for i in range(NUM_ION_TYPES_MAPPING[peaks_version]):
        tmp = []
        for j in range(len(modpeptide)-3):
            tmp.append(results[(len(modpeptide)-3) * i + j])
        result_parsed.append(tmp)
    return result_parsed
