#include <stdlib.h>
#include <assert.h>
#include "ms2pip.h"

#define PEPTIDE_BUFFER_SIZE (MAX_PEPLEN + 1)
#define SHARED_FEATURES_SIZE 27
#define FEATURE_VECTORS_SIZE (1 + MAX_PEPLEN * (47 + 23 * NUM_CHEMICAL_PROPERTIES))
#define CHEMICAL_PROPERTIES_BUFFER_SIZE MAX_PEPLEN

const unsigned int CHEMICAL_PROPERTIES[NUM_CHEMICAL_PROPERTIES][19] = {
    {37, 35, 59, 129, 94, 0, 210, 81, 191, 106, 101, 117, 115, 343, 49, 90, 60,
     134, 104}, // basicity
    {68, 23, 33, 29, 70, 58, 41, 73, 32, 66, 38, 0, 40, 39, 44, 53, 71, 51,
     55}, // helicity
    {51, 75, 25, 35, 100, 16, 3, 94, 0, 82, 12, 0, 22, 22, 21, 39, 80, 98,
     70}, // hydrophobicity
    {32, 23, 0, 4, 27, 32, 48, 32, 69, 29, 26, 35, 28, 79, 29, 28, 31, 31,
     28} // pI
};

unsigned short peptide_buffer[PEPTIDE_BUFFER_SIZE];
unsigned int shared_features[SHARED_FEATURES_SIZE];
unsigned int feature_vectors[FEATURE_VECTORS_SIZE];
unsigned int count_n[NUM_AMINO_MASSES];
unsigned int count_c[NUM_AMINO_MASSES];
unsigned int chemical_properties_buffer[CHEMICAL_PROPERTIES_BUFFER_SIZE];

int cmpfunc (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}

// Compute feature vectors from peptide
// TODO: ce
unsigned int* get_ms2pip_feature_vector(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge, int ce)
{
    int i, j, k;
    int fnum = 1; //first value in v is its length

    for (i = 0; i < NUM_AMINO_MASSES; i++) {
        count_n[i] = 0;
        count_c[i] = 0;
    }

    //I need this for Omega
    //important for sptms!!
    // TODO: only do this when we use sptms
    peptide_buffer[0] = peptide[0];
    for (i = 0; i < peplen; i++) {
        if (peptide[i + 1] > 18) {
            peptide_buffer[i + 1] = sptm_mapper[peptide[i + 1]];
        }
        else {
            peptide_buffer[i + 1] = peptide[i + 1];
        }
        count_c[peptide_buffer[i + 1]]++;
    }

    int num_shared = 0;

    shared_features[num_shared++] = peplen;
    shared_features[num_shared++] = charge;

    // NOTE: in theory a small optimisation (setting all to 0, and using charge
    // as an index to set the correct one to 1) is possible here by setting
    // then all to 0, and one to 1. But this is more readable, and you might
    // not notice.
    shared_features[num_shared++] = (charge == 1) ? 1 : 0;
    shared_features[num_shared++] = (charge == 2) ? 1 : 0;
    shared_features[num_shared++] = (charge == 3) ? 1 : 0;
    shared_features[num_shared++] = (charge == 4) ? 1 : 0;
    shared_features[num_shared++] = (charge >= 5) ? 1 : 0;

    for (j = 0; j < NUM_CHEMICAL_PROPERTIES; j++) {
        for (i = 0; i < peplen; i++) {
            chemical_properties_buffer[i] = CHEMICAL_PROPERTIES[j][peptide_buffer[i + 1]];
        }
        qsort(chemical_properties_buffer, peplen, sizeof(unsigned int), cmpfunc);
        shared_features[num_shared++] = chemical_properties_buffer[0];
        shared_features[num_shared++] = chemical_properties_buffer[(int)(0.25 * (peplen - 1))];
        shared_features[num_shared++] = chemical_properties_buffer[(int)(0.5 * (peplen - 1))];
        shared_features[num_shared++] = chemical_properties_buffer[(int)(0.75 * (peplen - 1))];
        shared_features[num_shared++] = chemical_properties_buffer[peplen - 1];
    }

    assert(num_shared == SHARED_FEATURES_SIZE);

    for (i = 0; i < peplen - 1; i++) {
        for (j = 0; j<num_shared; j++) {
            feature_vectors[fnum++] = shared_features[j];
        }
        feature_vectors[fnum++] = i + 1;
        feature_vectors[fnum++] = peplen - i;
        count_n[peptide_buffer[i + 1]]++;
        count_c[peptide_buffer[peplen - i]]--;

        for (j = 0; j < 19; j++) {
            feature_vectors[fnum++] = count_n[j];
            feature_vectors[fnum++] = count_c[j];
        }

        for (j = 0; j < NUM_CHEMICAL_PROPERTIES; j++) {
            feature_vectors[fnum++] = CHEMICAL_PROPERTIES[j][peptide_buffer[1]];
            feature_vectors[fnum++] = CHEMICAL_PROPERTIES[j][peptide_buffer[peplen]];
            if (i==0) {
                feature_vectors[fnum++] = 0;
            }
            else {
                feature_vectors[fnum++] = CHEMICAL_PROPERTIES[j][peptide_buffer[i - 1]];
            }
            feature_vectors[fnum++] = CHEMICAL_PROPERTIES[j][peptide_buffer[i]];
            feature_vectors[fnum++] = CHEMICAL_PROPERTIES[j][peptide_buffer[i + 1]];
            if (i==(peplen - 1)) {
                feature_vectors[fnum++] = 0;
            }
            else {
                feature_vectors[fnum++] = CHEMICAL_PROPERTIES[j][peptide_buffer[i + 2]];
            }
            unsigned int s = 0;
            for (k = 0; k <= i; k++) {
                chemical_properties_buffer[k] = CHEMICAL_PROPERTIES[j][peptide_buffer[k + 1]];
                s += chemical_properties_buffer[k];
            }
            feature_vectors[fnum++] = s;
            qsort(chemical_properties_buffer, i + 1, sizeof(unsigned int), cmpfunc);
            feature_vectors[fnum++] = chemical_properties_buffer[0];
            feature_vectors[fnum++] = chemical_properties_buffer[(int)(0.25 * i)];
            feature_vectors[fnum++] = chemical_properties_buffer[(int)(0.5 * i)];
            feature_vectors[fnum++] = chemical_properties_buffer[(int)(0.75 * i)];
            feature_vectors[fnum++] = chemical_properties_buffer[i];
            s = 0;
            for (k = i + 1; k < peplen; k++) {
                chemical_properties_buffer[k - i - 1] = CHEMICAL_PROPERTIES[j][peptide_buffer[k + 1]];
                s += chemical_properties_buffer[k - i - 1];
            }
            feature_vectors[fnum++] = s;
            qsort(chemical_properties_buffer, peplen - i - 1, sizeof(unsigned int), cmpfunc);
            feature_vectors[fnum++] = chemical_properties_buffer[0];
            feature_vectors[fnum++] = chemical_properties_buffer[(int)(0.25 * (peplen - i - 1))];
            feature_vectors[fnum++] = chemical_properties_buffer[(int)(0.5 * (peplen - i - 1))];
            feature_vectors[fnum++] = chemical_properties_buffer[(int)(0.75 * (peplen - i - 1))];
            feature_vectors[fnum++] = chemical_properties_buffer[peplen - i - 2];
        }

        // TODO: correct?
        if (ce >= 0) {
            feature_vectors[fnum++] = ce;
        }
    }
    assert((peplen - 1) * (47 + 23 * NUM_CHEMICAL_PROPERTIES) == fnum - 1);
    feature_vectors[0] = fnum - 1;
    return feature_vectors;
}
