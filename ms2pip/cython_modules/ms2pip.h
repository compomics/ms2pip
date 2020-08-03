#ifndef MS2PIP_H
#define MS2PIP_H

#define MAX_PEPLEN 100
#define NUM_AMINO_MASSES 19
#define NUM_CHEMICAL_PROPERTIES 4

extern float* amino_masses;
extern unsigned short* amino_F;
extern unsigned short* sptm_mapper;
extern float ntermmod;

struct annotations{
    float* peaks;
    float* msms;
};
typedef struct annotations annotations;

void init_ms2pip(char* amino_masses_fname, char* modifications_fname, char* modifications_fname_sptm);

float* get_ms2pip_predictions(int peplen, unsigned short *peptide,
                              unsigned short *modifications_buffer, int charge,
                              int model_id, int ce);

unsigned int* get_ms2pip_feature_vector(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge, int ce);


float* get_mz_ms2pip_general(int peplen, unsigned short* modpeptide);
float* get_mz_ms2pip_etd(int peplen, unsigned short* modpeptide);
annotations get_t_ms2pip_all(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz);
float* get_t_ms2pip_general(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz);
float* get_t_ms2pip_etd(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz);
float* get_t_ms2pip_ch2(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz);

#endif
