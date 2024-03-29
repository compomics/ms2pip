#include <stdlib.h>
#include <stdio.h>
#include <string.h>


float* amino_masses;
unsigned short* amino_F;
unsigned short* sptm_mapper;
float ntermmod;


// For ms2pip_features_c.c
unsigned int v[300000];

int num_props = 4;
unsigned int props[5][19] = {
	{37,35,59,129,94,0,210,81,191,106,101,117,115,343,49,90,60,134,104}, //basicity
	{68,23,33,29,70,58,41,73,32,66,38,0,40,39,44,53,71,51,55}, //helicity
	{51,75,25,35,100,16,3,94,0,82,12,0,22,22,21,39,80,98,70}, //hydrophobicity
	{32,23,0,4,27,32,48,32,69,29,26,35,28,79,29,28,31,31,28}, //pI
	//{71,103,115,129,147,57,137,113,128,131,114,97,128,156,87,101,99,186,163} //mass
};

unsigned int props_buffer[100]; //100 is max pep length
unsigned int shared_features[100]; //100 is max num shared features
unsigned int count_n[19];
unsigned int count_c[19];
unsigned short peptide_buf[200]; //IONBOT


// Function required in ms2pip_features_c_general.c and ms2pip_features_c_catboost.c
int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}


// This function initializes amino acid masses and PTMs from a configuration file generated by Omega
void init_ms2pip(char* amino_masses_fname, char* modifications_fname, char* modifications_fname_sptm) {
	int i;
	int nummods;
	int nummods_sptm;
	float mz;
	int numptm;
	int before;
	int after;

	FILE* f = fopen(modifications_fname,"rt");
	fscanf(f,"%i\n",&nummods);
	fclose(f);

	f = fopen(modifications_fname_sptm,"rt");
	fscanf(f,"%i\n",&nummods_sptm);
	fclose(f);

	//malloc
	amino_masses = (float*) malloc((38+nummods+nummods_sptm)*sizeof(float));
	amino_F = (unsigned short*) malloc((38+nummods+nummods_sptm)*sizeof(unsigned short));
	sptm_mapper = (unsigned short*) malloc((38+nummods+nummods_sptm)*sizeof(unsigned short));

	f = fopen(amino_masses_fname,"rt");
	for (i=0; i< 19; i++) {
		fscanf(f,"%f\n",&amino_masses[i]);
		amino_F[i] = (unsigned short) (amino_masses[i]-57.021464);
		}
	fscanf(f,"%f\n",&ntermmod);
	fclose(f);

	for (i=0; i< 19; i++) {
		amino_masses[19+i]=amino_masses[i];
		amino_F[19+i] = amino_F[i];
		}

	f = fopen(modifications_fname_sptm,"rt");
	fscanf(f,"%i\n",&nummods_sptm);
	for (i=0; i< nummods_sptm; i++) {
		fscanf(f,"%f,%i,%i,%i\n",&mz,&numptm,&before,&after);
		sptm_mapper[after] = before;
		sptm_mapper[after] = before;
		if (after > 18) {
			if (before<0) {
				amino_masses[after] = mz;
			}
			else
			{
				amino_masses[after] = amino_masses[before]+mz;
				amino_F[after] = (unsigned short) (amino_masses[before]+mz - 57.021464);
			}
			}
		}
	fclose(f);
	f = fopen(modifications_fname,"rt");
	fscanf(f,"%i\n",&nummods);
	for (i=0; i< nummods; i++) {
		fscanf(f,"%f,%i,%i,%i\n",&mz,&numptm,&before,&after);
		if (after > 18) {
			if (before<0) {
				amino_masses[after] = mz;
			}
			else
			{
				amino_masses[after] = amino_masses[before]+mz;
				amino_F[after] = (unsigned short) (amino_masses[before]+mz - 57.021464);
			}
			}
		}
	fclose(f);
}
