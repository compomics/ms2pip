#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ms2pip_init_c.c"
#include "ms2pip_features_c.c"

#define H_MASS 1.00782503207f
#define H2O_MASS 18.0105646837f
#define NH3_MASS 17.02654910101f
#define OH_MASS 17.002739651629998f
#define NH_MASS 15.01089903687f
#define CO_MASS 27.99491461956f

#define ZERO_INTENSITY -9.96578428466f

float membuffer[10000];
float ions[2000];
float mzs[2000];
float predictions[1000];

struct annotations{
	float* peaks;
	float* msms;
};
typedef struct annotations annotations;

//compute feature vector from peptide + predict intensities
float* get_p_ms2pip(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge, int model_id, int ce)
	{
	return NULL;
}


//get fragment ion mz values (b, y)
float* get_mz_ms2pip_general(int peplen, unsigned short* modpeptide)
	{
	int i,j;
	float mz;
	j=0;

	mz = 0;
	if (modpeptide[0] != 0) {
		mz = amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = mz+H_MASS;  //b-ion
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = H2O_MASS + mz + H_MASS;  //y-ion
	}

	mz = 0;
	if (modpeptide[0] != 0) {
		mz = amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = (mz + H_MASS + H_MASS)/2;  //b2-ion: (b-ion + H)/2
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = (H2O_MASS + mz + H_MASS + H_MASS)/2;  //y2-ion: (y-ion + H)/2
	}

	return membuffer;
}


//get fragment ion mz values (b, y, c, z)
float* get_mz_ms2pip_etd(int peplen, unsigned short* modpeptide)
	{
	int i,j;
	float mz;
	j=0;

	mz = 0;
	if (modpeptide[0] != 0) {
		mz = amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = mz + H_MASS;  //b-ion
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = H2O_MASS + mz + H_MASS;  //y-ion
	}

	mz = 0;
	if (modpeptide[0] != 0) {
		mz = amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = mz + H_MASS + NH3_MASS;  //c-ion: peptide + H + NH3
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = mz + OH_MASS - NH_MASS + H_MASS;  //z-ion: peptide + OH - NH
	}

	return membuffer;
}


//get fragment ion mz values (b, y, b++, y++)
float* get_mz_ms2pip_ch2(int peplen, unsigned short* modpeptide)
	{
	int i,j;
	float mz;
	j=0;

	mz = 0;
	if (modpeptide[0] != 0) {
		mz = amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = mz+H_MASS;  //b-ion
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = H2O_MASS + mz + H_MASS;  //y-ion
	}

	mz = 0;
	if (modpeptide[0] != 0) {
		mz = amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = (mz + H_MASS + H_MASS)/2;  //b2-ion: (b-ion + H)/2
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = (H2O_MASS + mz + H_MASS + H_MASS)/2;  //y2-ion: (y-ion + H)/2
	}

	return membuffer;
}

//get all fragment ion peaks from spectrum
annotations get_t_ms2pip_all(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz)
	{
	int i,tmp;
	float mz;
	int msms_pos;
	int mem_pos;
	float max, maxmz, tmp2;

	//for (i=0; i < numpeaks; i++) {
	//  fprintf(stderr,"m %f\n",msms[i]);
	//}

	for (i=0; i < 18*(peplen-1); i++) { // 2*9 iontypes: b: a -H2O -NH3 b c y: -H2O z y x
		ions[i] = ZERO_INTENSITY;
		mzs[i] = 0.0f;
	}

	//b-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	int pos = 0;
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[pos++] = mz+H_MASS-CO_MASS; // a
		membuffer[pos++] = mz+H_MASS-H2O_MASS; // -H2O
		membuffer[pos++] = mz+H_MASS-NH3_MASS; // -NH3
		membuffer[pos++] = mz+H_MASS; // b
		membuffer[pos++] = mz+H_MASS+NH3_MASS; // c
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= 5*(peplen-1)) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			maxmz = msms[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
						maxmz = msms[tmp];
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[mem_pos] = max;
			mzs[mem_pos] = maxmz;
			mem_pos += 1;
		}
	}

	//b2-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	pos = 0;
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[pos++] = (mz+H_MASS+H_MASS-CO_MASS)/2.; // a
		membuffer[pos++] = (mz+H_MASS+H_MASS-H2O_MASS)/2.; // -H2O
		membuffer[pos++] = (mz+H_MASS+H_MASS-NH3_MASS)/2.; // -NH3
		membuffer[pos++] = (mz+H_MASS+H_MASS)/2.; // b
		membuffer[pos++] = (mz+H_MASS+H_MASS+NH3_MASS)/2.; // c
	}

	msms_pos = 0;
	mem_pos=0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= 5*(peplen-1)) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			maxmz = msms[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
						maxmz = msms[tmp];
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[5*(peplen-1)+mem_pos] = max;
			mzs[5*(peplen-1)+mem_pos] = maxmz;
			mem_pos += 1;
		}
	}


	// y-ions
	mz = 0.;
	if (modpeptide[peplen+1] != 0) {
		mz += modpeptide[peplen+1];
	}
	pos = 0;
	for (i=peplen; i >= 2; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[pos++] = H2O_MASS+mz+H_MASS-H2O_MASS; //-H2O
		membuffer[pos++] = H2O_MASS+mz+H_MASS-NH3_MASS; // z
		membuffer[pos++] = H2O_MASS+mz+H_MASS; // y
		membuffer[pos++] = H2O_MASS+mz+CO_MASS-H_MASS; // x
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= 4*(peplen-1)) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			maxmz = msms[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
						maxmz = msms[tmp];
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[10*(peplen-1)+mem_pos] = max;
			mzs[10*(peplen-1)+mem_pos] = maxmz;
			mem_pos += 1;
		}
	}

	// y2-ions
	mz = 0.;
	if (modpeptide[peplen+1] != 0) {
		mz += modpeptide[peplen+1];
	}
	pos = 0;
	for (i=peplen; i >= 2; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[pos++] = (H2O_MASS+mz+H_MASS-H2O_MASS+H_MASS)/2; //-H2O
		membuffer[pos++] = (H2O_MASS+mz+H_MASS-NH3_MASS+H_MASS)/2; // z
		membuffer[pos++] = (H2O_MASS+mz+H_MASS+H_MASS)/2; // y
		membuffer[pos++] = (H2O_MASS+mz+CO_MASS)/2; // x
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= 4*(peplen-1)) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			maxmz = msms[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
						maxmz = msms[tmp];
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[14*(peplen-1)+mem_pos] = max;
			mzs[14*(peplen-1)+mem_pos] = maxmz;
			mem_pos += 1;
		}
	}

	//for (i=0; i < 18*(peplen-1); i++) { // 2*9 iontypes: b: a -H2O -NH3 b c y: -H2O z y x
	//    fprintf(stderr,"%f ",ions[i]); //HARD CODED!!
	//}
	//fprintf(stderr,"\n");

	struct annotations r = {ions,mzs};

	return r;
}

//get fragment ion peaks from spectrum
float* get_t_ms2pip_general(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz)
	{
	int i,j,tmp;
	float mz;
	int msms_pos;
	int mem_pos;
	float max, tmp2;

	//for (i=0; i < numpeaks; i++) {
	//  fprintf(stderr,"m %f\n",msms[i]);
	//}

	for (i=0; i < 2*(peplen-1); i++) {
		ions[i] = ZERO_INTENSITY;
	}

	//b-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[i-1] = mz+H_MASS;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[mem_pos] = max;
			mem_pos += 1;
		}
	}

	// y-ions
	mz = 0.;
	if (modpeptide[peplen+1] != 0) {
		mz += modpeptide[peplen+1];
	}
	j=0;
	for (i=peplen; i >= 2; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j] = H2O_MASS+mz+H_MASS;
		j++;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[(peplen-1)+mem_pos] = max;
			mem_pos += 1;
		}
	}


	return ions;
}



//get fragment ion peaks from spectrum (b, y, c, z)
float* get_t_ms2pip_etd(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz)
	{
	int i,j,tmp;
	float mz;
	int msms_pos;
	int mem_pos;
	float max, tmp2;

	for (i=0; i < 4*(peplen-1); i++) {
		ions[i] = ZERO_INTENSITY;
	}

	//b-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[i-1] = mz+H_MASS;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[mem_pos] = max;
			mem_pos += 1;
		}
	}

	// y-ions
	mz = 0.;
	if (modpeptide[peplen+1] != 0) {
		mz += modpeptide[peplen+1];
	}
	j=0;
	for (i=peplen; i >= 2; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j] = H2O_MASS+mz+H_MASS;
		j++;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[(peplen-1)+mem_pos] = max;
			mem_pos += 1;
		}
	}

	//c-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[i-1] = mz + H_MASS + NH3_MASS;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[2*(peplen-1)+mem_pos] = max;
			mem_pos += 1;
		}
	}

	// z-ions
	mz = 0.;
	if (modpeptide[peplen+1] != 0) {
		mz += modpeptide[peplen+1];
	}
	j=0;
	for (i=peplen; i >= 2; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j] = mz + OH_MASS - NH_MASS + H_MASS;
		j++;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[3*(peplen-1)+mem_pos] = max;
			mem_pos += 1;
		}
	}

	return ions;
}


//get fragment ion peaks from spectrum (b, y, b++, y++)
float* get_t_ms2pip_ch2(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks, float tolmz)
	{
	int i,j,tmp;
	float mz;
	int msms_pos;
	int mem_pos;
	float max, tmp2;

	//for (i=0; i < numpeaks; i++) {
	//  fprintf(stderr,"m %f\n",msms[i]);
	//}

	for (i=0; i < 4*(peplen-1); i++) {
		ions[i] = ZERO_INTENSITY;
	}

	//b-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[i-1] = mz+H_MASS;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[mem_pos] = max;
			mem_pos += 1;
		}
	}

	// y-ions
	mz = 0.;
	if (modpeptide[peplen+1] != 0) {
		mz += modpeptide[peplen+1];
	}
	j=0;
	for (i=peplen; i >= 2; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j] = H2O_MASS+mz+H_MASS;
		j++;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[(peplen-1)+mem_pos] = max;
			mem_pos += 1;
		}
	}

	//b2-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[i-1] = (mz + H_MASS + H_MASS)/2;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[2*(peplen-1)+mem_pos] = max;
			mem_pos += 1;
		}
	}

	// y2-ions
	mz = 0.;
	if (modpeptide[peplen+1] != 0) {
		mz += modpeptide[peplen+1];
	}
	j=0;
	for (i=peplen; i >= 2; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j] = (H2O_MASS + mz + H_MASS + H_MASS)/2;
		j++;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen-1) {
			break;
		}
		mz = membuffer[mem_pos];
		if (msms[msms_pos] > (mz+tolmz)) {
			mem_pos += 1;
		}
		else if (msms[msms_pos] < (mz-tolmz)) {
			msms_pos += 1;
		}
		else {
			max = peaks[msms_pos];
			tmp = msms_pos + 1;
			if (tmp < numpeaks) {
				while (msms[tmp] <= (mz+tolmz)) {
					tmp2 = peaks[tmp];
					if (max < tmp2) {
						max = tmp2;
					}
					tmp += 1;
					if (tmp == numpeaks) {
						break;
					}
				}
			}
			ions[3*(peplen-1)+mem_pos] = max;
			mem_pos += 1;
		}
	}

	return ions;
}
