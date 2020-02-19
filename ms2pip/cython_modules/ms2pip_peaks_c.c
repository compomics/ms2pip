#define compile_HCD 0
#define compile_HCD_ce 0
#define compile_TMT 0
#define compile_ALL 1

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ms2pip_init_c.c"
#include "ms2pip_features_c_general.c"
#include "ms2pip_features_c_ce.c"
#include "ms2pip_features_c_old.c"
#include "ms2pip_features_c_catboost.c"

// Import models
//#include "../models/HCD/pkl_B_4deep.c"
//#include "../models/modelGBM_B.c"
//#include "../models/HCD/pkl_Y_4deep.c"
//#include "../models/merged_500_Y.c"
//#include "../models/HCD/model_20190107_HCD_train_B.c"
//#include "../models/HCD/model_20190107_HCD_train_Y.c"

#if compile_HCD == 1
	#include "../models/HCD.h"
#endif

#if compile_HCD_ce == 1
	#include "../models/HCD/pkl_B_9deep.c"
	#include "../models/HCD/pkl_Y_9deep.c"
#endif

#if compile_TMT == 1
	#include "../models/TMT/tmt_human_mouse_sven_15_B.c"
	#include "../models/TMT/tmt_human_mouse_sven_20_Y.c"
#endif

#if compile_ALL == 1
	#include "../models/CID.h"
	#include "../models/HCD.h"
	#include "../models/TTOF5600.h"
	#include "../models/TMT.h"
	#include "../models/iTRAQ.h"
	#include "../models/iTRAQphospho.h"
#endif

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
	int i;

#if compile_HCD_ce == 1
	// HCD with collision energy
	if (model_id == 1) {
		unsigned int* v = get_v_ms2pip_ce(peplen, peptide, modpeptide, charge, ce);
		int fnum = v[0]/(peplen-1);
		for (i=0; i < peplen-1; i++) {
			predictions[0*(peplen-1)+i] = score_HCD_B(v+1+(i*fnum))+0.5;
			predictions[2*(peplen-1)-i-1] = score_HCD_Y(v+1+(i*fnum))+0.5;
		}
	}
#endif
#if compile_TMT == 1
	// TMT with integer intentities
	if (model_id == 3) {
		unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
		int fnum = v[0]/(peplen-1);
		for (i=0; i < peplen-1; i++) {
			predictions[0*(peplen-1)+i] = (((float)score_TMT_B(v+1+(i*fnum)))/1000)+0.5;
			predictions[2*(peplen-1)-i-1] = (((float)score_TMT_Y(v+1+(i*fnum)))/1000)+0.5;
		}
	}
#endif

#if compile_HCD == 1
	// HCD
	if (model_id == 1) {
		unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
		int fnum = v[0]/(peplen-1);
		for (i=0; i < peplen-1; i++) {
			predictions[0*(peplen-1)+i] = score_HCD_B(v+1+(i*fnum))+0.5;
			predictions[2*(peplen-1)-i-1] = score_HCD_Y(v+1+(i*fnum))+0.5;
		}
	}
#endif


#if compile_ALL == 1
	// CID
	if (model_id == 0) {
		unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
		int fnum = v[0]/(peplen-1);
		for (i=0; i < peplen-1; i++) {
			predictions[0*(peplen-1)+i] = score_CID_B(v+1+(i*fnum))+0.5;
			predictions[2*(peplen-1)-i-1] = score_CID_Y(v+1+(i*fnum))+0.5;
		}
	}

	// HCD
	if (model_id == 1) {
		unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
		int fnum = v[0]/(peplen-1);
		for (i=0; i < peplen-1; i++) {
			predictions[0*(peplen-1)+i] = score_HCD_B(v+1+(i*fnum))+0.5;
			predictions[2*(peplen-1)-i-1] = score_HCD_Y(v+1+(i*fnum))+0.5;
		}
	}

	// TTOF5600
	if (model_id == 2) {
		unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
		int fnum = v[0]/(peplen-1);
		for (i=0; i < peplen-1; i++) {
			predictions[0*(peplen-1)+i] = score_TTOF5600_B(v+1+(i*fnum))+0.5;
			predictions[2*(peplen-1)-i-1] = score_TTOF5600_Y(v+1+(i*fnum))+0.5;
		}
	}

	// iTRAQ
	if (model_id == 4) {
		unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
		int fnum = v[0]/(peplen-1);
		for (i=0; i < peplen-1; i++) {
			predictions[0*(peplen-1)+i] = score_iTRAQ_B(v+1+(i*fnum))+0.5;
			predictions[2*(peplen-1)-i-1] = score_iTRAQ_Y(v+1+(i*fnum))+0.5;
		}
	}

	// iTRAQphospho
	if (model_id == 5) {
		unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
		int fnum = v[0]/(peplen-1);
		for (i=0; i < peplen-1; i++) {
			predictions[0*(peplen-1)+i] = score_iTRAQphospho_B(v+1+(i*fnum))+0.5;
			predictions[2*(peplen-1)-i-1] = score_iTRAQphospho_Y(v+1+(i*fnum))+0.5;
		}
	}
	
	// EThcD
	//if (model_id == 6) {
	//	unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
	//	int fnum = v[0]/(peplen-1);
	//	for (i=0; i < peplen-1; i++) {
	//		predictions[0*(peplen-1)+i] = score_EThcD_B(v+1+(i*fnum))+0.5;
	//		predictions[2*(peplen-1)-i-1] = score_EThcD_Y(v+1+(i*fnum))+0.5;
	//		predictions[2*(peplen-1)+i] = score_EThcD_C(v+1+(i*fnum))+0.5;
	//		predictions[4*(peplen-1)-i-1] = score_EThcD_Z(v+1+(i*fnum))+0.5;
	//	}
	//}

	// HCDch2
	if (model_id == 7) {
		unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
		int fnum = v[0]/(peplen-1);
		for (i=0; i < peplen-1; i++) {
			predictions[0*(peplen-1)+i] = score_HCD_B(v+1+(i*fnum))+0.5;
			predictions[2*(peplen-1)-i-1] = score_HCD_Y(v+1+(i*fnum))+0.5;
			predictions[2*(peplen-1)+i] = score_HCD_B2(v+1+(i*fnum))+0.5;
			predictions[4*(peplen-1)-i-1] = score_HCD_Y2(v+1+(i*fnum))+0.5;
		}
	}
#endif
	return predictions;
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
		membuffer[j++] = mz+1.007236;  //b-ion
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = 18.0105647 + mz + 1.007236;  //y-ion
	}

	mz = 0;
	if (modpeptide[0] != 0) {
		mz = amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = (mz + 1.007236 + 1.007236)/2;  //b2-ion: (b-ion + H)/2
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = (18.0105647 + mz + 1.007236 + 1.007236)/2;  //y2-ion: (y-ion + H)/2
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
		membuffer[j++] = mz + 1.007236;  //b-ion
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = 18.0105647 + mz + 1.007236;  //y-ion
	}

	mz = 0;
	if (modpeptide[0] != 0) {
		mz = amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = mz + 1.007825032 + 17.0265491;  //c-ion: peptide + H + NH3
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = mz + 17.00273965 - 15.01089904 + 1.007825032;  //z-ion: peptide + OH - NH
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
		membuffer[j++] = mz+1.007236;  //b-ion
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = 18.0105647 + mz + 1.007236;  //y-ion
	}

	mz = 0;
	if (modpeptide[0] != 0) {
		mz = amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = (mz + 1.007236 + 1.007236)/2;  //b2-ion: (b-ion + H)/2
	}

	mz = 0;
	if (modpeptide[peplen+1] != 0) {
		mz = amino_masses[modpeptide[peplen+1]];
	}
	for (i=peplen; i > 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = (18.0105647 + mz + 1.007236 + 1.007236)/2;  //y2-ion: (y-ion + H)/2
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
		ions[i] = -9.96578428466; //HARD CODED!!
		mzs[i] = 0; //HARD CODED!!
	}

	//b-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	int pos = 0;
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[pos++] = mz+1.007236-27.99492; // a
		membuffer[pos++] = mz+1.007236-18.010565; // -H2O
		membuffer[pos++] = mz+1.007236-17.026001; // -NH3
		membuffer[pos++] = mz+1.007236; // b
		membuffer[pos++] = mz+1.007236+17.02654; // c
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
		membuffer[pos++] = (mz+1.007236+1.007236-27.99492)/2.; // a
		membuffer[pos++] = (mz+1.007236+1.007236-18.010565)/2.; // -H2O
		membuffer[pos++] = (mz+1.007236+1.007236-17.026001)/2.; // -NH3
		membuffer[pos++] = (mz+1.007236+1.007236)/2.; // b
		membuffer[pos++] = (mz+1.007236+1.007236+17.02654)/2.; // c
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
		membuffer[pos++] = 18.0105647+mz+1.007236-18.010565; //-H2O
		membuffer[pos++] = 18.0105647+mz+1.007236-17.02545; // z
		membuffer[pos++] = 18.0105647+mz+1.007236; // y
		membuffer[pos++] = 18.0105647+mz+1.007236+25.97926; // x
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
		membuffer[pos++] = (18.0105647+mz+1.007236+1.007236-18.010565)/2.; //-H2O
		membuffer[pos++] = (18.0105647+mz+1.007236+1.007236-17.02545)/2.; // z
		membuffer[pos++] = (18.0105647+mz+1.007236+1.007236)/2.; // y
		membuffer[pos++] = (18.0105647+mz+1.007236+1.007236+25.97926)/2.; // x
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
		ions[i] = -9.96578428466; //HARD CODED!!
	}

	//b-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[i-1] = mz+1.007236;
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
		membuffer[j] = 18.0105647+mz+1.007236;
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
		ions[i] = -9.96578428466; //HARD CODED!!
	}

	//b-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[i-1] = mz+1.007236;
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
		membuffer[j] = 18.0105647+mz+1.007236;
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
		membuffer[i-1] = mz + 1.007825032 + 17.026549;
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
		membuffer[j] = mz + 17.00274 - 15.010899 + 1.007825032;
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
		ions[i] = -9.96578428466; //HARD CODED!!
	}

	//b-ions
	mz = ntermmod;
	if (modpeptide[0] != 0) {
		mz += amino_masses[modpeptide[0]];
	}
	for (i=1; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[i-1] = mz+1.007236;
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
		membuffer[j] = 18.0105647+mz+1.007236;
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
		membuffer[i-1] = (mz + 1.007236 + 1.007236)/2;
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
		membuffer[j] = (18.0105647 + mz + 1.007236 + 1.007236)/2;
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
