#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "models/modelB.c"
#include "models/modelY.c"

float membuffer[10000];
unsigned int v[30000];
float ions[1000];
float predictions[1000];

unsigned short bas[19] = {2064,2062,2086,2156,2121,2027,2237,2108,2218,2133,2128,2144,2142,2370,2076,2117,2087,2161,2131};
unsigned short heli[19] = {124,79,89,85,126,114,97,129,88,122,94,56,96,95,100,109,127,107,111};
unsigned short hydro[19] = {516,750,250,350,1000,169,37,941,0,823,121,8,224,223,215,392,802,988,700};
unsigned short pI[19] = {600,507,277,322,548,597,759,602,974,574,541,630,565,1076,568,560,596,589,566};
//float a_mass[19] = {71.037114,103.009185,115.026943,129.042593,147.068414,57.021464,137.058912,113.084064,128.094963,131.040485,114.042927,97.052764,128.058578,156.101111,87.032028,101.047679,99.068414,186.079313,163.063329};

//hack: fixed C + Oxidation on 19
float amino_masses[20] = {71.037114,160.030645,115.026943,129.042593,147.068414,57.021464,137.058912,113.084064,128.094963,131.040485,114.042927,97.052764,128.058578,156.101111,87.032028,101.047679,99.068414,186.079313,163.063329,147.0354};

float* get_t(int peplen, unsigned short* modpeptide, int numpeaks, float* msms, float* peaks)
	{
	int i,j,tmp;
	float mz;
	int msms_pos;
	int mem_pos;
	float tolmz = 0.02;
	float max, tmp2;
	int c_stretch;
	int last_hit;
	
	//b-ions
	for (i=0; i < 2*peplen; i++) {
		ions[i] = -9.96578428466;
	}
	
	mz = 0.;
	for (i=0; i < peplen-1; i++) {
		mz += amino_masses[modpeptide[i]];
		membuffer[i] = mz+1.007236;
	}
	
	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen) {
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
	j=0;
	for (i=peplen-1; i >= 1; i--) {
		mz += amino_masses[modpeptide[i]];
		membuffer[j++] = 18.0105647+mz+1.007236;
	}

	msms_pos = 0;
	mem_pos = 0;
	while (1) {
		if (msms_pos >= numpeaks) {
			break;
		}
		if (mem_pos >= peplen) {
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

unsigned int* get_v(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	{
	int i,j,tmp;
	float mz;
	int msms_pos;
	int mem_pos;
	float tolmz = 0.02;
	unsigned short max, tmp2;
	int c_stretch;
	int last_hit;
	
	int total_bas = 0;
	int total_heli = 0;
	int total_hydro = 0;
	int total_pI = 0;

	mz = 0.;
	for (i=0; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		total_bas += bas[peptide[i]];
		total_heli += heli[peptide[i]];
		total_hydro += hydro[peptide[i]];
		total_pI += pI[peptide[i]];
	}
	
	int mean_mz = (int) ((float)mz/peplen);
	int mean_bas = (int) ((float)total_bas/peplen);
	int mean_heli = (int) ((float)total_heli/peplen);
	int mean_hydro = (int) ((float)total_hydro/peplen);
	int mean_pI = (int) ((float)total_pI/peplen);
	
	float mzb = 0.;
	int sum_bas = 0;
	int sum_heli = 0;
	int sum_hydro = 0;
	int sum_pI = 0;
	int offset = 0;
	for (i=0; i < peplen-1; i++) {
		v[0+offset] = (int) mz;
		v[1+offset] = peplen;
		v[2+offset] = i;
		v[3+offset] = (int) 100*(float)i/peplen;
		v[4+offset] = mean_mz;
		v[5+offset] = mean_bas;
		v[6+offset] = mean_heli;
		v[7+offset] = mean_hydro;
		v[8+offset] = mean_pI;

		mzb += amino_masses[i];
		v[9+offset] = (int) mzb;
		v[10+offset] = (int) (mz - mzb);
		v[11+offset] = (int) (mzb/(i+1));
		v[12+offset] = (int) ((mz-mzb)/(peplen-1-i));
		sum_bas += bas[i];
		v[13+offset] = sum_bas;
		v[14+offset] = total_bas-sum_bas;
		v[15+offset] = (int) ((float)sum_bas/(i+1));
		v[16+offset] = (int) ((float)(total_bas-sum_bas)/(peplen-1-i));
		sum_heli += heli[i];
		v[17+offset] = sum_heli;
		v[18+offset] = total_heli-sum_heli;
		v[19+offset] = (int) ((float)sum_heli/(i+1));
		v[20+offset] = (int) ((float)(total_heli-sum_heli)/(peplen-1-i));
		sum_hydro += hydro[i];
		v[21+offset] = sum_hydro;
		v[22+offset] = total_hydro-sum_hydro;
		v[23+offset] = (int) ((float)sum_hydro/(i+1));
		v[24+offset] = (int) ((float)(total_hydro-sum_hydro)/(peplen-1-i));
		sum_pI += pI[i];
		v[25+offset] = sum_pI;
		v[26+offset] = total_pI-sum_pI;
		v[27+offset] = (int) ((float)sum_pI/(i+1));
		v[28+offset] = (int) ((float)(total_pI-sum_pI)/(peplen-1-i));

		v[29+offset] = bas[peptide[0]];
		v[30+offset] = heli[peptide[0]];
		v[31+offset] = hydro[peptide[0]];
		v[32+offset] = 0;
		if (peptide[0] == 13) {
			v[32+offset] = 1;
		}
		v[33+offset] = bas[peptide[1]];
		v[34+offset] = heli[peptide[1]];
		v[35+offset] = hydro[peptide[1]];
		v[36+offset] = pI[peptide[1]];
		v[37+offset] = bas[peptide[-2]];
		v[38+offset] = heli[peptide[-2]];
		v[39+offset] = hydro[peptide[-2]];
		v[40+offset] = pI[peptide[-2]];
		v[41+offset] = heli[peptide[-1]];
		
		v[42+offset] = bas[peptide[i]];
		if (i==0) {
			v[43+offset] = bas[peptide[i]];
		}
		else {
			v[43+offset] = bas[peptide[i-1]];
		}	
		v[44+offset] = bas[peptide[i+1]];
		if (i==(peplen-2)) {
			v[45+offset] = bas[peptide[i+1]];
		}
		else {
			v[45+offset] = bas[peptide[i+2]];
		}
			
		v[46+offset] = heli[peptide[i]];
		if (i==0) {
			v[47+offset] = heli[peptide[i]];
		}
		else {
			v[47+offset] = heli[peptide[i-1]];
		}	
		v[48+offset] = heli[peptide[i+1]];
		if (i==(peplen-2)) {
			v[49+offset] = heli[peptide[i+1]];
		}
		else {
			v[49+offset] = heli[peptide[i+2]];
		}	
		
		v[50+offset] = hydro[peptide[i]];
		if (i==0) {
			v[51+offset] = hydro[peptide[i]];
		}
		else {
			v[51+offset] = hydro[peptide[i-1]];
		}	
		v[52+offset] = hydro[peptide[i+1]];
		if (i==(peplen-2)) {
			v[53+offset] = hydro[peptide[i+1]];
		}
		else {
			v[53+offset] = hydro[peptide[i+2]];
		}	
		
		v[54+offset] = pI[peptide[i]];
		if (i==0) {
			v[55+offset] = pI[peptide[i]];
		}
		else {
			v[55+offset] = pI[peptide[i-1]];
		}	
		v[56+offset] = pI[peptide[i+1]];
		if (i==(peplen-2)) {
			v[57+offset] = pI[peptide[i+1]];
		}
		else {
			v[57+offset] = pI[peptide[i+2]];
		}	
		
		v[58+offset] = (int) 10*amino_masses[modpeptide[i]];
		if (i==0) {
			v[59+offset] = (int) 10*amino_masses[modpeptide[i]];
		}
		else {
			v[59+offset] = (int) 10*amino_masses[modpeptide[i-1]];
		}	
		v[60+offset] = (int) 10*amino_masses[modpeptide[i+1]];
		if (i==(peplen-2)) {
			v[61+offset] = (int) 10*amino_masses[modpeptide[i+1]];
		}
		else {
			v[61+offset] = (int) 10*amino_masses[modpeptide[i+2]];
		}	
				
		v[62+offset] = 0;
		if (peptide[i] == 16) {
			v[62+offset] = 1;
		}
		v[63+offset] = 0;
		if (peptide[i+1] == 16) {
			v[63+offset] = 1;
		}
		v[64+offset] = 0;
		if (peptide[i] == 11) {
			v[64+offset] = 1;
		}
		v[65+offset] = 0;
		if (peptide[i+1] == 11) {
			v[65+offset] = 1;
		}
		v[66+offset] = charge;
		offset+=67;
	}	
	return v;
}

float* get_p(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	{
	int i,j,tmp;
	float mz;
	int msms_pos;
	int mem_pos;
	float tolmz = 0.02;
	unsigned short max, tmp2;
	int c_stretch;
	int last_hit;
	
	int total_bas = 0;
	int total_heli = 0;
	int total_hydro = 0;
	int total_pI = 0;

	mz = 0.;
	for (i=0; i < peplen; i++) {
		mz += amino_masses[modpeptide[i]];
		total_bas += bas[peptide[i]];
		total_heli += heli[peptide[i]];
		total_hydro += hydro[peptide[i]];
		total_pI += pI[peptide[i]];
	}
	
	int mean_mz = (int) ((float)mz/peplen);
	int mean_bas = (int) ((float)total_bas/peplen);
	int mean_heli = (int) ((float)total_heli/peplen);
	int mean_hydro = (int) ((float)total_hydro/peplen);
	int mean_pI = (int) ((float)total_pI/peplen);
	
	float mzb = 0.;
	int sum_bas = 0;
	int sum_heli = 0;
	int sum_hydro = 0;
	int sum_pI = 0;
	for (i=0; i < peplen-1; i++) {
		v[0] = (int) mz;
		v[1] = peplen;
		v[2] = i;
		v[3] = (int) 100*(float)i/peplen;
		v[4] = mean_mz;
		v[5] = mean_bas;
		v[6] = mean_heli;
		v[7] = mean_hydro;
		v[8] = mean_pI;

		mzb += amino_masses[i];
		v[9] = (int) mzb;
		v[10] = (int) (mz - mzb);
		v[11] = (int) (mzb/(i+1));
		v[12] = (int) ((mz-mzb)/(peplen-1-i));
		sum_bas += bas[i];
		v[13] = sum_bas;
		v[14] = total_bas-sum_bas;
		v[15] = (int) ((float)sum_bas/(i+1));
		v[16] = (int) ((float)(total_bas-sum_bas)/(peplen-1-i));
		sum_heli += heli[i];
		v[17] = sum_heli;
		v[18] = total_heli-sum_heli;
		v[19] = (int) ((float)sum_heli/(i+1));
		v[20] = (int) ((float)(total_heli-sum_heli)/(peplen-1-i));
		sum_hydro += hydro[i];
		v[21] = sum_hydro;
		v[22] = total_hydro-sum_hydro;
		v[23] = (int) ((float)sum_hydro/(i+1));
		v[24] = (int) ((float)(total_hydro-sum_hydro)/(peplen-1-i));
		sum_pI += pI[i];
		v[25] = sum_pI;
		v[26] = total_pI-sum_pI;
		v[27] = (int) ((float)sum_pI/(i+1));
		v[28] = (int) ((float)(total_pI-sum_pI)/(peplen-1-i));

		v[29] = bas[peptide[0]];
		v[30] = heli[peptide[0]];
		v[31] = hydro[peptide[0]];
		v[32] = 0;
		if (peptide[0] == 13) {
			v[32] = 1;
		}
		v[33] = bas[peptide[1]];
		v[34] = heli[peptide[1]];
		v[35] = hydro[peptide[1]];
		v[36] = pI[peptide[1]];
		v[37] = bas[peptide[-2]];
		v[38] = heli[peptide[-2]];
		v[39] = hydro[peptide[-2]];
		v[40] = pI[peptide[-2]];
		v[41] = heli[peptide[-1]];
		
		v[42] = bas[peptide[i]];
		if (i==0) {
			v[43] = bas[peptide[i]];
		}
		else {
			v[43] = bas[peptide[i-1]];
		}	
		v[44] = bas[peptide[i+1]];
		if (i==(peplen-2)) {
			v[45] = bas[peptide[i+1]];
		}
		else {
			v[45] = bas[peptide[i+2]];
		}
			
		v[46] = heli[peptide[i]];
		if (i==0) {
			v[47] = heli[peptide[i]];
		}
		else {
			v[47] = heli[peptide[i-1]];
		}	
		v[48] = heli[peptide[i+1]];
		if (i==(peplen-2)) {
			v[49] = heli[peptide[i+1]];
		}
		else {
			v[49] = heli[peptide[i+2]];
		}	
		
		v[50] = hydro[peptide[i]];
		if (i==0) {
			v[51] = hydro[peptide[i]];
		}
		else {
			v[51] = hydro[peptide[i-1]];
		}	
		v[52] = hydro[peptide[i+1]];
		if (i==(peplen-2)) {
			v[53] = hydro[peptide[i+1]];
		}
		else {
			v[53] = hydro[peptide[i+2]];
		}	
		
		v[54] = pI[peptide[i]];
		if (i==0) {
			v[55] = pI[peptide[i]];
		}
		else {
			v[55] = pI[peptide[i-1]];
		}	
		v[56] = pI[peptide[i+1]];
		if (i==(peplen-2)) {
			v[57] = pI[peptide[i+1]];
		}
		else {
			v[57] = pI[peptide[i+2]];
		}	
		
		v[58] = (int) 10*amino_masses[modpeptide[i]];
		if (i==0) {
			v[59] = (int) 10*amino_masses[modpeptide[i]];
		}
		else {
			v[59] = (int) 10*amino_masses[modpeptide[i-1]];
		}	
		v[60] = (int) 10*amino_masses[modpeptide[i+1]];
		if (i==(peplen-2)) {
			v[61] = (int) 10*amino_masses[modpeptide[i+1]];
		}
		else {
			v[61] = (int) 10*amino_masses[modpeptide[i+2]];
		}	
				
		v[62] = 0;
		if (peptide[i] == 16) {
			v[62] = 1;
		}
		v[63] = 0;
		if (peptide[i+1] == 16) {
			v[63] = 1;
		}
		v[64] = 0;
		if (peptide[i] == 11) {
			v[64] = 1;
		}
		v[65] = 0;
		if (peptide[i+1] == 11) {
			v[65] = 1;
		}
		v[66] = charge;
		predictions[i] = score_B(v);
		predictions[2*peplen-2-i] = score_Y(v);		
	}	
	return predictions;
}

