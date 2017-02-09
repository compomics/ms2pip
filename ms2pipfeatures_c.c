#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "models/modelB.c"
#include "models/modelY.c"

//#include "models/dB.c"
//#include "models/dY.c"


float membuffer[10000];
unsigned int v[30000];
float ions[1000];
float predictions[1000];

//unsigned short bas[19] = {2064,2062,2086,2156,2121,2027,2237,2108,2218,2133,2128,2144,2142,2370,2076,2117,2087,2161,2131};
//unsigned short heli[19] = {124,79,89,85,126,114,97,129,88,122,94,56,96,95,100,109,127,107,111};
//unsigned short hydro[19] = {516,750,250,350,1000,169,37,941,0,823,121,8,224,223,215,392,802,988,700};
//unsigned short pI[19] = {600,507,277,322,548,597,759,602,974,574,541,630,565,1076,568,560,596,589,566};
//float a_mass[19] = {71.037114,103.009185,115.026943,129.042593,147.068414,57.021464,137.058912,113.084064,128.094963,131.040485,114.042927,97.052764,128.058578,156.101111,87.032028,101.047679,99.068414,186.079313,163.063329};

//hack: fixed C + Oxidation on 19
float amino_masses[20] = {71.037114,160.030645,115.026943,129.042593,147.068414,57.021464,137.058912,113.084064,128.094963,131.040485,114.042927,97.052764,128.058578,156.101111,87.032028,101.047679,99.068414,186.079313,163.063329,147.0354};

unsigned short bas[19] = {37,35,59,129,94,0,210,81,191,106,101,117,115,343,49,90,60,134,104};
unsigned short heli[19] = {68,23,33,29,70,58,41,73,32,66,38,0,40,39,44,53,71,51,55};
unsigned short hydro[19] = {51,75,25,35,100,16,3,94,0,82,12,0,22,22,21,39,80,98,70};
unsigned short pI[19] = {32,23,0,4,27,32,48,32,69,29,26,35,28,79,29,28,31,31,28};
unsigned short amino_F[20] = {14,103,58,72,90,0,80,56,71,74,57,40,71,99,30,44,42,129,106,90};

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
		ions[i] = -9.96578428466; //HARD CODED!!
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
		membuffer[j] = 18.0105647+mz+1.007236;
		//printf("%f ",membuffer[j]);
		j++;
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
			//printf("F %f %f\n",mz,max);
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
		mz += amino_F[modpeptide[i]];
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
	int fnum = 0;
	for (i=0; i < peplen-1; i++) {
		v[fnum++] = mz;
		v[fnum++] = peplen;
		v[fnum++] = i;
		v[fnum++] = (int) 100*(float)i/peplen;
		v[fnum++] = mean_mz;
		v[fnum++] = mean_bas;
		v[fnum++] = mean_heli;
		v[fnum++] = mean_hydro;
		v[fnum++] = mean_pI;

		mzb += amino_F[modpeptide[i]];
		v[fnum++] = (int) mzb;
		v[fnum++] = (int) (mz - mzb);
		v[fnum++] = (int) (mzb/(i+1));
		v[fnum++] = (int) ((mz-mzb)/(peplen-1-i));
		sum_bas += bas[peptide[i]];
		v[fnum++] = sum_bas;
		v[fnum++] = total_bas-sum_bas;
		v[fnum++] = (int) ((float)sum_bas/(i+1));
		v[fnum++] = (int) ((float)(total_bas-sum_bas)/(peplen-1-i));
		sum_heli += heli[peptide[i]];
		v[fnum++] = sum_heli;
		v[fnum++] = total_heli-sum_heli;
		v[fnum++] = (int) ((float)sum_heli/(i+1));
		v[fnum++] = (int) ((float)(total_heli-sum_heli)/(peplen-1-i));
		sum_hydro += hydro[peptide[i]];
		v[fnum++] = sum_hydro;
		v[fnum++] = total_hydro-sum_hydro;
		v[fnum++] = (int) ((float)sum_hydro/(i+1));
		v[fnum++] = (int) ((float)(total_hydro-sum_hydro)/(peplen-1-i));
		sum_pI += pI[peptide[i]];
		v[fnum++] = sum_pI;
		v[fnum++] = total_pI-sum_pI;
		v[fnum++] = (int) ((float)sum_pI/(i+1));
		v[fnum++] = (int) ((float)(total_pI-sum_pI)/(peplen-1-i));

		int pos = 0;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];			
		v[fnum] = 0;
		if (peptide[pos] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 13) {
			v[fnum] = 1;
		}
		fnum++;

		pos = 1;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];			
		v[fnum] = 0;
		if (peptide[pos] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 13) {
			v[fnum] = 1;
		}
		fnum++;
		
		pos = peplen-2;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];			
		v[fnum] = 0;
		if (peptide[pos] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 13) {
			v[fnum] = 1;
		}
		fnum++;

		pos = peplen-1;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];			
		v[fnum] = 0;
		if (peptide[pos] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 13) {
			v[fnum] = 1;
		}
		fnum++;

		v[fnum] = 0;
		if (peptide[i] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i] == 13) {
			v[fnum] = 1;
		}
		fnum++;

		v[fnum] = 0;
		if (peptide[i+1] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+1] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+1] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+1] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+1] == 13) {
			v[fnum] = 1;
		}
		fnum++;


		v[fnum++] = bas[peptide[i]];
		if (i==0) {
			v[fnum++] = bas[peptide[i]];
		}
		else {
			v[fnum++] = bas[peptide[i-1]];
		}	
		v[fnum++] = bas[peptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = bas[peptide[i+1]];
		}
		else {
			v[fnum++] = bas[peptide[i+2]];
		}
			
		v[fnum++] = heli[peptide[i]];
		if (i==0) {
			v[fnum++] = heli[peptide[i]];
		}
		else {
			v[fnum++] = heli[peptide[i-1]];
		}	
		v[fnum++] = heli[peptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = heli[peptide[i+1]];
		}
		else {
			v[fnum++] = heli[peptide[i+2]];
		}	
		
		v[fnum++] = hydro[peptide[i]];
		if (i==0) {
			v[fnum++] = hydro[peptide[i]];
		}
		else {
			v[fnum++] = hydro[peptide[i-1]];
		}	
		v[fnum++] = hydro[peptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = hydro[peptide[i+1]];
		}
		else {
			v[fnum++] = hydro[peptide[i+2]];
		}	
		
		v[fnum++] = pI[peptide[i]];
		if (i==0) {
			v[fnum++] = pI[peptide[i]];
		}
		else {
			v[fnum++] = pI[peptide[i-1]];
		}	
		v[fnum++] = pI[peptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = pI[peptide[i+1]];
		}
		else {
			v[fnum++] = pI[peptide[i+2]];
		}	
		
		v[fnum++] = amino_F[modpeptide[i]];
		if (i==0) {
			v[fnum++] = amino_F[modpeptide[i]];
		}
		else {
			v[fnum++] = amino_F[modpeptide[i-1]];
		}	
		v[fnum++] = amino_F[modpeptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = amino_F[modpeptide[i+1]];
		}
		else {
			v[fnum++] = amino_F[modpeptide[i+2]];
		}	

		v[fnum++] = charge;		
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
		mz += amino_F[modpeptide[i]];
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
		int fnum = 0;
		v[fnum++] = mz;
		v[fnum++] = peplen;
		v[fnum++] = i;
		v[fnum++] = (int) 100*(float)i/peplen;
		v[fnum++] = mean_mz;
		v[fnum++] = mean_bas;
		v[fnum++] = mean_heli;
		v[fnum++] = mean_hydro;
		v[fnum++] = mean_pI;

		mzb += amino_F[modpeptide[i]];
		v[fnum++] = (int) mzb;
		v[fnum++] = (int) (mz - mzb);
		v[fnum++] = (int) (mzb/(i+1));
		v[fnum++] = (int) ((mz-mzb)/(peplen-1-i));
		sum_bas += bas[peptide[i]];
		v[fnum++] = sum_bas;
		v[fnum++] = total_bas-sum_bas;
		v[fnum++] = (int) ((float)sum_bas/(i+1));
		v[fnum++] = (int) ((float)(total_bas-sum_bas)/(peplen-1-i));
		sum_heli += heli[peptide[i]];
		v[fnum++] = sum_heli;
		v[fnum++] = total_heli-sum_heli;
		v[fnum++] = (int) ((float)sum_heli/(i+1));
		v[fnum++] = (int) ((float)(total_heli-sum_heli)/(peplen-1-i));
		sum_hydro += hydro[peptide[i]];
		v[fnum++] = sum_hydro;
		v[fnum++] = total_hydro-sum_hydro;
		v[fnum++] = (int) ((float)sum_hydro/(i+1));
		v[fnum++] = (int) ((float)(total_hydro-sum_hydro)/(peplen-1-i));
		sum_pI += pI[peptide[i]];
		v[fnum++] = sum_pI;
		v[fnum++] = total_pI-sum_pI;
		v[fnum++] = (int) ((float)sum_pI/(i+1));
		v[fnum++] = (int) ((float)(total_pI-sum_pI)/(peplen-1-i));

		int pos = 0;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];			
		v[fnum] = 0;
		if (peptide[pos] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 13) {
			v[fnum] = 1;
		}
		fnum++;

		pos = 1;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];			
		v[fnum] = 0;
		if (peptide[pos] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 13) {
			v[fnum] = 1;
		}
		fnum++;
		
		pos = peplen-2;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];			
		v[fnum] = 0;
		if (peptide[pos] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 13) {
			v[fnum] = 1;
		}
		fnum++;

		pos = peplen-1;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];			
		v[fnum] = 0;
		if (peptide[pos] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[pos] == 13) {
			v[fnum] = 1;
		}
		fnum++;

		v[fnum] = 0;
		if (peptide[i] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i] == 13) {
			v[fnum] = 1;
		}
		fnum++;

		v[fnum] = 0;
		if (peptide[i+1] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+1] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+1] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+1] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+1] == 13) {
			v[fnum] = 1;
		}
		fnum++;


		v[fnum++] = bas[peptide[i]];
		if (i==0) {
			v[fnum++] = bas[peptide[i]];
		}
		else {
			v[fnum++] = bas[peptide[i-1]];
		}	
		v[fnum++] = bas[peptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = bas[peptide[i+1]];
		}
		else {
			v[fnum++] = bas[peptide[i+2]];
		}
			
		v[fnum++] = heli[peptide[i]];
		if (i==0) {
			v[fnum++] = heli[peptide[i]];
		}
		else {
			v[fnum++] = heli[peptide[i-1]];
		}	
		v[fnum++] = heli[peptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = heli[peptide[i+1]];
		}
		else {
			v[fnum++] = heli[peptide[i+2]];
		}	
		
		v[fnum++] = hydro[peptide[i]];
		if (i==0) {
			v[fnum++] = hydro[peptide[i]];
		}
		else {
			v[fnum++] = hydro[peptide[i-1]];
		}	
		v[fnum++] = hydro[peptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = hydro[peptide[i+1]];
		}
		else {
			v[fnum++] = hydro[peptide[i+2]];
		}	
		
		v[fnum++] = pI[peptide[i]];
		if (i==0) {
			v[fnum++] = pI[peptide[i]];
		}
		else {
			v[fnum++] = pI[peptide[i-1]];
		}	
		v[fnum++] = pI[peptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = pI[peptide[i+1]];
		}
		else {
			v[fnum++] = pI[peptide[i+2]];
		}	
		
		v[fnum++] = amino_F[modpeptide[i]];
		if (i==0) {
			v[fnum++] = amino_F[modpeptide[i]];
		}
		else {
			v[fnum++] = amino_F[modpeptide[i-1]];
		}	
		v[fnum++] = amino_F[modpeptide[i+1]];
		if (i==(peplen-2)) {
			v[fnum++] = amino_F[modpeptide[i+1]];
		}
		else {
			v[fnum++] = amino_F[modpeptide[i+2]];
		}	

		v[fnum++] = charge;		
				
		predictions[i] = score_B(v);
		predictions[2*peplen-2-i] = score_Y(v);		
	}	
	return predictions;
}

