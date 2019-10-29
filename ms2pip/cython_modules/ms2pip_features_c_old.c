// Old MS2PIP feature vectors 


unsigned int v[300000];

unsigned short bas[19] = {37,35,59,129,94,0,210,81,191,106,101,117,115,343,49,90,60,134,104};
unsigned short heli[19] = {68,23,33,29,70,58,41,73,32,66,38,0,40,39,44,53,71,51,55};
unsigned short hydro[19] = {51,75,25,35,100,16,3,94,0,82,12,0,22,22,21,39,80,98,70};
unsigned short pI[19] = {32,23,0,4,27,32,48,32,69,29,26,35,28,79,29,28,31,31,28};
unsigned short aG[19] = {1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1};
unsigned short wikiG[19] = {0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0};


// Compute feature vectors from peptide
unsigned int* get_v_ms2pip_old(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	{
	int i,j;
	float mz;

	int fnum = 1; //first value in v is its length

	int max_bas_b = 0;
	int max_heli_b = 0;
	int max_hydro_b = 0;
	int max_pI_b = 0;
	int max_bas_y = 0;
	int max_heli_y = 0;
	int max_hydro_y = 0;
	int max_pI_y = 0;
	int min_bas_b = 999;
	int min_heli_b = 999;
	int min_hydro_b = 999;
	int min_pI_b = 999;
	int min_bas_y = 999;
	int min_heli_y = 999;
	int min_hydro_y = 999;
	int min_pI_y = 999;

	unsigned int buf2[19];
	unsigned int buf3[19];
	unsigned int sum_aG = 0;
	unsigned int sum_wikiG = 0;
	unsigned int sum_aG_tot = 0;
	unsigned int sum_wikiG_tot = 0;

	for (i=0; i < 19; i++) {
		buf2[i] = 0;
		buf3[i] = 0;
	}

	//I need this for Omega
	//important for sptms!!
	for (i=0; i < peplen; i++) {
		if (peptide[i+1] > 18) {
			peptide[i+1] = sptm_mapper[peptide[i+1]];
		}
		buf3[peptide[i+1]]++;
		sum_aG_tot += aG[peptide[i+1]];
		sum_wikiG_tot += wikiG[peptide[i+1]];
	}

	unsigned int total_bas = 0;
	unsigned int total_heli = 0;
	unsigned int total_hydro = 0;
	unsigned int total_pI = 0;
	unsigned int max_bas = 0;
	unsigned int max_heli = 0;
	unsigned int max_hydro = 0;
	unsigned int max_pI = 0;
	unsigned int min_bas = 999;
	unsigned int min_heli = 999;
	unsigned int min_hydro = 999;
	unsigned int min_pI = 999;

	mz = 0.;
	for (i=0; i < peplen; i++) {
		mz += amino_F[modpeptide[i+1]];
		total_bas += bas[peptide[i+1]];
		total_heli += heli[peptide[i+1]];
		total_hydro += hydro[peptide[i+1]];
		total_pI += pI[peptide[i+1]];
		if (max_bas < bas[peptide[i+1]]) {
			max_bas = bas[peptide[i+1]];
		}
		if (max_heli < heli[peptide[i+1]]) {
			max_heli = heli[peptide[i+1]];
		}
		if (max_hydro < hydro[peptide[i+1]]) {
			max_hydro = hydro[peptide[i+1]];
		}
		if (max_pI < pI[peptide[i+1]]) {
			max_pI = pI[peptide[i+1]];
		}
		if (min_bas > bas[peptide[i+1]]) {
			min_bas = bas[peptide[i+1]];
		}
		if (min_heli > heli[peptide[i+1]]) {
			min_heli = heli[peptide[i+1]];
		}
		if (min_hydro > hydro[peptide[i+1]]) {
			min_hydro = hydro[peptide[i+1]];
		}
		if (min_pI > pI[peptide[i+1]]) {
			min_pI = pI[peptide[i+1]];
		}
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
		max_bas_b = 0;
		max_heli_b = 0;
		max_hydro_b = 0;
		max_pI_b = 0;
		max_bas_y = 0;
		max_heli_y = 0;
		max_hydro_y = 0;
		max_pI_y = 0;
		min_bas_b = 999;
		min_heli_b = 999;
		min_hydro_b = 999;
		min_pI_b = 999;
		min_bas_y = 999;
		min_heli_y = 999;
		min_hydro_y = 999;
		min_pI_y = 999;
		
		buf2[peptide[i+1]]++;
		sum_aG += aG[peptide[i+1]];
		sum_wikiG += wikiG[peptide[i+1]];
		for (j=0; j < 19; j++) {
			v[fnum++] = (int) 100*(((float) buf2[j])/(i+1));
		}
		v[fnum++] = sum_aG;
		v[fnum++] = sum_wikiG;
		v[fnum++] = (int) 100*(((float) sum_aG)/(i+1));
		v[fnum++] = (int) 100*(((float) sum_wikiG)/(i+1));


		buf3[peptide[i+1]]--;
		for (j=0; j < 19; j++) {
			v[fnum++] = (int) 100*(((float) buf3[j])/(peplen-i-1));
		}
		v[fnum++] = sum_aG_tot-sum_aG;
		v[fnum++] = sum_wikiG_tot-sum_wikiG;
		v[fnum++] = (int) 100*(((float) (sum_aG_tot-sum_aG)/(peplen-i-1)));
		v[fnum++] = (int) 100*(((float) (sum_wikiG_tot-sum_wikiG)/(peplen-i-1)));

		v[fnum++] = (int) mz;
		v[fnum++] = peplen;
		v[fnum++] = i;
		v[fnum++] = (int) 100*(float)i/peplen;
		v[fnum++] = sum_aG_tot;
		v[fnum++] = sum_wikiG_tot;
		v[fnum++] = total_bas;
		v[fnum++] = total_heli;
		v[fnum++] = total_hydro;
		v[fnum++] = total_pI;
		v[fnum++] = mean_mz;
		v[fnum++] = mean_bas;
		v[fnum++] = mean_heli;
		v[fnum++] = mean_hydro;
		v[fnum++] = mean_pI;
		v[fnum++] = max_bas;
		v[fnum++] = max_heli;
		v[fnum++] = max_hydro;
		v[fnum++] = max_pI;
		v[fnum++] = min_bas;
		v[fnum++] = min_heli;
		v[fnum++] = min_hydro;
		v[fnum++] = min_pI;

		for (j=0; j<=i; j++) {
			if (bas[peptide[j+1]] > max_bas_b) {
				max_bas_b = bas[peptide[j+1]];
			}
			if (heli[peptide[j+1]] > max_heli_b) {
				max_heli_b = heli[peptide[j+1]];
			}
			if (hydro[peptide[j+1]] > max_hydro_b) {
				max_hydro_b = hydro[peptide[j+1]];
			}
			if (pI[peptide[j+1]] > max_pI_b) {
				max_pI_b = pI[peptide[j+1]];
			}
			if (bas[peptide[j+1]] < min_bas_b) {
				min_bas_b = bas[peptide[j+1]];
			}
			if (heli[peptide[j+1]] < min_heli_b) {
				min_heli_b = heli[peptide[j+1]];
			}
			if (hydro[peptide[j+1]] < min_hydro_b) {
				min_hydro_b = hydro[peptide[j+1]];
			}
			if (pI[peptide[j+1]] < min_pI_b) {
				min_pI_b = pI[peptide[j+1]];
			}
		}
		for (j=i+1; j<peplen; j++) {
			if (bas[peptide[j+1]] > max_bas_y) {
				max_bas_y = bas[peptide[j+1]];
			}
			if (heli[peptide[j+1]] > max_heli_y) {
				max_heli_y = heli[peptide[j+1]];
			}
			if (hydro[peptide[j+1]] > max_hydro_y) {
				max_hydro_y = hydro[peptide[j+1]];
			}
			if (pI[peptide[j+1]] > max_pI_y) {
				max_pI_y = pI[peptide[j+1]];
			}
			if (bas[peptide[j+1]] < min_bas_y) {
				min_bas_y = bas[peptide[j+1]];
			}
			if (heli[peptide[j+1]] < min_heli_y) {
				min_heli_y = heli[peptide[j+1]];
			}
			if (hydro[peptide[j+1]] < min_hydro_y) {
				min_hydro_y = hydro[peptide[j+1]];
			}
			if (pI[peptide[j+1]] < min_pI_y) {
				min_pI_y = pI[peptide[j+1]];
			}
		}

		v[fnum++] = max_bas_b;
		v[fnum++] = max_heli_b;
		v[fnum++] = max_hydro_b;
		v[fnum++] = max_pI_b;
		v[fnum++] = min_bas_b;
		v[fnum++] = min_heli_b;
		v[fnum++] = min_hydro_b;
		v[fnum++] = min_pI_b;

		v[fnum++] = max_bas_y;
		v[fnum++] = max_heli_y;
		v[fnum++] = max_hydro_y;
		v[fnum++] = max_pI_y;
		v[fnum++] = min_bas_y;
		v[fnum++] = min_heli_y;
		v[fnum++] = min_hydro_y;
		v[fnum++] = min_pI_y;

		mzb += amino_F[modpeptide[i+1]];
		v[fnum++] = (int) mzb;
		v[fnum++] = (int) (mz - mzb);
		v[fnum++] = (int) (mzb/(i+1));
		v[fnum++] = (int) ((mz-mzb)/(peplen-1-i));
		sum_bas += bas[peptide[i+1]];
		v[fnum++] = sum_bas;
		v[fnum++] = total_bas-sum_bas;
		v[fnum] = v[fnum-1] - v[fnum-2] + 100000;
		fnum++;
		//v[fnum++] = (int) ((float)sum_bas/(i+1));
		//v[fnum++] = (int) ((float)(total_bas-sum_bas)/(peplen-1-i));
		sum_heli += heli[peptide[i+1]];
		v[fnum++] = sum_heli;
		v[fnum++] = total_heli-sum_heli;
		v[fnum] = v[fnum-1] - v[fnum-2] + 100000;
		fnum++;
		//v[fnum++] = (int) ((float)sum_heli/(i+1));
		//v[fnum++] = (int) ((float)(total_heli-sum_heli)/(peplen-1-i));
		sum_hydro += hydro[peptide[i+1]];
		v[fnum++] = sum_hydro;
		v[fnum++] = total_hydro-sum_hydro;
		v[fnum] = v[fnum-1] - v[fnum-2] + 100000;
		fnum++;
		//v[fnum++] = (int) ((float)sum_hydro/(i+1));
		//v[fnum++] = (int) ((float)(total_hydro-sum_hydro)/(peplen-1-i));
		sum_pI += pI[peptide[i+1]];
		v[fnum++] = sum_pI;
		v[fnum++] = total_pI-sum_pI;
		v[fnum] = v[fnum-1] - v[fnum-2] + 100000;
		fnum++;
		//v[fnum++] = (int) ((float)sum_pI/(i+1));
		//v[fnum++] = (int) ((float)(total_pI-sum_pI)/(peplen-1-i));

		v[fnum++] = bas[peptide[i+1]]+bas[peptide[i+2]];
		v[fnum++] = heli[peptide[i+1]]+heli[peptide[i+2]];
		v[fnum++] = hydro[peptide[i+1]]+hydro[peptide[i+2]];
		v[fnum++] = pI[peptide[i+1]]+pI[peptide[i+2]];
		v[fnum++] = bas[peptide[i+1]]*bas[peptide[i+2]];
		v[fnum++] = heli[peptide[i+1]]*heli[peptide[i+2]];
		v[fnum++] = hydro[peptide[i+1]]*hydro[peptide[i+2]];
		v[fnum++] = pI[peptide[i+1]]*pI[peptide[i+2]];

		v[fnum++] = bas[peptide[i+1]]-bas[peptide[i+2]]+1000;
		v[fnum++] = heli[peptide[i+1]]-heli[peptide[i+2]]+1000;
		v[fnum++] = hydro[peptide[i+1]]-hydro[peptide[i+2]]+1000;
		v[fnum++] = pI[peptide[i+1]]-pI[peptide[i+2]]+1000;
		v[fnum++] = bas[peptide[i+2]]-bas[peptide[i+1]]+1000;
		v[fnum++] = heli[peptide[i+2]]-heli[peptide[i+1]]+1000;
		v[fnum++] = hydro[peptide[i+2]]-hydro[peptide[i+1]]+1000;
		v[fnum++] = pI[peptide[i+2]]-pI[peptide[i+1]]+1000;

		v[fnum++] = bas[peptide[i+1]]+bas[peptide[1]];
		v[fnum++] = heli[peptide[i+1]]+heli[peptide[1]];
		v[fnum++] = hydro[peptide[i+1]]+hydro[peptide[1]];
		v[fnum++] = pI[peptide[i+1]]+pI[peptide[1]];

		v[fnum++] = bas[peptide[peplen]]+bas[peptide[i+2]];
		v[fnum++] = heli[peptide[peplen]]+heli[peptide[i+2]];
		v[fnum++] = hydro[peptide[peplen]]+hydro[peptide[i+2]];
		v[fnum++] = pI[peptide[peplen]]+pI[peptide[i+2]];

		int pos = 1;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];
		v[fnum++] = wikiG[peptide[pos]];
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

		pos = 2;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];
		v[fnum++] = wikiG[peptide[pos]];
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
		v[fnum++] = wikiG[peptide[pos]];
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

		pos = peplen;
		v[fnum++] = amino_F[modpeptide[pos]];
		v[fnum++] = bas[peptide[pos]];
		v[fnum++] = heli[peptide[pos]];
		v[fnum++] = hydro[peptide[pos]];
		v[fnum++] = pI[peptide[pos]];
		v[fnum++] = wikiG[peptide[pos]];
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

		v[fnum++] = wikiG[peptide[i+1]];
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

		v[fnum++] = wikiG[peptide[i+2]];
		v[fnum] = 0;
		if (peptide[i+2] == 11) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+2] == 2) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+2] == 3) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+2] == 8) {
			v[fnum] = 1;
		}
		fnum++;
		v[fnum] = 0;
		if (peptide[i+2] == 13) {
			v[fnum] = 1;
		}
		fnum++;

		v[fnum++] = bas[peptide[i+1]];
		if (i==0) {
			v[fnum++] = bas[peptide[i+1]];
		}
		else {
			v[fnum++] = bas[peptide[i]];
		}
		v[fnum++] = bas[peptide[i+2]];
		if (i==(peplen-2)) {
			v[fnum++] = bas[peptide[i+2]];
		}
		else {
			v[fnum++] = bas[peptide[i+3]];
		}

		v[fnum++] = heli[peptide[i+1]];
		if (i==0) {
			v[fnum++] = heli[peptide[i+1]];
		}
		else {
			v[fnum++] = heli[peptide[i]];
		}
		v[fnum++] = heli[peptide[i+2]];
		if (i==(peplen-2)) {
			v[fnum++] = heli[peptide[i+2]];
		}
		else {
			v[fnum++] = heli[peptide[i+3]];
		}

		v[fnum++] = hydro[peptide[i+1]];
		if (i==0) {
			v[fnum++] = hydro[peptide[i+1]];
		}
		else {
			v[fnum++] = hydro[peptide[i]];
		}
		v[fnum++] = hydro[peptide[i+2]];
		if (i==(peplen-2)) {
			v[fnum++] = hydro[peptide[i+2]];
		}
		else {
			v[fnum++] = hydro[peptide[i+3]];
		}

		v[fnum++] = pI[peptide[i+1]];
		if (i==0) {
			v[fnum++] = pI[peptide[i+1]];
		}
		else {
			v[fnum++] = pI[peptide[i]];
		}
		v[fnum++] = pI[peptide[i+2]];
		if (i==(peplen-2)) {
			v[fnum++] = pI[peptide[i+2]];
		}
		else {
			v[fnum++] = pI[peptide[i+3]];
		}

		v[fnum++] = amino_F[modpeptide[i+1]];
		if (i==0) {
			v[fnum++] = amino_F[modpeptide[i+1]];
		}
		else {
			v[fnum++] = amino_F[modpeptide[i]];
		}
		v[fnum++] = amino_F[modpeptide[i+2]];
		if (i==(peplen-2)) {
			v[fnum++] = amino_F[modpeptide[i+2]];
		}
		else {
			v[fnum++] = amino_F[modpeptide[i+3]];
		}

		v[fnum++] = charge;

	}
	v[0] = fnum-1;
	return v;
}
