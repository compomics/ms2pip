// Compute feature vectors from peptide
unsigned int* get_v_ms2pip_catboost(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge)
	{
	int i,j,k;

	int fnum = 1; //first value in v is its length

	for (i=0; i < 19; i++) {
		count_n[i] = 0;
		count_c[i] = 0;
	}

	//I need this for Omega
	//important for sptms!!
	peptide_buf[0] = peptide[0];
	for (i=0; i < peplen; i++) {
		if (peptide[i+1] > 18) {
			peptide_buf[i+1] = sptm_mapper[peptide[i+1]];
		}
		else {
			peptide_buf[i+1] = peptide[i+1]; 
		}
		count_c[peptide_buf[i+1]]++;
	}
	
	int num_shared = 0;
	
	shared_features[num_shared++] = peplen;
	shared_features[num_shared++] = charge;

	shared_features[num_shared] = 0;
	if (charge == 1) {
		shared_features[num_shared] = 1;
		}
	num_shared++;
	shared_features[num_shared] = 0;
	if (charge == 2) {
		shared_features[num_shared] =1;
		}
	num_shared++;
	shared_features[num_shared] = 0;
	if (charge == 3) {
		shared_features[num_shared] =1;    
		}
	num_shared++;
	shared_features[num_shared] = 0;
	if (charge == 4) {
		shared_features[num_shared] =1;    
		}
	num_shared++;
	shared_features[num_shared] = 0;
	if (charge >= 5) {
		shared_features[num_shared]=1;    
		}
	num_shared++;

	for (j=0; j < num_props; j++) {
		for (i=0; i < peplen; i++) {
			props_buffer[i] = props[j][peptide_buf[i+1]];
		}   
		qsort(props_buffer,peplen,sizeof(unsigned int),cmpfunc);
		shared_features[num_shared++] = props_buffer[0];
		shared_features[num_shared++] = props_buffer[(int)(0.25*(peplen-1))];
		shared_features[num_shared++] = props_buffer[(int)(0.5*(peplen-1))];
		shared_features[num_shared++] = props_buffer[(int)(0.75*(peplen-1))];
		shared_features[num_shared++] = props_buffer[peplen-1];
	}

	for (i=0; i < peplen-1; i++) {
		v[fnum++] = peptide_buf[1];
		v[fnum++] = peptide_buf[peplen];
		v[fnum++] = peptide_buf[i];
		v[fnum++] = peptide_buf[i+1];
		for (j=0; j<num_shared; j++) {
			v[fnum++] = shared_features[j];
		}
		v[fnum++] = i+1;
		v[fnum++] = peplen-i;       
		count_n[peptide_buf[i+1]]++;
		count_c[peptide_buf[peplen-i]]--;

		for (j=0; j < 19; j++) {
			v[fnum++] = count_n[j];
			v[fnum++] = count_c[j];
		}
				
		for (j=0; j < num_props; j++) {
			v[fnum++] = props[j][peptide_buf[1]];  
			v[fnum++] = props[j][peptide_buf[peplen]];  
			if (i==0) {
				v[fnum++] = 0;  
			}
			else {  
				v[fnum++] = props[j][peptide_buf[i-1]];  
			}
			v[fnum++] = props[j][peptide_buf[i]];  
			v[fnum++] = props[j][peptide_buf[i+1]];  
			if (i==(peplen-1)) {
				v[fnum++] = 0;                          
			}
			else {
				v[fnum++] = props[j][peptide_buf[i+2]];                         
			}
			unsigned int s = 0;
			for (k=0; k <= i; k++) {
				props_buffer[k] = props[j][peptide_buf[k+1]];
				s+= props_buffer[k];
			}   
			v[fnum++] = s;
			qsort(props_buffer,i+1,sizeof(unsigned int),cmpfunc);
			v[fnum++] = props_buffer[0];
			v[fnum++] = props_buffer[(int)(0.25*i)];
			v[fnum++] = props_buffer[(int)(0.5*i)];
			v[fnum++] = props_buffer[(int)(0.75*i)];
			v[fnum++] = props_buffer[i];
			s = 0;
			for (k=i+1; k < peplen; k++) {
				props_buffer[k-i-1] = props[j][peptide_buf[k+1]];
				s+= props_buffer[k-i-1];
			}   
			v[fnum++] = s;
			qsort(props_buffer,peplen-i-1,sizeof(unsigned int),cmpfunc);
			v[fnum++] = props_buffer[0];
			v[fnum++] = props_buffer[(int)(0.25*(peplen-i-1))];
			v[fnum++] = props_buffer[(int)(0.5*(peplen-i-1))];
			v[fnum++] = props_buffer[(int)(0.75*(peplen-i-1))];
			v[fnum++] = props_buffer[peplen-i-2];
		}
	}
	v[0] = fnum-1;
	return v;
}
