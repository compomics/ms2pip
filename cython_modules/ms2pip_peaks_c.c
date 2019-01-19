#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ms2pip_init_c.c"
#include "ms2pip_features_c_general.c"
#include "ms2pip_features_c_old.c"
#include "ms2pip_features_c_catboost.c"

// Import models
#include "../models/CID/model_20190107_CID_train_B.c"
#include "../models/CID/model_20190107_CID_train_Y.c"

//#include "../models/HCD/hcd_fast_B.c"
//#include "../models/HCD/hcd_fast_Y.c"
#include "../models/HCD/model_20190107_HCD_train_B.c"
#include "../models/HCD/model_20190107_HCD_train_y.c"

#include "../models/TTOF5600/model_20190107_TTOF5600_train_B.c"
#include "../models/TTOF5600/model_20190107_TTOF5600_train_y.c"

#include "../models/TMT/model_20190107_TMT_train_B.c"
#include "../models/TMT/model_20190107_TMT_train_y.c"

#include "../models/iTRAQ/model_20190107_iTRAQ_train_B.c"
#include "../models/iTRAQ/model_20190107_iTRAQ_train_y.c"

#include "../models/iTRAQphospho/model_20190107_iTRAQphospho_train_B.c"
#include "../models/iTRAQphospho/model_20190107_iTRAQphospho_train_y.c"

//#include "../models/EThcD/model_20190107_EThcD_train_B.c"
//#include "../models/EThcD/model_20190107_EThcD_train_C.c"
//#include "../models/EThcD/model_20190107_EThcD_train_Y.c"
//#include "../models/EThcD/model_20190107_EThcD_train_Z.c"


float membuffer[10000];
float ions[1000];
float predictions[1000];


//compute feature vector from peptide + predict intensities
float* get_p_ms2pip(int peplen, unsigned short* peptide, unsigned short* modpeptide, int charge, int model_id)
    {
    int i;

    // CID
    if (model_id == 0) {
        unsigned int* v = get_v_ms2pip_old(peplen, peptide, modpeptide, charge);
        int fnum = v[0]/(peplen-1);
        for (i=0; i < peplen-1; i++) {
            predictions[0*(peplen-1)+i] = score_CID_B(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)-i-1] = score_CID_Y(v+1+(i*fnum))+0.5;
        }
    }

    // HCD
    else if (model_id == 1) {
        unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
        int fnum = v[0]/(peplen-1);
        for (i=0; i < peplen-1; i++) {
            predictions[0*(peplen-1)+i] = score_HCD_B(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)-i-1] = score_HCD_Y(v+1+(i*fnum))+0.5;
        }
    }
    
    // TTOF5600
    else if (model_id == 2) {
        unsigned int* v = get_v_ms2pip_old(peplen, peptide, modpeptide, charge);
        int fnum = v[0]/(peplen-1);
        for (i=0; i < peplen-1; i++) {
            predictions[0*(peplen-1)+i] = score_TTOF5600_B(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)-i-1] = score_TTOF5600_Y(v+1+(i*fnum))+0.5;
        }
    }

    // TMT
    else if (model_id == 3) {
        unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
        int fnum = v[0]/(peplen-1);
        for (i=0; i < peplen-1; i++) {
            predictions[0*(peplen-1)+i] = score_TMT_B(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)-i-1] = score_TMT_Y(v+1+(i*fnum))+0.5;
        }
    }

    // iTRAQ
    else if (model_id == 4) {
        unsigned int* v = get_v_ms2pip_old(peplen, peptide, modpeptide, charge);
        int fnum = v[0]/(peplen-1);
        for (i=0; i < peplen-1; i++) {
            predictions[0*(peplen-1)+i] = score_iTRAQ_B(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)-i-1] = score_iTRAQ_Y(v+1+(i*fnum))+0.5;
        }
    }

    // iTRAQphospho
    else if (model_id == 5) {
        unsigned int* v = get_v_ms2pip_old(peplen, peptide, modpeptide, charge);
        int fnum = v[0]/(peplen-1);
        for (i=0; i < peplen-1; i++) {
            predictions[0*(peplen-1)+i] = score_iTRAQphospho_B(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)-i-1] = score_iTRAQphospho_Y(v+1+(i*fnum))+0.5;
        }
    }
    
    /*
    // EThcD
    else if (model_id == 6) {
        unsigned int* v = get_v_ms2pip_old(peplen, peptide, modpeptide, charge);
        int fnum = v[0]/(peplen-1);
        for (i=0; i < peplen-1; i++) {
            predictions[0*(peplen-1)+i] = score_EThcD_B(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)-i-1] = score_EThcD_Y(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)+i] = score_EThcD_C(v+1+(i*fnum))+0.5;
            predictions[4*(peplen-1)-i-1] = score_EThcD_Z(v+1+(i*fnum))+0.5;
        }
    }

    // HCDch2
    else if (model_id == 7) {
        unsigned int* v = get_v_ms2pip(peplen, peptide, modpeptide, charge);
        int fnum = v[0]/(peplen-1);
        for (i=0; i < peplen-1; i++) {
            predictions[0*(peplen-1)+i] = score_HCD_B(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)-i-1] = score_HCD_Y(v+1+(i*fnum))+0.5;
            predictions[2*(peplen-1)+i] = score_HCD_B2(v+1+(i*fnum))+0.5;
            predictions[4*(peplen-1)-i-1] = score_HCD_Y2(v+1+(i*fnum))+0.5;
        }
    }
    */

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


//get fragment ion peaks from spectrum (b, y)
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
