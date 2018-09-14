#!/usr/bin/env python
import sys
import pandas as pd
#import numpy as np
#import cPickle as pickle


def return_len(x):
    return len(x)


def is_mod(x):
    if "|" in str(x):
        tmp = x.split("|")
        ok = True
        for i in range(1, len(tmp), 2):
            if (tmp[i] != "CAM") & (tmp[i] != "Oxidation"):
                ok = False
        if ok:
            return 0
        else:
            return 1
    return 0


def return_charge(x):
    return tmap[x]


def is_tryp(x):
    if (x[-1] == 'R') | (x[-1] == 'K'):
        return 1
    return 0


def add_pepmass(x):
    return tmap[x]


def add_cpepmass(x):
    return 18.010601+sum([AminoMass[a] for a in x[2]])+x[6]*57.02146


def num_CAM(x):
    n = 0
    if "|" in str(x):
        tmp = x.split("|")
        # ok = True
        for i in range(1, len(tmp), 2):
            if tmp[i] == "CAM":
                n += 1
    return n


def isA(x):
    return x[0]


tmap = {}

AminoMass = {
    'A': 71.037114, 'C': 103.009185, 'D': 115.026943, 'E': 129.042593, 'F': 147.068414, 'G': 57.021464,
    'H': 137.058912, 'I': 113.084064, 'K': 128.094963, 'L': 113.084064, 'M': 131.040485, 'N': 114.042927,
    'P': 97.052764, 'Q': 128.058578, 'R': 156.101111, 'S': 87.032028, 'T': 101.047679, 'V': 99.068414,
    'W': 186.079313, 'Y': 163.063329,
}

f = open(sys.argv[2])
title = ""
# pepmass = 0
charge = 0
while True:
    rows = f.readlines(1000000)
    if not rows:
        break
    for row in rows:
        row = row.rstrip()
        if row == "":
            continue
        if row[:5] == "TITLE":
            title = row[6:]
        elif row[:6] == "CHARGE":
            charge = int(row[7:9].replace("+", ""))
        # elif row[:7] == "PEPMASS":
            # pepmass = float(row[8:].split()[0])
        elif row[:8] == "END IONS":
            # tmap[title] = (float(pepmass) * (charge)) - ((charge)*1.007825035)
            tmap[title] = charge

# pickle.dump( tmap, open( "tmap.p", "wb" ) )
# tmap = pickle.load( open( "tmap.p", "rb" ) )

sys.stderr.write("done\n")

sys.stderr.write('reading file\n')
data = pd.read_csv(sys.argv[1], sep=' ')
data['peplen'] = data['peptide'].apply(return_len)
data['mod'] = data['modifications'].apply(is_mod)
data['charge'] = data['spec_id'].apply(return_charge)

# data['numcam'] = data['modifications'].apply(num_CAM)
# data['pepmass'] = data['spec_id'].apply(add_pepmass)
# data['cpepmass'] = data.apply(add_cpepmass,axis=1)
# data['diff'] = np.abs(data['pepmass']-data['cpepmass'])

data = data[data.peplen >= 8]
data = data[data['mod'] == 0]

datatmp = data.drop_duplicates(subset=['peptide', 'charge'], keep="last")
datatmp[['spec_id', 'modifications', 'peptide', 'charge']].to_csv(sys.argv[1] + ".ms2pip", index=False, sep=" ")
