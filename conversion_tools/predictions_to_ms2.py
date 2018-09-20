"""
## Create MS2+SSL files (SkyLine spectral library formats) from MS2PIP predictions
Takes spectra predicted by MS2PIP and write SkyLine spectral library format MS2+SSL

*predictions_to_ms2.py*
**Input:** MS2PIP predictions file
**Output:** MS2 and SSL files

**Requirements:** Pyteomics for mass calculations; tqdm for progress bar.

```
usage: predictions_to_ms2.py [-h] pep_file

Generate MS2 and SSL files from MS2PIP predictions

positional arguments:
  pep_file             PEPREC file used to generate predictions

optional arguments:
  -h, --help            show this help message and exit
```
"""

# --------------
# Import modules
# --------------
import sys
import argparse
from time import localtime, strftime
from operator import itemgetter
from pyteomics import mass
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("pep_file", help="PEPREC file used to get the ms2pip predictions")
args = parser.parse_args()

pep_dict = {}
# pep_dic[spec_id] = [sequence, charge]
pep = open(args.pep_file)
for line in pep:
    split_line = line.split(' ')
    # first line must be the header, and first header must be spec_id:
    if line.startswith('spec_id'):
        seq_id = split_line.index('peptide')
        ch_id =  split_line.index('charge')
        continue
    pep_dict[split_line[0]] = [split_line[2], int(split_line[3])]

pred_dict = {}
# pred_dict[spec_id] = [[mz_list], [int_list]]
pred = open(args.pep_file.replace(".PEPREC", "_predictions.csv"))
for line in pred:
    if line.startswith('spec_id'): continue
    split_line = line.strip().split(',')
    if split_line[0] in pred_dict.keys():
        pred_dict[split_line[0]][0].append(float(split_line[5]))
        pred_dict[split_line[0]][1].append(float(split_line[6]))
    else:
        pred_dict[split_line[0]] = [[float(split_line[5])], [float(split_line[6])]]

ssl_output = open(args.pep_file + ".ssl", "w+")
ssl_output.write('file\tscan\tcharge\tsequence\n')

sys.stdout.write("writing {} file\n".format(args.pep_file.replace(".PEPREC", ".ssl")))

peptides = pred_dict.keys()

ms2_name = args.pep_file.replace(".PEPREC",".ms2")
sys.stdout.write("writing {} file\n".format(ms2_name))
ms2_output = open(ms2_name, "w+")
ms2_output.write("H\tCreationDate\t{}\n".format(strftime("%Y-%m-%d %H:%M:%S", localtime())))
ms2_output.write("H\tExtractor\tMS2PIP Predictions\n")

for i, peptide in tqdm(enumerate(pep_dict.keys()), total=len(pep_dict.keys())):
    intensities = list(map(lambda x: (2**x)+0.001, pred_dict[peptide][1]))
    spectrum = zip(pred_dict[peptide][0], intensities)
    spectrum = sorted(spectrum, key=itemgetter(0))

    seq = pep_dict[peptide][0]
    prec_mass = mass.calculate_mass(sequence=seq)
    charge = pep_dict[peptide][1]

    ssl_output.write("{}\t{}\t{}\t{}\n".format(ms2_name, peptide, charge, seq))
    ms2_output.write("S\t{}\t{}\n".format(peptide, prec_mass))
    ms2_output.write("Z\t{}\t{}\n".format(int(charge), int(charge)*prec_mass))
    ms2_output.write("D\tseq\t{}\n".format(seq))

    for mz, inte in spectrum:
        ms2_output.write("{}\t{}\n".format(mz, inte))

ms2_output.close()

ssl_output.close()
