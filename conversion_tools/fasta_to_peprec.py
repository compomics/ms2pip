"""
## Create PEPREC file from a FASTA file
Takes every protein in a FASTA file, generates a PEPREC file with all tryptic peptides (check global variables to set charge states, min/max lengths and number of missed cleavages)

*fasta_to_peprec.py*
**Input:** FASTA file
**Output:** PEPREC file

**Requirements:** Biopython to parse FASTA file; Pyteomics for in silico cleavage; tqdm for progress bar.

```
usage: fasta_to_peprec.py [-h] fasta_file

Generate a PEPREC file for all proteins in a fasta file

positional arguments:
  fasta_file             FASTA file with proteins of interest

optional arguments:
  -h, --help            show this help message and exit
```
"""

# --------------
# Import modules
# --------------
from Bio import SeqIO
from pyteomics.parser import cleave
from pyteomics.parser import expasy_rules
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('fasta_file', help='FASTA file with all proteins of interest')

args = parser.parse_args()

# --------------
# Global variables
# --------------
CHARGES = [1, 2, 3]
NUM_MISSED = 2
MIN_LEN = 8
MAX_LEN = 50

peprec_file = open(args.fasta_file.replace('.fasta', '.PEPREC'), 'wt')
peprec_file.write('spec_id modifications peptide charge protein\n')

n_prots = (sum(1 for x in SeqIO.parse(open(args.fasta_file),'fasta')))
print('{} proteins in fasta file \n'.format(n_prots))

fasta_sequences = SeqIO.parse(open(args.fasta_file),'fasta')
aa = set(["A", "C", "D", "E", "F", "G", "H", "L", "I", "K", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"])

i = 0 # identifier for each (peptide, charge)

for fasta in tqdm(fasta_sequences, total=n_prots):
    prot, seq = fasta.id, str(fasta.seq)
    # seq = seq.replace('L','I')
    RKcut = cleave(seq, '[KR]', NUM_MISSED, MIN_LEN)
    # RKcut = cleave(seq, expasy_rules['trypsin'], NUM_MISSED, MIN_LEN)
    for k in RKcut:
        if (set(k) <= aa) & (len(k) <= MAX_LEN): # to exclude peptides with aa which aren't the standard
            for charge in CHARGES:
                i += 1
                to_write = '{} {} {} {} {}\n'.format(i, '', k, charge, prot)
                peprec_file.write(to_write)
        else: continue

peprec_file.close()
