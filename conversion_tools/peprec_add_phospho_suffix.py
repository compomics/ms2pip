"""
## PEPREC: Add Phospho Suffix
Adds amino acid suffix to "Phospho" modifications in PEPREC file. "Phospho" becomes,
for instance, "PhosphoY". Also, for unmodified peptides, a hyphen is added to the
PEPREC file.

*peprec_add_phospho_suffix.py*
**Input:** Folder with PEPREC files
**Output:** PEPREC files with amino acid suffix added to "Phospho" modifications
usage: peprec_add_phospho_suffix.py [-h] [-f PEPREC_FOLDER] [-r]

Add amino acid suffix to "Phospho" in modifications column in PEPREC file(s).

optional arguments:
  -h, --help        show this help message and exit
  -f PEPREC_FOLDER  Folder with input PEPREC files (default: "")
  -r                Replace the original PEPREC files instead of writing a new
                    file (default: False)
"""

# --------------
# Import modules
# --------------
import argparse
import pandas as pd
from glob import glob


# ---------------------------------------------------------
# Find all PEPREC files in folder in a case-insensitive way
# ---------------------------------------------------------
def FindFiles(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob(''.join(map(either, pattern)))


# ------------------------------------
# Add AA suffix to 'Phospho' in PEPREC
# ------------------------------------
def AddPhosphoSuffix(peprec):
    peprec = peprec.fillna('-')
    peprec['modifications'] = [[x for x in zip([x for (i, x) in enumerate(mods) if i % 2 == 0], [x for (i, x) in enumerate(mods) if i % 2 != 0])] for mods in peprec['modifications'].str.split('|')]
    res = []
    for _, row in peprec.iterrows():
        if row['modifications']:
            res.append([(mod[0], mod[1] + row['peptide'][int(mod[0]) - 1]) if mod[1] == 'Phospho' else mod for mod in row['modifications']])
        else:
            res.append('-')
    res = ['|'.join(['|'.join(mod) for mod in row]) for row in res]
    peprec['modifications'] = res
    return(peprec)


# ---------------
# Argument parser
# ---------------
def ArgParse():
    parser = argparse.ArgumentParser(description='Add amino acid suffix to "Phospho" in modifications column in PEPREC file(s).')
    parser.add_argument('-f', dest='peprec_folder', action='store', default='',
                        help='Folder with input PEPREC files (default: "")')
    parser.add_argument('-r', dest='replace', action="store_true", default=False,
                        help='Replace the original PEPREC files instead of writing a new file (default: False)')
    args = parser.parse_args()
    return(args)


# ----
# Run!
# ----
def run():
    args = ArgParse()
    if args.peprec_folder:
        files = FindFiles("{}/*.peprec".format(args.peprec_folder))
    else:
        files = FindFiles("*.peprec")

    if not files:
        print("No PEPREC files found.")
    else:
        for peprec_file in files:
            peprec = pd.read_csv(peprec_file, sep=' ')
            peprec_out = AddPhosphoSuffix(peprec)
            if not args.replace:
                peprec_file = "{}_WithSuffix.peprec".format(peprec_file.strip('.peprec'))
            peprec_out.to_csv(peprec_file, index=None, sep=' ', na_rep='-')
        print("Ready!")


if __name__ == "__main__":
    run()
