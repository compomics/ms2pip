"""
## Split MS2PIP Spectral Library
Split MS2PIP spectral library (PEPREC and MGF file) into a train and test set.

*Split_MS2PIP_SpecLib.py*
**Input:** PEPREC and MGF file
**Output:** PEPREC and MGF files for both train and test data set.

```
usage: Split_MS2PIP_SpecLib.py [-h] [-o OUT_FILENAME] [-f TEST_FRACTION]
                               peprec_file mgf_file

Split MS2PIP spectral library (PEPREC and MGF file) into a train and test set.

positional arguments:
  peprec_file       PEPREC file input
  mgf_file          MGF file input

optional arguments:
  -h, --help        show this help message and exit
  -o OUT_FILENAME   Name for output files (default: "SpecLib")
  -f TEST_FRACTION  Fraction of input to use for test data set (default: 0.1)
```
"""

# --------------
# Import modules
# --------------
import argparse
import numpy as np
import pandas as pd


# --------------
# Split PEPREC
# --------------
def SplitPeprec(peprec, test_fraction):
    msk = np.random.rand(len(peprec)) > test_fraction
    peprec_train = peprec[msk]
    peprec_test = peprec[~msk]
    return(peprec_train, peprec_test)


# --------------
# Split MGF file
# --------------
def SplitMGF(peprec_train, peprec_test, mgf_filename, out_filename='SpecLib'):
    count_train = 0
    count_test = 0
    print("{} spectra to go...".format(len(peprec_train) + len(peprec_test)), end='')
    with open('{}_Train.mgf'.format(out_filename), 'w') as out_train:
        with open('{}_Test.mgf'.format(out_filename), 'w') as out_test:
            spec_ids_train = list(peprec_train['spec_id'])
            spec_ids_test = list(peprec_test['spec_id'])
            found_train = False
            found_test = False
            with open(mgf_filename, 'r') as mgf:
                for i, line in enumerate(mgf):
                    if 'TITLE' in line:
                        spec_id = int(line.split('TITLE=')[1])
                        if spec_id in spec_ids_train:
                            found_train = True
                            count_train += 1
                            out_train.write("BEGIN IONS\n")
                            if count_train % 100 == 0:
                                print('.', end='')
                        elif spec_id in spec_ids_test:
                            found_test = True
                            count_test += 1
                            out_test.write("BEGIN IONS\n")
                            if count_test % 100 == 0:
                                print('.', end='')
                    if 'END IONS' in line:
                        if found_train:
                            out_train.write(line + '\n')
                            found_train = False
                        elif found_test:
                            out_test.write(line + '\n')
                            found_test = False
                    if found_train:
                        out_train.write(line)
                    elif found_test:
                        out_test.write(line)

    print("\nReady! Wrote {}/{} spectra to {}_Train.mgf and {}/{} spectra to {}_Test.mgf.".format(count_train, len(peprec_train),
                                                                                                  out_filename, count_test, len(peprec_test),
                                                                                                  out_filename))


# ---------------
# Argument parser
# ---------------
def ArgParse():
    parser = argparse.ArgumentParser(description='Split MS2PIP spectral library (PEPREC and MGF file) into a train and test set.')
    parser.add_argument('peprec_file', action='store', help='PEPREC file input')
    parser.add_argument('mgf_file', action='store', help='MGF file input')
    parser.add_argument('-o', dest='out_filename', action='store', default='SpecLib',
                        help='Name for output files (default: "SpecLib")')
    parser.add_argument('-f', dest='test_fraction', action='store', default=0.1, type=float,
                        help='Fraction of input to use for test data set (default: 0.1)')
    args = parser.parse_args()
    return(args)


# ----
# Run!
# ----
def run():
    args = ArgParse()
    peprec = pd.read_csv(args.peprec_file, sep=' ')
    (peprec_train, peprec_test) = SplitPeprec(peprec, args.test_fraction)
    SplitMGF(peprec_train, peprec_test, args.mgf_file, args.out_filename)
    peprec_train.to_csv('{}_Train.peprec'.format(args.out_filename), sep=' ', na_rep='-', index=False)
    peprec_test.to_csv('{}_Test.peprec'.format(args.out_filename), sep=' ', na_rep='-', index=False)


if __name__ == "__main__":
    run()
