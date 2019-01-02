"""
Convert mzTab to MS2PIP PEPREC spectral library (only unique peptides)
"""

import argparse
import pandas as pd


def argument_parser():
    parser = argparse.ArgumentParser(
        description='Convert mzTab to MS2PIP PEPREC spectral library (only\
        unique peptides)'
    )
    parser.add_argument(
        'filename', metavar="<mztab filename>",
        help='filename of mzTab to convert.'
    )
    args = parser.parse_args()
    return args


def get_peptide_table_start(filename):
    """
    Get line number of mzTab peptide table start.
    """
    with open(filename, 'rt') as f:
        for num, line in enumerate(f, 1):
            if line[:3] == 'PSH':
                return num


def parse_mods(modifications, psimod_mapping=None):
    """
    Convert mzTab modifications to MS2PIP PEPREC modifications.

    N-term and C-term modifications are not supported!
    Entries without modifications should be given as '-'

    modifications - Pandas Series containing mzTab-formatted modifications
    psimod_mapping - dictionary that maps PSI-MOD entries to desired modification names
    """
    if not psimod_mapping:
        psimod_mapping = psimod_mapping = {
            'MOD:00394': 'Acetyl',
            'MOD:00400': 'Deamidated',
            'MOD:00425': 'Oxidation',
            'MOD:01214': 'Carbamidomethyl'
        }

    parsed_mods = []
    for mods in modifications.fillna('-').str.split(',|-'):
        if mods == ['', '']:
            parsed_mods.append('-')
        else:
            pos = [str(int(p) + 1) for p in mods[::2]]
            names = [psimod_mapping[n] for n in mods[1::2]]
            parsed_mods.append('|'.join(['|'.join(i) for i in list(zip(pos, names))]))

    return parsed_mods


def main():
    args = argument_parser()

    start = get_peptide_table_start(args.filename)
    mztab = pd.read_csv(args.filename, sep='\t', skiprows=start-1)

    # Filter unique peptides
    unique_count = mztab.duplicated(['sequence', 'modifications', 'charge']).value_counts().to_dict()
    if False in unique_count:
        unique_count = unique_count[False]
    else:
        unique_count = 0
    print("mzTab contains {} unique peptides".format(unique_count))

    mztab.sort_values('search_engine_score[1]', ascending=False, inplace=True)
    mztab = mztab[~mztab.duplicated(['sequence', 'modifications', 'charge'])].copy()


    # Check modifications present in mzTab
    flatten = lambda l: [item for sublist in l for item in sublist]
    unique_mods = set(flatten([mods[1::2] for mods in mztab['modifications'].dropna().str.split(',|-')]))
    print("mzTab contains these modifications: {}".format(unique_mods))

    mztab['parsed_modifications'] = parse_mods(mztab['modifications'])

    peprec_cols = [
        'PSM_ID',
        'sequence',
        'parsed_modifications',
        'charge',
        'retention_time',
        'calc_mass_to_charge',
        'search_engine_score[1]',
    ]

    peprec_col_mapping = {
        'PSM_ID': 'spec_id',
        'sequence': 'peptide',
        'parsed_modifications': 'modifications',
        'charge': 'charge',
        'retention_time': 'rt',
        'calc_mass_to_charge': 'mz',
        'search_engine_score[1]': 'score'
    }

    peprec = mztab[peprec_cols]
    peprec = peprec.rename(columns=peprec_col_mapping)

    filename_stripped = '.'.join(args.filename.split('.')[:-1])
    peprec.to_csv(filename_stripped + '.peprec', sep=' ', index=False)


if __name__ == '__main__':
    main()
