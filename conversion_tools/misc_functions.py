"""
Miscellaneous functions regarding MS2PIP file conversions.
"""


import re
import pandas as pd


def add_fixed_mods(peprec, fixed_mods=None, n_term=None, c_term=None):
    """
    Add 'fixed' modifications to all peptides in an MS2PIP PEPREC file.
    Return list with MS2PIP modifications with fixed mods added.

    Positional arguments:
    peprec - MS2PIP PEPREC DataFrame

    Keyword arguments:
    fixed_mods - List of tuples. First tuple element is amino acid, second tuple
    element is modification name. E.g. `[('K', 'TMT6plex')]`
    n_term - Name of fixed N-terminal modification to add
    c_term - Name of fixed C-terminal modification to add
    """

    if not fixed_mods:
        fixed_mods = []

    result = []

    for _, row in peprec.iterrows():
        mods = row['modifications']
        if mods == '-':
            mods = []
        else:
            mods = mods.split('|')

        current_mods = list(zip([int(i) for i in mods[::2]], mods[1::2]))

        for aa, mod in fixed_mods:
            current_mods.extend([(m.start()+1, mod) for m in re.finditer(aa, row['peptide'])])

        if n_term and not 0 in [i for i, n in current_mods]:
            current_mods.append((0, n_term))
        if c_term and not -1 in [i for i, n in current_mods]:
            current_mods.append((-1, c_term))

        current_mods = sorted(current_mods, key=lambda x: x[0])
        current_mods = '|'.join(['|'.join(m) for m in [(str(i), n) for i, n in current_mods]])
        result.append(current_mods)

    return result


def peprec_add_charges(peprec_filename, mgf_filename, overwrite=False):
    """
    Get precursor charges from MGF file and add them to a PEPREC
    """
    peprec = pd.read_csv(peprec_filename, sep=' ', index_col=None)

    if not overwrite and 'charge' in peprec.columns:
        print('Charges already in PEPREC')
        return None

    spec_count = 0
    charges = {}
    with open(mgf_filename, 'rt') as f:
        for line in f:
            if line.startswith('TITLE='):
                title = line[6:].strip()
                spec_count += 1
            if line.startswith('CHARGE='):
                charge = line[7:].strip()
                charges[title] = charge

    if not spec_count == len(charges):
        print('Something went wrong')
        return None

    peprec['charge'] = peprec['spec_id'].map(charges)
    
    new_peprec_filename = re.sub('\.peprec$|\.PEPREC$', '', peprec_filename) + '_withcharges.peprec'
    peprec.to_csv(new_peprec_filename, sep=' ', index=False)

    print('PEPREC with charges written to ' + new_peprec_filename)
    return peprec