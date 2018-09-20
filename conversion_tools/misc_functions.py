"""
Miscellaneous functions regarding MS2PIP file conversions.
"""


import re


def add_fixed_mods(peprec, fixed_mods=[], n_term=None, c_term=None):
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
