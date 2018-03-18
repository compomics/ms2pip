"""
Write MSP Spectrum Library from MS2PIP predictions.
Run by calling function `write_msp`.
"""


__author__ = "Ralf Gabriels"
__copyright__ = "Copyright 2018"
__credits__ = ["Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__email__ = "Ralf.Gabriels@ugent.be"


# Native libraries
from ast import literal_eval
import multiprocessing as mp
from multiprocessing import Pool

# Other libraries
import numpy as np
import pandas as pd


def process(spec_id, all_preds, peprec, add_protein, q):
    out = ''

    preds = all_preds[all_preds['spec_id'] == spec_id]
    preds = preds.sort_values('mz')
    preds = preds.reset_index(drop=True)

    tmp = peprec[peprec['spec_id'] == spec_id]
    sequence = tmp['peptide'].iloc[0]
    charge = tmp['charge'].iloc[0]
    mods = tmp['modifications'].iloc[0]
    numpeaks = len(preds)

    # Calculate mass from fragment ions
    mass_b = preds[(preds['ion'] == 'b') & (preds['ionnumber'] == 1)]['mz'].iloc[0]
    mass_y = preds[(preds['ion'] == 'y') & (preds['ionnumber'] == numpeaks / 2)]['mz'].iloc[0]
    pepmass = mass_b + mass_y - 2 * 1.007236

    out += 'Name: {}/{}\n'.format(sequence, charge)
    out += 'MW: {}\n'.format(pepmass)
    out += 'Comment: '

    if mods == '-':
        out += "Mods=0 "
    else:
        mods = mods.split('|')
        mods = [(int(mods[i]), mods[i + 1]) for i in range(0, len(mods), 2)]
        # Turn MS2PIP mod indexes into actual list indexes (eg 0 for first AA)
        mods = [(x, y) if x == 0 else (x - 1, y) for (x, y) in mods]
        mods = [(str(x), sequence[x], y) for (x, y) in mods]
        out += "Mods={}/{} ".format(len(mods), '/'.join([','.join(list(x)) for x in mods]))

    out += "Parent={} ".format(pepmass / charge)

    if add_protein:
        out += 'Protein="{}" '.format('/'.join(literal_eval(tmp['protein_list'].iloc[0])))

    out += 'MS2PIP_ID="{}"'.format(spec_id)

    out += '\nNum peaks: {}\n'.format(numpeaks)
    lines = list(zip(preds['mz'], preds['prediction'], preds['ion'], preds['ionnumber']))
    out += ''.join(['{:.4f}\t{}\t"{}{}"\n'.format(*l) for l in lines])

    out += '\n'

    q.put(out)
    return(out)


def writer(output_filename, q):
    f = open("{}_predictions.msp".format(output_filename), 'w')
    while 1:
        m = q.get()
        if m == 'kill':
            break
        else:
            f.write(str(m))
            f.flush()
    f.close()


def write_msp(all_preds, peprec, output_filename, num_cpu=8):
    # Normalize spectra:
    all_preds.reset_index(drop=True, inplace=True)
    all_preds['prediction'] = ((2**all_preds['prediction']) - 0.001).clip(lower=0)
    all_preds['prediction'] = all_preds.groupby(['spec_id'])['prediction'].apply(lambda x: (x / x.max()) * 10000)
    all_preds['prediction'] = all_preds['prediction'].astype(int)

    # Check if protein list is present in peprec
    add_protein = 'protein_list' in peprec.columns

    # Use manager for advanced multiprocessing
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_cpu)
    watcher = pool.apply_async(writer, (output_filename, q,))

    # Fire off workers
    jobs = []
    for spec_id in peprec['spec_id']:
        job = pool.apply_async(process, (spec_id, all_preds, peprec, add_protein, q))
        jobs.append(job)

    # Collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # Now we are done, kill the listener
    q.put('kill')
    pool.close()
