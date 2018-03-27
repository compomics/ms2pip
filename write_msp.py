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
from operator import itemgetter

# Other libraries
import numpy as np
import pandas as pd


def process(spec_ids_sel, all_preds, peprec, add_protein, q):
    preds_col_names = list(all_preds.columns)
    preds_to_slice = {}
    preds_list = all_preds.values.tolist()

    preds_spec_id_index = preds_col_names.index('spec_id')
    mz_index = preds_col_names.index('mz')
    prediction_index = preds_col_names.index('prediction')
    ion_index = preds_col_names.index('ion')
    ionnumber_index = preds_col_names.index('ionnumber')

    for row in preds_list:
        spec_id = row[preds_spec_id_index]
        if spec_id in preds_to_slice.keys():
            preds_to_slice[spec_id].append(row)
        else:
            preds_to_slice[spec_id] = [row]

    peprec_col_names = list(peprec.columns)
    peprec_to_slice = {}
    peprec_list = peprec.values.tolist()

    spec_id_index = peprec_col_names.index('spec_id')
    peptide_index = peprec_col_names.index('peptide')
    charge_index = peprec_col_names.index('charge')
    modifications_index = peprec_col_names.index('modifications')
    protein_list_index = peprec_col_names.index('protein_list')

    for row in peprec_list:
        peprec_to_slice[row[spec_id_index]] = row

    for spec_id in spec_ids_sel:
        out = []
        preds = preds_to_slice[spec_id]
        peprec_sel = peprec_to_slice[spec_id]

        preds = sorted(preds, key=itemgetter(mz_index))

        sequence = peprec_sel[peptide_index]
        charge = peprec_sel[charge_index]
        mods = peprec_sel[modifications_index]
        numpeaks = len(preds)

        # Calculate mass from fragment ions
        mass_b = [row[mz_index] for row in preds if row[ion_index] == 'b' and row[ionnumber_index] == 1][0]
        mass_y = [row[mz_index] for row in preds if row[ion_index] == 'y' and row[ionnumber_index] == numpeaks / 2][0]
        pepmass = mass_b + mass_y - 2 * 1.007236

        out.append('Name: {}/{}\n'.format(sequence, charge))
        out.append('MW: {}\n'.format(pepmass))
        out.append('Comment: ')

        if mods == '-':
            out.append("Mods=0 ")
        else:
            mods = mods.split('|')
            mods = [(int(mods[i]), mods[i + 1]) for i in range(0, len(mods), 2)]
            # Turn MS2PIP mod indexes into actual list indexes (eg 0 for first AA)
            mods = [(x, y) if x == 0 else (x - 1, y) for (x, y) in mods]
            mods = [(str(x), sequence[x], y) for (x, y) in mods]
            out.append("Mods={}/{} ".format(len(mods), '/'.join([','.join(list(x)) for x in mods])))

        out.append("Parent={} ".format(pepmass + charge * 1.007236) / charge)

        if add_protein:
            try:
                out.append('Protein="{}" '.format('/'.join(literal_eval(peprec_sel[protein_list_index]))))
            except ValueError:
                out.append('Protein="{}" '.format(peprec_sel[protein_list_index]))

        out.append('MS2PIP_ID="{}"'.format(spec_id))

        out.append('\nNum peaks: {}\n'.format(numpeaks))

        lines = list(zip(
            [row[mz_index] for row in preds],
            [row[prediction_index] for row in preds],
            [row[ion_index] for row in preds],
            [row[ionnumber_index] for row in preds]
        ))
        out.append(''.join(['{:.4f}\t{}\t"{}{}"\n'.format(*l) for l in lines]))

        out_string = "".join(out)
        q.put(out_string)


def writer(output_filename, write_mode, q):
    f = open("{}_predictions.msp".format(output_filename), write_mode)
    while 1:
        m = q.get()
        if m == 'kill':
            break
        else:
            f.write(str(m))
            f.flush()
    f.close()


def write_msp(all_preds, peprec, output_filename, write_mode='w', num_cpu=8):
    all_preds.reset_index(drop=True, inplace=True)
    # If not already normalized, normalize spectra
    if not (all_preds['prediction'].min() == 0 and all_preds['prediction'].max() == 10000):
        all_preds['prediction'] = ((2**all_preds['prediction']) - 0.001).clip(lower=0)
        all_preds['prediction'] = all_preds.groupby(['spec_id'])['prediction'].apply(lambda x: (x / x.max()) * 10000)
        all_preds['prediction'] = all_preds['prediction'].astype(int)

    # Check if protein list is present in peprec
    add_protein = 'protein_list' in peprec.columns

    # Use manager for advanced multiprocessing
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_cpu)
    watcher = pool.apply_async(writer, (output_filename, write_mode, q,))

    # Split titles (according to MS2PIPc)
    spec_ids = peprec['spec_id'].tolist()
    split_spec_ids = [spec_ids[i * len(spec_ids) // num_cpu: (i + 1) * len(spec_ids) // num_cpu] for i in range(num_cpu)]

    # Fire off workers
    jobs = []
    for spec_ids_sel in split_spec_ids:
        job = pool.apply_async(process, (spec_ids_sel, all_preds, peprec, add_protein, q))
        jobs.append(job)

    # Collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    # Now we are done, kill the listener
    q.put('kill')
    pool.close()
