"""
Create a spectral library starting from a proteome in fasta format.
The script runs through the following steps:
- In silico cleavage of proteins from the fasta file
- Removes peptide redundancy
- Add all variations of variable modifications (max 7 PTMs/peptide)
- Add variations on charge state
- Run peptides through MS2PIP
- Write to MSP, MGF or HDF file
"""


__author__ = "Ralf Gabriels"
__copyright__ = "CompOmics 2018"
__credits__ = ["Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__email__ = "Ralf.Gabriels@ugent.be"


# Native libraries
import logging
import argparse
from math import ceil
from re import finditer
from datetime import datetime
from multiprocessing import Pool
from itertools import combinations

# Third party libraries
import numpy as np
import pandas as pd
from pyteomics import mass
from pyteomics.parser import cleave, expasy_rules
from Bio import SeqIO

# MS2PIP
from ms2pipC import run, write_mgf
from write_msp import write_msp


def ArgParse():
    parser = argparse.ArgumentParser(description='Create an MS2PIP PEPREC file by in silico cleaving proteins from a fasta file.')
    parser.add_argument('fasta_filename', action='store', help='Name of the fasta input file')
    parser.add_argument('-o', dest='output_filename', action='store',
                        help='Name for output file (default: derived from input file)')
    parser.add_argument('-t', dest='output_filetype', action='store', nargs='+', default=['msp'],
                        help='Output file formats for spectral library: HDF, MSP and/or MGF (default msp).')
    parser.add_argument('-c', dest='charges', action='store', nargs='+', default=[2, 3],
                        help='Precusor charges to include in peprec (default [2, 3]')
    parser.add_argument('-p', dest='min_peplen', action='store', default=8, type=int,
                        help='Minimum length of peptides to include in peprec (default: 8)')
    parser.add_argument('-w', dest='max_pepmass', action='store', default=5000, type=int,
                        help='Maximum peptide mass to include in Dalton (default: 5000)')
    parser.add_argument('-m', dest='missed_cleavages', action='store', default=2, type=int,
                        help='Number of missed cleavages to be allowed (default: 2)')
    parser.add_argument('-f', dest='frag_method', action='store', default='HCD', type=str,
                        help='Fragmentation method to use for MS2PIP predictions')
    parser.add_argument('-d', dest='decoy', action='store_true',
                        help='Create decoy spectral library by reversing peptide sequences')
    parser.add_argument('-n', dest='num_cpu', action='store', default=24, type=int,
                        help='Number of processes for multithreading (default: 24)')
    args = parser.parse_args()
    return(args)


def get_params():
    args = ArgParse()
    params = {
        'fasta_filename': args.fasta_filename,
        'charges': args.charges,
        'min_peplen': args.min_peplen,
        'max_pepmass': args.max_pepmass,
        'missed_cleavages': args.missed_cleavages,
        'frag_method': args.frag_method,
        'num_cpu': args.num_cpu,
        'modifications': [
            # (name, specific AA, mass-shift, N-term)
            ('Glu->pyro-Glu', 'E', -18.0153, True),
            ('Gln->pyro-Glu', 'Q', -17.0305, True),
            ('Acetyl', None, 42.0367, True),
            ('Oxidation', 'M', 15.9994, False),
            ('Carbamidomethyl', 'C', 57.0513, False),
        ],
        'decoy': args.decoy,
        'output_filetype': args.output_filetype,
        'batch_size': 5000,
    }

    if args.output_filename:
        params['output_filename'] = args.output_filename
    else:
        params['output_filename'] = '_'.join(params['fasta_filename'].split('\\')[-1].split('.')[:-1])

    return(params)


def prot_to_peprec(protein):
    params = get_params()
    tmp = pd.DataFrame(columns=['spec_id', 'peptide', 'modifications', 'charge'])
    pep_count = 0
    for peptide in cleave(str(protein.seq), expasy_rules['trypsin'], params['missed_cleavages']):
        if False not in [aa not in peptide for aa in ['B', 'J', 'O', 'U', 'X', 'Z']]:
            if params['min_peplen'] <= len(peptide) < int(params['max_pepmass'] / 186 + 2):
                if not mass.calculate_mass(sequence=peptide) > params['max_pepmass']:
                    pep_count += 1
                    row = {'spec_id': '{}_{:03d}'.format(protein.id, pep_count),
                           'peptide': peptide, 'modifications': '-', 'charge': np.nan}
                    tmp = tmp.append(row, ignore_index=True)
    return(tmp)


def get_protein_list(df):
    peptide_to_prot = {}
    for pi, pep in zip(df['spec_id'], df['peptide']):
        pi = '_'.join(pi.split('_')[0:2])
        if pep in peptide_to_prot.keys():
            peptide_to_prot[pep].append(pi)
        else:
            peptide_to_prot[pep] = [pi]
    df['protein_list'] = [list(set(peptide_to_prot[pep])) for pep in df['peptide']]
    df = df[~df.duplicated(['peptide', 'charge', 'modifications'])]
    return(df)


def add_mods(tup):
    """
    C-terminal modifications not yet supported!
    N-terminal modifications WITH specific first AA do not yet prevent other modifications
    to be added on that first AA. This means that the function will, for instance, combine
    Glu->pyro-Glu (combination of N-term and normal PTM) with other PTMS for Glu on the first
    AA, while this is not possible in reality!
    """
    _, row = tup
    params = get_params()
    mod_versions = [dict()]

    for i, mod in enumerate(params['modifications']):
        all_pos = [i for i, aa in enumerate(row['peptide']) if aa == mod[1]]
        if len(all_pos) > 4:
            all_pos = all_pos[:4]
        for version in mod_versions:
            # For non-position-specific mods:
            if not mod[3]:
                pos = [p for p in all_pos if p not in version.keys()]
                combos = [x for l in range(1, len(pos) + 1) for x in combinations(pos, l)]
                for combo in combos:
                    new_version = version.copy()
                    for pos in combo:
                        new_version[pos] = mod[0]
                    mod_versions.append(new_version)

            # For N-term mods and position not yet modified:
            elif mod[3] and 'N' not in version.keys():
                    # N-term with specific first AA:
                    if mod[1]:
                        if row['peptide'][0] == mod[1]:
                            new_version = version.copy()
                            new_version['N'] = mod[0]
                            mod_versions.append(new_version)
                    # N-term without specific first AA:
                    else:
                        new_version = version.copy()
                        new_version['N'] = mod[0]
                        mod_versions.append(new_version)

    df_out = pd.DataFrame(columns=row.index)
    df_out['modifications'] = ['|'.join('{}|{}'.format(0, value) if key == 'N' else '{}|{}'.format(key + 1, value) for key, value in version.items()) for version in mod_versions]
    df_out['modifications'] = ['-' if len(mods) == 0 else mods for mods in df_out['modifications']]
    df_out['spec_id'] = ['{}_{:03d}'.format(row['spec_id'], i) for i in range(len(mod_versions))]
    df_out['charge'] = row['charge']
    df_out['peptide'] = row['peptide']
    if 'protein_list' in row.index:
        df_out['protein_list'] = str(row['protein_list'])
    return(df_out)


def add_charges(df_in):
    params = get_params()
    df_out = pd.DataFrame(columns=df_in.columns)
    for charge in params['charges']:
        tmp = df_in.copy()
        tmp['spec_id'] = tmp['spec_id'] + '_{}'.format(charge)
        tmp['charge'] = charge
        df_out = df_out.append(tmp, ignore_index=True)
    df_out.sort_values(['spec_id', 'charge'], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return(df_out)


def run_batches(peprec, decoy=False):
    params = get_params()
    if decoy:
        params['output_filename'] += '_decoy'

    ms2pip_params = {
        'frag_method': params['frag_method'],
        'frag_error': 0.02,
        'ptm': ['{},{},opt,{}'.format(mods[0], mods[2], mods[1]) if not mods[3] else '{},{},opt,N-term'.format(mods[0], mods[2]) for mods in params['modifications']],
        'sptm': [],
        'gptm': [],
    }

    # Split up into batches to save memory:
    b_size = params['batch_size']
    b_count = 0
    num_b_counts = ceil(len(peprec) / b_size)
    for i in range(0, len(peprec), b_size):
        if i + b_size < len(peprec):
            peprec_batch = peprec[i:i + b_size]
        else:
            peprec_batch = peprec[i:]
        b_count += 1
        logging.info("Working on batch {} of {}, containing {} unmodified peptides".format(b_count, num_b_counts, len(peprec_batch)))

        logging.info("Adding all modification combinations")
        peprec_mods = pd.DataFrame(columns=peprec_batch.columns)
        with Pool(params['num_cpu']) as p:
            peprec_mods = peprec_mods.append(p.map(add_mods, peprec_batch.iterrows()), ignore_index=True)
        peprec_batch = peprec_mods

        logging.info("Adding charge states {}".format(params['charges']))
        peprec_batch = add_charges(peprec_batch)

        # Write ptm/charge-extended peprec from this batch to H5 file:
        # peprec_batch.astype(str).to_hdf(
        #     '{}_expanded_{}.peprec.hdf'.format(params['output_filename'], b_count), key='table',
        #     format='table', complevel=3, complib='zlib', mode='w'
        # )

        logging.info("Running MS2PIPc for {} peptides".format(len(peprec_batch)))
        all_preds = run(peprec_batch, num_cpu=params['num_cpu'], output_filename=params['output_filename'],
                        params=ms2pip_params, return_results=True)

        if b_count == 1:
            write_mode = 'w'
            append = False
        else:
            write_mode = 'a'
            append = True

        if 'hdf' in params['output_filetype']:
            logging.info("Writing predictions to {}_predictions.hdf".format(params['output_filename']))
            all_preds.astype(str).to_hdf(
                '{}_predictions.hdf'.format(params['output_filename']),
                key='table', format='table', complevel=3, complib='zlib',
                mode=write_mode, append=append, min_itemsize=50
            )

        if 'msp' in params['output_filetype']:
            logging.info("Writing MSP file with unmodified peptides")
            write_msp(
                all_preds,
                peprec_batch[peprec_batch['modifications'] == '-'],
                output_filename="{}_unmodified".format(params['output_filename']),
                write_mode=write_mode,
                num_cpu=params['num_cpu']
            )

            logging.info("Writing MSP file with all peptides")
            write_msp(
                all_preds,
                peprec_batch,
                output_filename="{}_withmods".format(params['output_filename']),
                write_mode=write_mode,
                num_cpu=params['num_cpu']
            )

        if 'mgf' in params['output_filetype']:
            logging.info("Writing MGF file with unmodified peptides")
            write_mgf(
                all_preds,
                peprec=peprec_batch[peprec_batch['modifications'] == '-'],
                output_filename="{}_unmodified".format(params['output_filename']),
                write_mode=write_mode
            )

            logging.info("Writing MGF file with all peptides")
            write_mgf(
                all_preds,
                peprec=peprec_batch,
                output_filename="{}_withmods".format(params['output_filename']),
                write_mode=write_mode
            )

        del all_preds
        del peprec_batch


def main():
    params = get_params()
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    peprec = pd.DataFrame(columns=['spec_id', 'peptide', 'modifications', 'charge'])

    logging.info("Cleaving proteins, adding peptides to peprec")
    with Pool(params['num_cpu']) as p:
        peprec = peprec.append(p.map(prot_to_peprec, SeqIO.parse(params['fasta_filename'], "fasta")), ignore_index=True)

    logging.info("Removing peptide redundancy, adding protein list to peptides")
    peprec = get_protein_list(peprec)

    peprec_nonmod = peprec.copy()

    save_peprec = False
    if save_peprec:
        logging.info("Saving non-expanded PEPREC to {}.peprec.hdf".format(params['output_filename']))
        peprec_nonmod['protein_list'] = ['/'.join(prot) for prot in peprec_nonmod['protein_list']]
        peprec_nonmod.astype(str).to_hdf(
            '{}_nonexpanded.peprec.hdf'.format(params['output_filename']), key='table',
            format='table', complevel=3, complib='zlib', mode='w'
        )

    if not params['decoy']:
        del peprec_nonmod

    run_batches(peprec, decoy=False)

    # For testing
    # peprec_nonmod = pd.read_hdf('data/uniprot_proteome_yeast_head_nonexpanded.peprec.hdf', key='table')

    if params['decoy']:
        logging.info("Reversing sequences for decoy peptides")
        # Copy peptides, add 'decoy_' to spec_id, reverse sequences, remove palindromic sequences, delete protein_list
        peprec_decoy = peprec_nonmod.copy()
        peprec_decoy['spec_id'] = 'decoy_' + peprec_decoy['spec_id']
        peprec_decoy['peptide'] = peprec_decoy['peptide'].apply(lambda pep: pep[-2::-1] + pep[-1])
        peprec_decoy = peprec_decoy[~peprec_decoy['peptide'].isin(peprec_nonmod['peptide'])]
        peprec_decoy['protein_list'] = 'decoy'
        del peprec_nonmod

        logging.info("Predicting spectra for decoy peptides")
        run_batches(peprec_decoy, decoy=True)

    logging.info("Fasta2SpecLib is ready!")


if __name__ == "__main__":
    main()
