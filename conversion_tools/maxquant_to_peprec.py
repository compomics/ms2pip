__author__ = "Ralf Gabriels"
__credits__ = ["Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__email__ = "Ralf.Gabriels@ugent.be"


# Native libraries
import re
import logging
import argparse

# Third party libraries
import pandas as pd
from tqdm import tqdm


def maxquant_to_peprec(evidence_file, msms_file,
                       ptm_mapping={'(ox)': 'Oxidation', '(ac)': 'Acetyl', '(cm)': 'Carbamidomethyl'},
                       fixed_modifications=[]):
    """
    Make an MS2PIP PEPREC file starting from the MaxQuant Evidence.txt and
    MSMS.txt files.

    Positional arguments:
    `evidence_file`: str with the file location of the Evidence.txt file
    `msms_file`: str with the file location of the MSMS.txt file

    Keyword arguments:
    `ptm_mapping` (dict) is used to convert the MaxQuant PTM labels to PSI-MS
    modification names. For correct parsing, the key should always include the
    two brackets.
    `fixed_modifications` (list of tuples, [(aa, ptm)]) can contain fixed
    modifications to be added to the peprec. E.g. `[('C', 'cm')]`. The first
    tuple element contains the one-letter amino acid code. The second tuple
    element contains a two-character label for the PTM. This PTM also needs
    to be present in the `ptm_mapping` dictionary.
    """

    for (aa, mod) in fixed_modifications:
        if re.fullmatch('[A-Z]|n_term', aa) is None:
            raise ValueError("Fixed modification amino acid `{}` should be a single capital character (e.g. `C`) or `n_term`.".format(aa))
        if re.fullmatch('[a-z]{2}', mod) is None:
            raise ValueError("Fixed modification label `{}` can only contain two non-capital characters. E.g. `cm`".format(mod))

    logging.debug("Start converting MaxQuant output to MS2PIP PEPREC")

    # Read files and merge into one dataframe
    evidence = pd.read_csv(evidence_file, sep='\t')
    msms = pd.read_csv(msms_file, sep='\t', low_memory=False)
    m = evidence.merge(msms[['Scan number', 'Scan index', 'id']], left_on='Best MS/MS', right_on='id')

    # Filter data
    logging.debug("Removing decoys, contaminants, spectrum redundancy and peptide redundancy")
    len_dict = {
        'len_original': len(m),
        'len_rev': len(m[m['Reverse'] == '+']),
        'len_con': len(m[m['Potential contaminant'] == '+']),
        'len_spec_red': len(m[m.duplicated(['Raw file', 'Scan number'], keep='first')]),
        'len_pep_red': len(m[m.duplicated(['Sequence', 'Modifications', 'Charge'], keep='first')])
    }

    m = m.sort_values('Score', ascending=False)
    m = m[m['Reverse'] != '+']
    m = m[m['Potential contaminant'] != '+']
    m = m[~m.duplicated(['Raw file', 'Scan number'], keep='first')]
    m = m[~m.duplicated(['Sequence', 'Modifications', 'Charge'], keep='first')]
    m.sort_index(inplace=True)

    len_dict['len_filtered'] = len(m)

    print_psm_counts = """
    Original number of PSMs: {len_original}
     - {len_rev} decoys
     - {len_con} potential contaminants
     - {len_spec_red} redundant spectra
     - {len_pep_red} redundant peptides
    = {len_filtered} PSMs in spectral library
    """.format(**len_dict)

    logging.info(print_psm_counts)
    logging.info("Spectral library FDR estimated from PEPs: {:.4f}".format(m['PEP'].sum()/len(m)))

    # Parse modifications
    logging.debug("Parsing modifications")
    for aa, mod in fixed_modifications:
        if aa == 'n_term':
            m['Modified sequence'] = ['_({})'.format(mod) + seq[1:] if seq[:2] != '_(' else seq for seq in m['Modified sequence']]
        else:
            m['Modified sequence'] = m['Modified sequence'].str.replace(aa, '{}({})'.format(aa, mod))

    pattern = r'\([a-z].\)'
    m['Parsed modifications'] = ['|'.join(['{}|{}'.format(m.start(0) - 1 - i*4, ptm_mapping[m.group()]) for i, m in enumerate(re.finditer(pattern, s))]) for s in m['Modified sequence']]
    m['Parsed modifications'] = ['-' if mods == '' else mods for mods in m['Parsed modifications']]

    # Prepare PEPREC columns
    logging.debug("Preparing PEPREC columns")
    m['Protein list'] = m['Proteins'].str.split(';')
    m['MS2PIP ID'] = range(len(m))
    peprec_cols = ['MS2PIP ID', 'Sequence', 'Parsed modifications', 'Charge', 'Protein list',
                   'Retention time', 'Score', 'Delta score', 'PEP', 'Scan number', 'Scan index', 'Raw file']
    peprec = m[peprec_cols].sort_index().copy()

    col_mapping = {
        'MS2PIP ID': 'spec_id',
        'Sequence': 'peptide',
        'Parsed modifications': 'modifications',
        'Charge': 'charge',
        'Protein list': 'protein_list',
        'Retention time': 'rt',
        'Score': 'andromeda_score',
        'Delta score': 'andromeda_delta_score',
        'PEP': 'andromeda_pep',
        'Scan number': 'scan_number',
        'Scan index': 'scan_index',
        'Raw file': 'raw_file'
    }

    peprec = peprec.rename(columns=col_mapping)

    logging.debug('PEPREC ready!')

    return peprec


def scan_mgf(peprec, mgf_folder, filename_col='raw_file', scan_num_col='scan_number', outname='SpecLib.mgf'):
    with open(outname, 'w') as out:
        count_runs = 0
        count = 0
        runs = peprec[filename_col].unique()
        logging.info("Scanning MGF files...")
        for run in tqdm(runs):
            spec_dict = dict(('msmsid:F{:06d}'.format(v), k) for k, v in peprec[(peprec[filename_col] == run)][scan_num_col].to_dict().items())

            # Parse file
            found = False
            with open('{}/{}.mgf'.format(mgf_folder, str(run)), 'r') as f:
                for i, line in enumerate(f):
                    if 'TITLE' in line:
                        scan_num = re.split('TITLE=|,', line)[1]  # Edit to match MGF title notation
                        if scan_num in spec_dict:
                            found = True
                            out.write("BEGIN IONS\n")
                            line = "TITLE=" + str(spec_dict[scan_num]) + '\n'
                            count += 1
                    if 'END IONS' in line:
                        if found:
                            out.write(line + '\n')
                            found = False
                    if found and line[-4:] != '0.0\n':
                        out.write(line)
    logging.info("{}/{} spectra found and written to new MGF file.".format(count, len(peprec)))


def ArgParse():
    parser = argparse.ArgumentParser(description='Convert MaxQuant txt files and MGF to MS2PIP spectral library.')
    parser.add_argument('-t', '--txt', dest='txt_folder', action='store', default='txt',
                        help='Folder with MaxQuant txt files (default: "msf")')
    parser.add_argument('-g', '--mgf', dest='mgf_folder', action='store', default='mgf',
                        help='Folder with MGF spectrum files (default: "mgf")')
    parser.add_argument('-o', '--out', dest='outname', action='store', default='SpecLib',
                        help='Name for output files (default: "SpecLib")')
    args = parser.parse_args()

    return(args)


def main():
    args = ArgParse()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG
    )

    peprec = maxquant_to_peprec(
        '{}/evidence.txt'.format(args.txt_folder),
        '{}/msms.txt'.format(args.txt_folder),
        ptm_mapping={'(ox)': 'Oxidation', '(ac)': 'Acetyl', '(cm)': 'Carbamidomethyl', '(tn)': 'TMT6plexN', '(tk)': 'TMT6plex'},
        fixed_modifications=[('C', 'cm'), ('n_term', 'tn'), ('K', 'tk')]
    )

    peprec.to_csv('{}.peprec'.format(args.outname), sep=' ', index=False)

    scan_mgf(peprec, args.mgf_folder, outname='{}.mgf'.format(args.outname))



if __name__ == '__main__':
    main()
