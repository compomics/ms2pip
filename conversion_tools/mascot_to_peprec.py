__author__ = "Ralf Gabriels"
__credits__ = ["Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__email__ = "Ralf.Gabriels@ugent.be"


# Native libraries
import re
import logging

# Third party libraries
import pandas as pd


def mascot_to_peprec(evidence_file, msms_file,
                     ptm_mapping={'(ox)': 'Oxidation', '(ac)': 'Acetyl', '(cm)': 'Carbamidomethyl'},
                     fixed_modifications=[]):
    """
    Make an MS2PIP PEPREC file starting from the Mascot Evidence.txt and
    MSMS.txt files.

    Positional arguments:
    `evidence_file`: str with the file location of the Evidence.txt file
    `msms_file`: str with the file location of the MSMS.txt file

    Keyword arguments:
    `ptm_mapping` (dict) is used to convert the Mascot PTM labels to PSI-MS
    modification names. For correct parsing, the key should always include the
    two brackets.
    `fixed_modifications` (list of tuples, [(aa, ptm)]) can contain fixed
    modifications to be added to the peprec. E.g. `[('C', 'cm')]`. The first
    tuple element contains the one-letter amino acid code. The second tuple
    element contains a two-character label for the PTM. This PTM also needs
    to be present in the `ptm_mapping` dictionary.
    """

    for (aa, mod) in fixed_modifications:
        if re.fullmatch('[A-Z]', aa) is None:
            raise ValueError("Fixed modification amino acid `{}` should be a single capital character. E.g. `C`".format(aa))
        if re.fullmatch('[a-z]{2}', mod) is None:
            raise ValueError("Fixed modification label `{}` can only contain two non-capital characters. E.g. `cm`".format(mod))

    logging.debug("Start converting Mascot output to MS2PIP PEPREC")

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
        m['Modified sequence'] = m['Modified sequence'].str.replace(aa, '{}({})'.format(aa, mod))

    pattern = r'\([a-z].\)'
    m['Parsed modifications'] = ['|'.join(['{}|{}'.format(m.start(0) - 1 - i*4, ptm_mapping[m.group()]) for i, m in enumerate(re.finditer(pattern, s))]) for s in m['Modified sequence']]
    m['Parsed modifications'] = ['-' if mods == '' else mods for mods in m['Parsed modifications']]

    # Prepare PEPREC columns
    logging.debug("Preparing PEPREC columns")
    m['Protein list'] = m['Proteins'].str.split(';')
    m['MS2PIP ID'] = range(len(m))
    peprec_cols = ['MS2PIP ID', 'Sequence', 'Parsed modifications', 'Charge', 'Protein list',
                   'Retention time', 'Scan number', 'Scan index', 'Raw file']
    peprec = m[peprec_cols].sort_index().copy()

    col_mapping = {
        'MS2PIP ID': 'spec_id',
        'Sequence': 'peptide',
        'Parsed modifications': 'modifications',
        'Charge': 'charge',
        'Protein list': 'protein_list',
        'Retention time': 'rt',
    }

    peprec = peprec.rename(columns=col_mapping)

    logging.debug('PEPREC ready!')

    return peprec
