"""
Write spectrum files from MS2PIP predictions.
"""


__author__ = "Ralf Gabriels"
__credits__ = ["Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__email__ = "Ralf.Gabriels@ugent.be"


# Native libraries
from ast import literal_eval
from operator import itemgetter
from io import StringIO

# Third party libraries
import pandas as pd
try:
    from tqdm import tqdm
except:
    use_tqdm = False
else:
    use_tqdm = True

def write_msp(all_preds_in, peprec_in, output_filename, write_mode='wt+'):
    """
    Write MS2PIP predictions to MSP spectral library file.
    """

    all_preds = all_preds_in.copy()
    peprec = peprec_in.copy()
    all_preds.reset_index(drop=True, inplace=True)
    # If not already normalized, normalize spectra
    if not (all_preds['prediction'].min() == 0 and all_preds['prediction'].max() == 10000):
        all_preds['prediction'] = ((2**all_preds['prediction']) - 0.001).clip(lower=0)
        all_preds['prediction'] = all_preds.groupby(['spec_id'])['prediction'].apply(lambda x: (x / x.max()) * 10000)
        all_preds['prediction'] = all_preds['prediction'].astype(int)

    # Check if protein list and rt are present in peprec
    add_protein = 'protein_list' in peprec.columns
    add_rt = 'rt' in peprec.columns

    # Convert RT from min to sec
    if add_rt:
        peprec['rt'] = peprec['rt'] * 60

    # Split titles (according to MS2PIPc)
    spec_ids = peprec['spec_id'].tolist()

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
    if add_protein:
        protein_list_index = peprec_col_names.index('protein_list')
    if add_rt:
        rt_index = peprec_col_names.index('rt')

    for row in peprec_list:
        peprec_to_slice[row[spec_id_index]] = row

    with open("{}_predictions.msp".format(output_filename), write_mode) as f:
        if use_tqdm & len(spec_ids) > 100000:
            spec_ids_iterator = tqdm(spec_ids)
        else:
            spec_ids_iterator = spec_ids
        for spec_id in spec_ids_iterator:
            out = []
            preds = preds_to_slice[spec_id]
            peprec_sel = peprec_to_slice[spec_id]

            preds = sorted(preds, key=itemgetter(mz_index))

            sequence = peprec_sel[peptide_index]
            charge = peprec_sel[charge_index]
            mods = peprec_sel[modifications_index]
            numpeaks = len(preds)

            # Calculate mass from fragment ions
            mass_b = [row[mz_index] for row in preds if row[ion_index] == 'B' and row[ionnumber_index] == 1][0]
            mass_y = [row[mz_index] for row in preds if row[ion_index] == 'Y' and row[ionnumber_index] == numpeaks / 2][0]
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

            out.append("Parent={} ".format((pepmass + charge * 1.007236) / charge))

            if add_protein:
                try:
                    out.append('Protein="{}" '.format('/'.join(literal_eval(peprec_sel[protein_list_index]))))
                except ValueError:
                    out.append('Protein="{}" '.format(peprec_sel[protein_list_index]))

            if add_rt:
                out.append('RTINSECONDS={} '.format(peprec_sel[rt_index]))

            out.append('MS2PIP_ID="{}"'.format(spec_id))

            out.append('\nNum peaks: {}\n'.format(numpeaks))

            lines = list(zip(
                [row[mz_index] for row in preds],
                [row[prediction_index] for row in preds],
                [row[ion_index] for row in preds],
                [row[ionnumber_index] for row in preds]
            ))
            out.append(''.join(['{:.4f}\t{}\t"{}{}"\n'.format(*l) for l in lines]))
            out.append('\n')

            out_string = "".join(out)

            f.write(out_string)


def write_mgf(all_preds_in, output_filename="MS2PIP", unlog=True, write_mode='w+', return_stringbuffer=False, peprec=None):
    """
    Write MS2PIP predictions to MGF spectrum file.
    """
    all_preds = all_preds_in.copy()
    if unlog:
        all_preds['prediction'] = ((2**all_preds['prediction']) - 0.001).clip(lower=0)
        all_preds.reset_index(inplace=True)
        all_preds['prediction'] = all_preds.groupby(['spec_id'])['prediction'].apply(lambda x: x / x.sum())

    def write(all_preds, mgf_output, peprec=None):
        out = []

        # Create easy to access dict from all_preds and peprec dataframe
        if type(peprec) == pd.DataFrame:
            peprec_to_dict = peprec.copy()

            rt_present = 'rt' in peprec_to_dict.columns
            if rt_present:
                peprec_to_dict['rt'] = peprec_to_dict['rt'] * 60

            peprec_to_dict.index = peprec_to_dict['spec_id']
            peprec_to_dict.drop('spec_id', axis=1, inplace=True)
            peprec_dict = peprec_to_dict.to_dict(orient='index')
            del peprec_to_dict
            spec_id_list = list(peprec['spec_id'])

        else:
            rt_present = False
            spec_id_list = list(all_preds['spec_id'].unique())

        preds_dict = {}
        preds_list = all_preds[['spec_id', 'charge', 'ion', 'mz', 'prediction']].values.tolist()

        for row in preds_list:
            spec_id = row[0]
            if spec_id in preds_dict.keys():
                if row[2] in preds_dict[spec_id]['peaks']:
                    preds_dict[spec_id]['peaks'][row[2]].append(tuple(row[3:]))
                else:
                    preds_dict[spec_id]['peaks'][row[2]] = [tuple(row[3:])]
            else:
                preds_dict[spec_id] = {
                    'charge': row[1],
                    'peaks': {row[2]: [tuple(row[3:])]}
                }

        # Write MGF
        for spec_id in spec_id_list:
            out.append('BEGIN IONS')
            charge = preds_dict[spec_id]['charge']
            pepmass = preds_dict[spec_id]['peaks']['B'][0][0] + preds_dict[spec_id]['peaks']['Y'][-1][0] - 2 * 1.007236
            peaks = [item for sublist in preds_dict[spec_id]['peaks'].values() for item in sublist]
            peaks = sorted(peaks, key=itemgetter(0))

            if type(peprec) == pd.DataFrame:
                seq = peprec_dict[spec_id]['peptide']
                mods = peprec_dict[spec_id]['modifications']
                if rt_present:
                    rt = peprec_dict[spec_id]['rt']
                if mods == '-':
                    mods_out = '0'
                else:
                    # Write MSP style PTM string
                    mods = mods.split('|')
                    mods = [(int(mods[i]), mods[i + 1]) for i in range(0, len(mods), 2)]
                    # Turn MS2PIP mod indexes into actual list indexes (eg 0 for first AA)
                    mods = [(x, y) if x == 0 else (x - 1, y) for (x, y) in mods]
                    mods = [(str(x), seq[x], y) for (x, y) in mods]
                    mods_out = '{}/{}'.format(len(mods), '/'.join([','.join(list(x)) for x in mods]))
                out.append('TITLE={} {} {}'.format(spec_id, seq, mods_out))
            else:
                out.append('TITLE={}'.format(spec_id))

            out.append('PEPMASS={}'.format((pepmass + (charge * 1.007825032)) / charge))
            out.append('CHARGE={}+'.format(charge))
            if rt_present:
                out.append('RTINSECONDS={}'.format(rt))
            out.append('\n'.join([' '.join(['{:.8f}'.format(p) for p in peak]) for peak in peaks]))
            out.append('END IONS\n')

        mgf_output.write('\n'.join(out))

    if return_stringbuffer:
        mgf_output = StringIO()
        write(all_preds, mgf_output, peprec=peprec)
        return mgf_output
    else:
        with open("{}_predictions.mgf".format(output_filename), write_mode) as mgf_output:
            write(all_preds, mgf_output, peprec=peprec)

    del all_preds
