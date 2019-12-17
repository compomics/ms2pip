
def write_slf(all_preds_in, output_filename="MS2PIP", unlog=True,
			  write_mode='w+', return_stringbuffer=False, peprec=None):
	"""
	Write MS2PIP predictions to PSI-SLF file.
	"""
	all_preds = all_preds_in.copy()
	if unlog:
		all_preds['prediction'] = ((2**all_preds['prediction']) - 0.001).clip(lower=0)
		all_preds.reset_index(inplace=True)
		all_preds['prediction'] = all_preds.groupby(['spec_id'])['prediction'].apply(lambda x: x / x.sum())

	def write(all_preds, mgf_output, peprec=None):
		out = []

		peprec_dict, preds_dict, rt_present = dfs_to_dicts(all_preds, peprec=peprec, rt_to_seconds=True)

		# Write MGF
		if peprec_dict:
			spec_id_list = peprec_dict.keys()
		else:
			spec_id_list = list(all_preds['spec_id'].unique())

		for spec_id in sorted(spec_id_list):
			out.append('BEGIN IONS')
			charge = preds_dict[spec_id]['charge']
			pepmass = preds_dict[spec_id]['peaks']['B'][0][0] + preds_dict[spec_id]['peaks']['Y'][-1][0] - 2 * 1.007236
			peaks = [item for sublist in preds_dict[spec_id]['peaks'].values() for item in sublist]
			peaks = sorted(peaks, key=itemgetter(0))

			if peprec_dict:
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
		with open("{}_predictions.slf".format(output_filename), write_mode) as mgf_output:
			write(all_preds, mgf_output, peprec=peprec)

	del all_preds
