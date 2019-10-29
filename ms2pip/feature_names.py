def get_feature_names():
	"""
	feature names for the fixed peptide length feature vectors
	"""
	aminos = ["A", "C", "D", "E", "F", "G", "H", "I", "K",
			  "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
	names = []
	for a in aminos:
		names.append("Ib_" + a)
	names.append("sumIbaG")
	names.append("meanIbwikiG")
	names.append("sumIywaG")
	names.append("meanIywikiG")

	names += ["pmz", "peplen", "ionnumber", "ionnumber_rel"]

	for c in ["aG", "wikiG", "mz", "bas", "heli", "hydro", "pI"]:
		names.append("sum_" + c)

	for c in ["mz", "bas", "heli", "hydro", "pI"]:
		names.append("mean_" + c)

	for c in ["max_{}", "min_{}", "max{}_b", "min{}_b", "max{}_y", "min{}_y"]:
		for b in ["bas", "heli", "hydro", "pI"]:
			names.append(c.format(b))

	names.append("mz_ion")
	names.append("mz_ion_other")
	names.append("mean_mz_ion")
	names.append("mean_mz_ion_other")

	for c in ["bas", "heli", "hydro", "pI"]:
		names.append("{}_ion".format(c))
		names.append("{}_ion_other".format(c))
		names.append("{}_ion_minus_ion_other".format(c))
		#names.append("mean_{}_ion".format(c))
		#names.append("mean_{}_ion_other".format(c))

	for c in ["plus_cleave{}", "times_cleave{}", "minus1_cleave{}", "minus2_cleave{}", "bsum{}", "ysum{}"]:
		for b in ["bas", "heli", "hydro", "pI"]:
			names.append(c.format(b))

	for pos in ["0", "1", "-2", "-1"]:
		for c in ["mz", "bas", "heli", "hydro", "pI", "wikiG", "P", "D", "E", "K", "R"]:
			names.append("loc_" + pos + "_" + c)

	for pos in ["i", "i+1"]:
		for c in ["wikiG", "P", "D", "E", "K", "R"]:
			names.append("loc_" + pos + "_" + c)

	for c in ["bas", "heli", "hydro", "pI", "mz"]:
		for pos in ["i", "i-1", "i+1", "i+2"]:
			names.append("loc_" + pos + "_" + c)

	names.append("charge")

	return names


def get_feature_names_catboost():
	num_props = 4
	names = ["amino_first", "amino_last", "amino_lcleave", "amino_rcleave", "peplen", "charge"]
	for t in range(5):
		names.append("charge"+str(t))
	for t in range(num_props):
		names.append("qmin_%i"%t)
		names.append("q1_%i"%t)
		names.append("q2_%i"%t)
		names.append("q3_%i"%t)
		names.append("qmax_%i"%t)
	names.append("len_n")
	names.append("len_c")

	for a in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'M',
			  'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
		names.append("I_n_%s"%a)
		names.append("I_c_%s"%a)

	for t in range(num_props):
		for pos in ["p0", "pend", "pi-1", "pi", "pi+1", "pi+2"]:
			names.append("prop_%i_%s"%(t, pos))
		names.append("sum_%i_n"%t)
		names.append("q0_%i_n"%t)
		names.append("q1_%i_n"%t)
		names.append("q2_%i_n"%t)
		names.append("q3_%i_n"%t)
		names.append("q4_%i_n"%t)
		names.append("sum_%i_c"%t)
		names.append("q0_%i_c"%t)
		names.append("q1_%i_c"%t)
		names.append("q2_%i_c"%t)
		names.append("q3_%i_c"%t)
		names.append("q4_%i_c"%t)

	return names


def get_feature_names_new():
	num_props = 4
	names = ["peplen", "charge"]
	for t in range(5):
		names.append("charge"+str(t))
	for t in range(num_props):
		names.append("qmin_%i"%t)
		names.append("q1_%i"%t)
		names.append("q2_%i"%t)
		names.append("q3_%i"%t)
		names.append("qmax_%i"%t)
	names.append("len_n")
	names.append("len_c")

	for a in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'M',
			  'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
		names.append("I_n_%s"%a)
		names.append("I_c_%s"%a)

	for t in range(num_props):
		for pos in ["p0", "pend", "pi-1", "pi", "pi+1", "pi+2"]:
			names.append("prop_%i_%s"%(t, pos))
		names.append("sum_%i_n"%t)
		names.append("q0_%i_n"%t)
		names.append("q1_%i_n"%t)
		names.append("q2_%i_n"%t)
		names.append("q3_%i_n"%t)
		names.append("q4_%i_n"%t)
		names.append("sum_%i_c"%t)
		names.append("q0_%i_c"%t)
		names.append("q1_%i_c"%t)
		names.append("q2_%i_c"%t)
		names.append("q3_%i_c"%t)
		names.append("q4_%i_c"%t)

	return names


def get_feature_names_small(ionnumber):
	"""
	feature names for the fixed peptide length feature vectors
	"""
	names = []
	names += ["pmz", "peplen"]

	for c in ["bas", "heli", "hydro", "pI"]:
		names.append("sum_" + c)

	for c in ["mz", "bas", "heli", "hydro", "pI"]:
		names.append("mean_" + c)

	names.append("mz_ion")
	names.append("mz_ion_other")
	names.append("mean_mz_ion")
	names.append("mean_mz_ion_other")

	for c in ["bas", "heli", "hydro", "pI"]:
		names.append("{}_ion".format(c))
		names.append("{}_ion_other".format(c))

	names.append("endK")
	names.append("endR")
	names.append("nextP")
	names.append("nextK")
	names.append("nextR")

	for c in ["bas", "heli", "hydro", "pI", "mz"]:
		for pos in ["i", "i-1", "i+1", "i+2"]:
			names.append("loc_" + pos + "_" + c)

	names.append("charge")

	for i in range(ionnumber):
		for c in ["bas", "heli", "hydro", "pI", "mz"]:
			names.append("P_%i_%s"%(i, c))
		names.append("P_%i_P"%i)
		names.append("P_%i_K"%i)
		names.append("P_%i_R"%i)

	return names


def get_feature_names_chem(peplen):
	"""
	feature names for the fixed peptide length feature vectors
	"""

	names = []
	names += ["pmz", "peplen", "ionnumber", "ionnumber_rel", "mean_mz"]

	for c in ["mean_{}", "max_{}", "min_{}", "max{}_b", "min{}_b", "max{}_y", "min{}_y"]:
		for b in ["bas", "heli", "hydro", "pI"]:
			names.append(c.format(b))

	for c in ["mz", "bas", "heli", "hydro", "pI"]:
		names.append("{}_ion".format(c))
		names.append("{}_ion_other".format(c))
		names.append("mean_{}_ion".format(c))
		names.append("mean_{}_ion_other".format(c))

	for c in ["plus_cleave{}", "times_cleave{}", "minus1_cleave{}", "minus2_cleave{}", "bsum{}", "ysum{}"]:
		for b in ["bas", "heli", "hydro", "pI"]:
			names.append(c.format(b))

	for i in range(peplen):
		for c in ["mz", "bas", "heli", "hydro", "pI"]:
			names.append("fix_" + c + "_" + str(i))

	names.append("charge")

	return names
