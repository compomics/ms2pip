#!/usr/bin/env python
# Native library
import sys
import argparse
import multiprocessing
from random import shuffle
import tempfile

# Third party
import numpy as np
import pandas as pd

# From project
from ms2pip_tools.spectrum_output import write_mgf, write_msp


def process_peptides(worker_num, data, a_map, afile, modfile, modfile2, PTMmap, fragmethod):
    """
    Function for each worker to process a list of peptides. The models are
    chosen based on fragmethod, PTMmap, Ntermmap and Ctermmap determine the
    modifications applied to each peptide sequence. Returns the predicted
    spectra for all the peptides.
    """

    # Import ms2pipfeatures_pyx
    # Import is variable, depending on the frag_method defined by the user.
    # This needs to be done inside process_peptides and inside process_spectra,
    # as ms2pipfeatures_pyx cannot be passed as an argument through multiprocessing.
    # Also, in order to be compatible with MS2PIP Server (which calls the
    # function Run), this can not be done globally.
    cython_module_name = 'ms2pipfeatures_pyx_{}'.format(fragmethod)
    ms2pipfeatures_pyx = getattr(__import__('cython_modules', fromlist=[cython_module_name]), cython_module_name)

    ms2pipfeatures_pyx.ms2pip_init(bytearray(afile.encode()), bytearray(modfile.encode()), bytearray(modfile2.encode()))

    pcount = 0

    # Prepare output variables
    mz_buf = []
    prediction_buf = []
    peplen_buf = []
    charge_buf = []
    pepid_buf = []

    # transform pandas dataframe into dictionary for easy access
    specdict = data[["spec_id", "peptide", "modifications", "charge"]].set_index("spec_id").to_dict()
    pepids = data['spec_id'].tolist()
    peptides = specdict["peptide"]
    modifications = specdict["modifications"]
    charges = specdict["charge"]
    del specdict

    for pepid in pepids:
        peptide = peptides[pepid]
        peptide = peptide.replace("L", "I")
        mods = modifications[pepid]

        # Peptides longer then 101 lead to "Segmentation fault (core dumped)"
        if len(peptide) > 100:
            continue

        # convert peptide string to integer list to speed up C code
        peptide = np.array([0] + [a_map[x] for x in peptide] + [0], dtype=np.uint16)

        modpeptide = apply_mods(peptide, mods, PTMmap)
        if type(modpeptide) == str:
            if modpeptide == "Unknown modification":
                continue

        pepid_buf.append(pepid)
        peplen_buf.append(len(peptide) - 2)

        ch = charges[pepid]
        charge_buf.append(ch)

        # get ion mzs
        mzs = ms2pipfeatures_pyx.get_mzs(modpeptide)
        mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])

        # Predict the b- and y-ion intensities from the peptide
        # For C-term ion types (y, y++, z), flip the order of predictions,
        # because get_predictions follows order from vector file
        # enumerate works for variable number (and all) ion types
        predictions = ms2pipfeatures_pyx.get_predictions(peptide, modpeptide, ch)
        prediction_buf.append([np.array(p, dtype=np.float32) if i%2 == 0 else np.array(p[::-1], dtype=np.float32) for i, p in enumerate(predictions)])

        pcount += 1
        if (pcount % 500) == 0:
            sys.stdout.write("(%i)%i "%(worker_num, pcount))
            sys.stdout.flush()

    return mz_buf, prediction_buf, peplen_buf, charge_buf, pepid_buf


def process_spectra(worker_num, spec_file, vector_file, data, a_map, afile, modfile, modfile2, PTMmap, fragmethod, fragerror):
    """
    Function for each worker to process a list of spectra. Each peptide's
    sequence is extracted from the mgf file. Then models are chosen based on
    fragmethod, PTMmap, Ntermmap and Ctermmap determine the modifications
    applied to each peptide sequence and the spectrum is predicted. Then either
    the feature vectors are returned, or a DataFrame with the predicted and
    empirical intensities.
    """

    # Import ms2pipfeatures_pyx
    # Import is variable, depending on the frag_method defined by the user.
    # This needs to be done inside process_peptides and inside process_spectra,
    # as ms2pipfeatures_pyx cannot be passed as an argument through multiprocessing.
    # Also, in order to be compatible with MS2PIP Server (which calls the
    # function Run), this can not be done globally.
    cython_module_name = 'ms2pipfeatures_pyx_{}'.format(fragmethod)
    ms2pipfeatures_pyx = getattr(__import__('cython_modules', fromlist=[cython_module_name]), cython_module_name)

    ms2pipfeatures_pyx.ms2pip_init(bytearray(afile.encode()), bytearray(modfile.encode()), bytearray(modfile2.encode()))

    # transform pandas datastructure into dictionary for easy access
    specdict = data[["spec_id", "peptide", "modifications"]].set_index("spec_id").to_dict()
    peptides = specdict["peptide"]
    modifications = specdict["modifications"]

    # cols contains the names of the computed features
    cols_n = get_feature_names_new()
    #cols_n = get_feature_names_catboost()

    #for ti,tt in enumerate(names):
    #   print("%i %s"%(ti+1,tt))

    #SD
    # dresults = []
    dvectors = []
    dtargets = dict()
    psmids = []

    mz_buf = []
    target_buf = []
    prediction_buf = []
    peplen_buf = []
    charge_buf = []
    pepid_buf = []

    title = ""
    charge = 0
    msms = []
    peaks = []
    f = open(spec_file)
    skip = False
    pcount = 0
    while 1:
        rows = f.readlines(50000)
        if not rows:
            break
        for row in rows:
            row = row.rstrip()
            if row == "":
                continue
            if skip:
                if row[0] == "B":
                    if row[:10] == "BEGIN IONS":
                        skip = False
                else:
                    continue
            if row == "":
                continue
            if row[0] == "T":
                if row[:5] == "TITLE":
                    title = row[6:]
                    if title not in peptides:
                        skip = True
                        continue
            elif row[0].isdigit():
                tmp = row.split()
                msms.append(float(tmp[0]))
                peaks.append(float(tmp[1]))
            elif row[0] == "B":
                if row[:10] == "BEGIN IONS":
                    msms = []
                    peaks = []
            elif row[0] == "C":
                if row[:6] == "CHARGE":
                    charge = int(row[7:9].replace("+", ""))
            elif row[:8] == "END IONS":
                # process current spectrum
                if title not in peptides:
                    continue

                peptide = peptides[title]
                peptide = peptide.replace("L", "I")
                mods = modifications[title]
                #SD
                if "mut" in mods:
                    continue

                # Peptides longer then 101 lead to "Segmentation fault (core dumped)"
                if len(peptide) > 100:
                    continue

                # convert peptide string to integer list to speed up C code
                peptide = np.array([0] + [a_map[x] for x in peptide] + [0], dtype=np.uint16)

                modpeptide = apply_mods(peptide, mods, PTMmap)
                if type(modpeptide) == str:
                    if modpeptide == "Unknown modification":
                        continue

                # remove reporter ions
                if 'iTRAQ' in fragmethod:
                    for mi, mp in enumerate(msms):
                        if (mp >= 113) & (mp <= 118):
                            peaks[mi] = 0

                # TMT6plex: 126.1277, 127.1311, 128.1344, 129.1378, 130.1411, 131.1382
                if 'TMT' in fragmethod:
                    for mi, mp in enumerate(msms):	
                        if (mp >= 125) & (mp <= 132):
                            peaks[mi] = 0

                # normalize and convert MS2 peaks
                msms = np.array(msms, dtype=np.float32)
                peaks = peaks / np.sum(peaks)
                peaks = np.log2(np.array(peaks) + 0.001)
                peaks = np.array(peaks)
                peaks = peaks.astype(np.float32)

                # get ion mzs
                mzs = ms2pipfeatures_pyx.get_mzs(modpeptide)

                # get targets
                targets = ms2pipfeatures_pyx.get_targets(modpeptide, msms, peaks, float(fragerror))

                if vector_file:
                    psmids.extend([title]*(len(targets[0])))
                    # Temporary: until new features are incorporated into other fragmethods
                    if fragmethod in ['HCD', 'HCDch2', 'HCDTMT']:
                        dvectors.append(np.array(ms2pipfeatures_pyx.get_vector_new(peptide, modpeptide, charge), dtype=np.uint16))
                        #dvectors.append(np.array(ms2pipfeatures_pyx.get_vector_catboost(peptide, modpeptide, charge), dtype=np.uint16))
                    else:
                        dvectors.append(np.array(ms2pipfeatures_pyx.get_vector(peptide, modpeptide, charge), dtype=np.uint16))

                    # Collecting targets to dict; works for variable number of ion types
                    # For C-term ion types (y, y++, z), flip the order of targets,
                    # for correct order in vectors DataFrame
                    for i, t in enumerate(targets):
                        if i in dtargets.keys():
                            if i % 2 == 0:
                                dtargets[i].extend(t)
                            else:
                                dtargets[i].extend(t[::-1])
                        else:
                            if i % 2 == 0:
                                dtargets[i] = [t]
                            else:
                                dtargets[i] = [t[::-1]]

                else:
                    # Predict the b- and y-ion intensities from the peptide
                    # For C-term ion types (y, y++, z), flip the order of predictions,
                    # because get_predictions follows order from vector file
                    # enumerate works for variable number (and all) ion types
                    pepid_buf.append(title)
                    peplen_buf.append(len(peptide) - 2)
                    charge_buf.append(charge)

                    # get/append ion mzs, targets and predictions
                    mzs = ms2pipfeatures_pyx.get_mzs(modpeptide)
                    mz_buf.append([np.array(m, dtype=np.float32) for m in mzs])

                    target_buf.append([np.array(t, dtype=np.float32) for t in targets])

                    predictions = ms2pipfeatures_pyx.get_predictions(peptide, modpeptide, charge)
                    prediction_buf.append([np.array(p, dtype=np.float32) if i%2 == 0 else np.array(p[::-1], dtype=np.float32) for i, p in enumerate(predictions)])

                pcount += 1
                if (pcount % 500) == 0:
                    sys.stdout.write("(%i)%i "%(worker_num, pcount))
                    sys.stdout.flush()

    f.close()

    if vector_file:
        # Temporary: until new features are incorporated into other fragmethods
        # Concatenating dvectors into a 2D ndarray before making DataFrame saves lots of memory!
        if fragmethod in ['HCD', 'HCDch2', 'HCDTMT']:
            dvectors = np.concatenate(dvectors)
            df = pd.DataFrame(dvectors, columns=cols_n, dtype=np.uint16, copy=False)            
        else:
            dvectors = np.concatenate(dvectors)
            df = pd.DataFrame(dvectors, dtype=np.uint16, copy=False)
            df.columns = df.columns.astype(str)
        return psmids, df, dtargets

    return mz_buf, prediction_buf, target_buf, peplen_buf, charge_buf, pepid_buf


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
    names = ["amino_first","amino_last","amino_lcleave","amino_rcleave","peplen", "charge"]
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


def scan_spectrum_file(filename):
    """
    go over mgf file and return list with all spectrum titles
    """
    titles = []
    f = open(filename)
    while 1:
        rows = f.readlines(10000)
        if not rows:
            break
        for row in rows:
            if row[0] == "T":
                if row[:5] == "TITLE":
                    titles.append(row.rstrip()[6:])#.replace(" ", "") # unnecessary? creates issues when PEPREC spec_id has spaces
    f.close()
    return titles


def prepare_titles(titles, num_cpu):
    """
    Take a list and return a list containing num_cpu smaller lists with the
    spectrum titles/peptides that will be split across the workers
    """
    # titles might be ordered from small to large peptides,
    # shuffling improves parallel speeds
    shuffle(titles)

    split_titles = [titles[i * len(titles) // num_cpu: (i + 1) * len(titles) // num_cpu] for i in range(num_cpu)]
    sys.stdout.write("{} spectra (~{:.0f} per cpu)\n".format(len(titles), np.mean([len(a) for a in split_titles])))

    return split_titles


def apply_mods(peptide, mods, PTMmap):
    """
    Takes a peptide sequence and a set of modifications. Returns the modified
    version of the peptide sequence, c- and n-term modifications. This modified
    version are hard coded in ms2pipfeatures_c.c for now.
    """
    modpeptide = np.array(peptide[:], dtype=np.uint16)


    if mods != "-":
        l = mods.split("|")
        for i in range(0, len(l), 2):
            tl = l[i + 1]
            if tl in PTMmap:
                modpeptide[int(l[i])] = PTMmap[tl]
            else:
                sys.stderr.write("Unknown modification: {}\n".format(tl))
                return "Unknown modification"

    return modpeptide


def load_configfile(filepath):
    params = {}
    params['ptm'] = []
    params['sptm'] = []
    params['gptm'] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            (par, val) = line.split('=')
            if par == "ptm":
                params["ptm"].append(val)
            elif par == "sptm":
                params["sptm"].append(val)
            elif par == "gptm":
                params["gptm"].append(val)
            else:
                params[par] = val
    return params


def generate_modifications_file(params, masses, a_map):
    PTMmap = {}

    ptmnum = 38  # Omega compatibility (mutations)
    spbuffer = []
    for v in params["sptm"]:
        l = v.split(',')
        tmpf = float(l[1])
        if l[2] == 'opt':
            if l[3] == "N-term":
                spbuffer.append([tmpf, -1, ptmnum])
                PTMmap[l[0]] = ptmnum
                ptmnum += 1
                continue
            if l[3] == "C-term":
                spbuffer.append([tmpf, -2, ptmnum])
                PTMmap[l[0]] = ptmnum
                ptmnum += 1
                continue
            if not l[3] in a_map:
                continue
            spbuffer.append([tmpf, a_map[l[3]], ptmnum])
            PTMmap[l[0]] = ptmnum
            ptmnum += 1
    pbuffer = []
    for v in params["ptm"]:
        l = v.split(',')
        tmpf = float(l[1])
        if l[2] == 'opt':
            if l[3] == "N-term":
                pbuffer.append([tmpf, -1, ptmnum])
                PTMmap[l[0]] = ptmnum
                ptmnum += 1
                continue
            if l[3] == "C-term":
                pbuffer.append([tmpf, -2, ptmnum])
                PTMmap[l[0]] = ptmnum
                ptmnum += 1
                continue
            if not l[3] in a_map:
                continue
            pbuffer.append([tmpf, a_map[l[3]], ptmnum])
            PTMmap[l[0]] = ptmnum
            ptmnum += 1

    f = tempfile.NamedTemporaryFile(delete=False, mode='wb')
    f.write(str.encode("{}\n".format(len(pbuffer))))
    for i, _ in enumerate(pbuffer):
        f.write(str.encode("{},1,{},{}\n".format(pbuffer[i][0], pbuffer[i][1], pbuffer[i][2])))
    f.close()

    f2 = tempfile.NamedTemporaryFile(delete=False, mode='wb')
    f2.write(str.encode("{}\n".format(len(spbuffer))))
    for i, _ in enumerate(spbuffer):
        f2.write(str.encode("{},1,{},{}\n".format(spbuffer[i][0], spbuffer[i][1], spbuffer[i][2])))
    f2.close()

    return f.name, f2.name, PTMmap


def peakcount(x):
    c = 0.
    for i in x:
        if i > -9.95:
            c += 1.
    return c / len(x)


def calc_correlations(df):
    correlations = df.groupby(['spec_id', 'charge', 'ion'])[['target', 'prediction']].corr().iloc[::2]['prediction']
    correlations.index = correlations.index.droplevel(3)
    correlations = correlations.to_frame().reset_index()
    correlations.columns = ['spec_id', 'charge', 'ion', 'pearsonr']
    return correlations


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("pep_file", metavar="<peptide file>",
                        help="list of peptides")
    parser.add_argument("-c", metavar="FILE", action="store", dest="config_file", default="config.txt",
                        help="config file (by default config.txt)")
    parser.add_argument("-s", metavar="FILE", action="store", dest="spec_file",
                        help=".mgf MS2 spectrum file (optional)")
    parser.add_argument("-w", metavar="FILE", action="store", dest="vector_file",
                        help="write feature vectors to FILE.{pkl,h5} (optional)")
    parser.add_argument("-m", metavar="INT", action="store", dest="num_cpu",
                        default="23", help="number of cpu's to use")
    args = parser.parse_args()

    if not args.config_file:
        print("Please provide a configfile (-c)!")
        exit(1)

    return(args.pep_file, args.spec_file, args.vector_file, args.config_file, int(args.num_cpu))


def print_logo():
    logo = """
 _____ _____ ___ _____ _____ _____
|     |   __|_  |  _  |     |  _  |
| | | |__   |  _|   __|-   -|   __|
|_|_|_|_____|___|__|  |_____|__|

           """
    print(logo)
    print("by sven.degroeve@ugent.be\n")


def run(pep_file, spec_file=None, vector_file=None, config_file=None, num_cpu=23, params=None,
        output_filename=None, datasetname=None, return_results=False, limit=None):
    # datasetname is needed for Omega compatibility. This can be set to None if a config_file is provided

    # Create a_map:
    # a_map converts the peptide amino acids to integers, note how "L" is removed
    aminos = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "M", "N", "P", "Q",
              "R", "S", "T", "V", "W", "Y"]
    masses = [
        71.037114, 103.00919, 115.026943, 129.042593, 147.068414,
        57.021464, 137.058912, 113.084064, 128.094963, 131.040485,
        114.042927, 97.052764, 128.058578, 156.101111, 87.032028,
        101.047679, 99.068414, 186.079313, 163.063329,
        # 147.0354 iTRAQ fixed N-term modification (gets written to amino acid masses file)
    ]
    a_map = {a: i for i, a in enumerate(aminos)}

    # If not specified, get parameters from config_file
    if params is None:
        if config_file is None:
            if datasetname is None:
                print("No config file specified")
                exit(1)
        else:
            params = load_configfile(config_file)

    fragmethod = params["frag_method"]
    fragerror = params["frag_error"]

    # Check if given fragmethod exists:
    frag_method_ion_types = {
        'CID': ['b', 'y'],
        'HCD': ['b', 'y'],
        'HCDiTRAQ4phospho': ['b', 'y'],
        'HCDiTRAQ4': ['b', 'y'],
        'HCDTMT': ['b', 'y'],
        'ETD': ['b', 'y', 'c', 'z'],
        'HCDch2': ['b', 'y', 'b2', 'y2'],
        'TTOF5600': ['b', 'y'],
    }
    if fragmethod in frag_method_ion_types.keys():
        print("using {} models".format(fragmethod))
    else:
        print("Unknown fragmentation method: {}".format(fragmethod))
        print("Should be one of the following methods: {}".format(frag_method_ion_types.keys()))
        exit(1)

    if output_filename is None and not return_results:
        output_filename = '{}_{}'.format('.'.join(pep_file.split('.')[:-1]), fragmethod)

    # Create amino acid masses file
    # to be compatible with Omega
    # that might have fixed modifications
    f = tempfile.NamedTemporaryFile(delete=False)
    for m in masses:
        f.write(str.encode("{}\n".format(m)))
    f.write(str.encode("0\n"))
    f.close()
    afile = f.name

    # PTMs are loaded the same as in Omega
    # This allows me to use the same C init() function in bot ms2ip and Omega
    (modfile, modfile2, PTMmap) = generate_modifications_file(params, masses, a_map)

    # read peptide information
    # the file contains the columns: spec_id, modifications, peptide and charge
    if type(pep_file) == str:
        with open(pep_file, 'rt') as f:
            line = f.readline()
            if line[:7] != 'spec_id':
                sys.stdout.write('PEPREC file should start with header column\n')
                exit(1)
            sep = line[7]
        data = pd.read_csv(pep_file,
                           sep=sep,
                           index_col=False,
                           dtype={"spec_id": str, "modifications": str},
                           nrows=limit)
    else:
        data = pep_file
    # for some reason the missing values are converted to float otherwise
    data = data.fillna("-")

    sys.stdout.write("starting workers...\n")
    myPool = multiprocessing.Pool(num_cpu)

    if spec_file:
        """
        When an mgf file is provided, MS2PIP either saves the feature vectors to
        train models with or writes a file with the predicted spectra next to
        the empirical one.
        """
        sys.stdout.write("scanning spectrum file... \n")
        titles = scan_spectrum_file(spec_file)
        split_titles = prepare_titles(titles, num_cpu)
        results = []

        for i in range(num_cpu):
            tmp = split_titles[i]
            """
            process_spectra(
                i,
                spec_file,
                vector_file,
                data[data["spec_id"].isin(tmp)],
                a_map, afile, modfile, modfile2, PTMmap, fragmethod, fragerror)
            """
            results.append(myPool.apply_async(process_spectra, args=(
                i,
                spec_file,
                vector_file,
                data[data["spec_id"].isin(tmp)],
                a_map, afile, modfile, modfile2, PTMmap, fragmethod, fragerror)))
            #"""
        myPool.close()
        myPool.join()

        sys.stdout.write("\nmerging results ")

        # Create vector file
        if vector_file:
            all_results = []
            for r in results:
                sys.stdout.write(".")
                psmids, df, dtargets = r.get()
                
                # dtargets is a dict, containing targets for every ion type (keys are int)
                for i, t in dtargets.items():
                    df["targets{}".format(frag_method_ion_types[fragmethod][i])] = np.concatenate(t, axis=None)
                df["psmid"] = psmids
                
                all_results.append(df)

            # Only concat DataFrames with content (we get empty ones if more cpu's than peptides)
            all_results = pd.concat([df for df in all_results if len(df) != 0])

            sys.stdout.write("\nwriting vector file {}... \n".format(vector_file))
            # write result. write format depends on extension:
            ext = vector_file.split(".")[-1]
            if ext == "pkl":
                all_results.to_pickle(vector_file + ".pkl")
            elif ext == "csv":
                all_results.to_csv(vector_file)
            else:
                # "table" is a tag used to read back the .h5
                all_results.to_hdf(vector_file, "table")

        # Predict and compare with MGF file
        else:
            mz_bufs = []
            prediction_bufs = []
            target_bufs = []
            peplen_bufs = []
            charge_bufs = []
            pepid_bufs = []
            for r in results:
                mz_buf, prediction_buf, target_buf, peplen_buf, charge_buf, pepid_buf = r.get()
                mz_bufs.extend(mz_buf)
                prediction_bufs.extend(prediction_buf)
                target_bufs.extend(target_buf)
                peplen_bufs.extend(peplen_buf)
                charge_bufs.extend(charge_buf)
                pepid_bufs.extend(pepid_buf)

            # Reconstruct DataFrame
            num_ion_types = len(mz_bufs[0])
            ions = []
            ionnumbers = []
            charges = []
            pepids = []
            for pi, pl in enumerate(peplen_bufs):
                [ions.extend([ion_type] * (pl - 1)) for ion_type in frag_method_ion_types[fragmethod]]
                ionnumbers.extend([x + 1 for x in range(pl - 1)] * num_ion_types)
                charges.extend([charge_bufs[pi]] * (num_ion_types * (pl - 1)))
                pepids.extend([pepid_bufs[pi]] * (num_ion_types * (pl - 1)))
            all_preds = pd.DataFrame()
            all_preds["spec_id"] = pepids
            all_preds["charge"] = charges
            all_preds["ion"] = ions
            all_preds["ionnumber"] = ionnumbers
            all_preds["mz"] = np.hstack(np.concatenate(mz_bufs, axis=None))
            all_preds["target"] = np.hstack(np.concatenate(target_bufs, axis=None))
            all_preds["prediction"] = np.hstack(np.concatenate(prediction_bufs, axis=None))

            sys.stdout.write("writing file {}_pred_and_emp.csv...\n".format(output_filename))
            all_preds.to_csv("{}_pred_and_emp.csv".format(output_filename), index=False)

            #sys.stdout.write('computing correlations...\n')
            #correlations = calc_correlations(all_preds)
            #correlations.to_csv("{}_correlations.csv".format(output_filename), index=True)

            sys.stdout.write("done! \n")

    # Only get the predictions
    else:
        sys.stdout.write("scanning peptide file... ")

        titles = data.spec_id.tolist()
        split_titles = prepare_titles(titles, num_cpu)
        results = []

        for i in range(num_cpu):
            tmp = split_titles[i]
            """
            process_peptides(
                i,
                data[data.spec_id.isin(tmp)],
                a_map, afile, modfile, modfile2, PTMmap, fragmethod)
            """
            results.append(myPool.apply_async(process_peptides, args=(
                i,
                data[data.spec_id.isin(tmp)],
                a_map, afile, modfile, modfile2, PTMmap, fragmethod)))
            #"""
        myPool.close()
        myPool.join()

        sys.stdout.write("merging results...\n")

        mz_bufs = []
        prediction_bufs = []
        peplen_bufs = []
        charge_bufs = []
        pepid_bufs = []
        for r in results:
            mz_buf, prediction_buf, peplen_buf, charge_buf, pepid_buf = r.get()
            mz_bufs.extend(mz_buf)
            prediction_bufs.extend(prediction_buf)
            peplen_bufs.extend(peplen_buf)
            charge_bufs.extend(charge_buf)
            pepid_bufs.extend(pepid_buf)

        # Reconstruct DataFrame
        num_ion_types = len(mz_bufs[0])

        ions = []
        ionnumbers = []
        charges = []
        pepids = []
        for pi, pl in enumerate(peplen_bufs):
            [ions.extend([ion_type] * (pl - 1)) for ion_type in frag_method_ion_types[fragmethod]]
            ionnumbers.extend([x + 1 for x in range(pl - 1)] * num_ion_types)
            charges.extend([charge_bufs[pi]] * (num_ion_types * (pl - 1)))
            pepids.extend([pepid_bufs[pi]] * (num_ion_types * (pl - 1)))
        all_preds = pd.DataFrame()
        all_preds["spec_id"] = pepids
        all_preds["charge"] = charges
        all_preds["ion"] = ions
        all_preds["ionnumber"] = ionnumbers
        all_preds["mz"] = np.hstack(np.concatenate(mz_bufs, axis=None))
        all_preds["prediction"] = np.hstack(np.concatenate(prediction_bufs, axis=None))


        mgf = False  # Set to True to write spectrum as MGF file
        if mgf:
            print("writing MGF file {}_predictions.mgf...".format(output_filename))
            write_mgf(all_preds, peprec=data, output_filename=output_filename)

        msp = False  # Set to True to write spectra as MSP file
        if msp:
            print("writing MSP file {}_predictions.msp...".format(output_filename))
            write_msp(all_preds, data, output_filename=output_filename)

        if not return_results:
            sys.stdout.write("writing file {}_predictions.csv...\n".format(output_filename))
            all_preds.to_csv("{}_predictions.csv".format(output_filename), index=False)
            sys.stdout.write("done!\n")
        else:
            return all_preds

if __name__ == "__main__":
    print_logo()
    pep_file, spec_file, vector_file, config_file, num_cpu = argument_parser()
    params = load_configfile(config_file)
    run(pep_file, spec_file=spec_file, vector_file=vector_file, params=params, num_cpu=num_cpu)
