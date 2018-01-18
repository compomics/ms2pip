# Native library
import sys
import pickle
import argparse
import multiprocessing
from random import shuffle
import math
import tempfile

# Other
import numpy as np
import pandas as pd

# Features
import ms2pipfeatures_pyx_HCD
# import ms2pipfeatures_pyx_CID
# import ms2pipfeatures_pyx_HCDiTRAQ4phospho
# import ms2pipfeatures_pyx_HCDiTRAQ4
# import ms2pipfeatures_pyx_ETD


def process_peptides(worker_num, data, a_map, afile, modfile, modfile2, PTMmap, fragmethod):
    """
    Function for each worker to process a list of peptides. The models are
    chosen based on fragmethod, PTMmap, Ntermmap and Ctermmap determine the
    modifications applied to each peptide sequence. Returns the predicted
    spectra for all the peptides.
    """

    # Rename ms2pipfeatures_pyx
    if fragmethod == "CID":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_CID
    elif fragmethod == "HCD":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_HCD
    elif fragmethod == "HCDiTRAQ4phospho":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_HCDiTRAQ4phospho
    elif fragmethod == "HCDiTRAQ4":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_HCDiTRAQ4
    elif fragmethod == "HCD":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_HCD
    elif fragmethod == "ETD":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_ETD

    ms2pipfeatures_pyx.ms2pip_init(bytearray(afile.encode()), bytearray(modfile.encode()), bytearray(modfile2.encode()))

    # transform pandas dataframe into dictionary for easy access
    specdict = data[["spec_id", "peptide", "modifications", "charge"]].set_index("spec_id").to_dict()
    peptides = specdict["peptide"]
    modifications = specdict["modifications"]
    charges = specdict["charge"]

    final_result = pd.DataFrame(columns=["spec_id", "peplen", "charge", "ion", "ionnumber", "mz", "prediction"])

    pcount = 0

    for pepid in peptides:
        peptide = peptides[pepid]
        peptide = peptide.replace("L", "I")
        mods = modifications[pepid]

        # convert peptide string to integer list to speed up C code
        peptide = np.array([0] + [a_map[x] for x in peptide] + [0], dtype=np.uint16)

        modpeptide = apply_mods(peptide, mods, PTMmap)
        ch = charges[pepid]

        # get ion mzs
        mzs = ms2pipfeatures_pyx.get_mzs(modpeptide)

        # get ion intensities
        predictions = ms2pipfeatures_pyx.get_predictions(peptide, modpeptide, ch)

        # return predictions as a DataFrame
        tmp = pd.DataFrame(columns=['spec_id', 'peplen', 'charge', 'ion', 'ionnumber', 'mz', 'prediction'])
        num_ions = len(predictions[0])
        if fragmethod == 'ETD':
            tmp["ion"] = ['b'] * num_ions + ['y'] * num_ions + ['c'] * num_ions + ['z'] * num_ions
            tmp["ionnumber"] = list(range(1, num_ions + 1)) * 4
            tmp["mz"] = mzs[0] + mzs[1] + mzs[2] + mzs[3]
            tmp["prediction"] = predictions[0] + predictions[1] + predictions[2] + predictions[3]
        else:
            tmp["ion"] = ['b'] * num_ions + ['y'] * num_ions
            tmp["mz"] = mzs[0] + mzs[1]
            tmp["ionnumber"] = list(range(1, num_ions + 1)) * 2
            tmp["prediction"] = predictions[0] + predictions[1]
        tmp["peplen"] = len(peptide) - 2
        tmp["charge"] = ch
        tmp["spec_id"] = pepid

        final_result = final_result.append(tmp)
        pcount += 1
        if (pcount % 500) == 0:
            sys.stderr.write("w" + str(worker_num) + "(" + str(pcount) + ") ")
    return final_result


def process_spectra(worker_num, spec_file, vector_file, data, a_map, afile, modfile, modfile2, PTMmap, fragmethod, fragerror):
    """
    Function for each worker to process a list of spectra. Each peptide's
    sequence is extracted from the mgf file. Then models are chosen based on
    fragmethod, PTMmap, Ntermmap and Ctermmap determine the modifications
    applied to each peptide sequence and the spectrum is predicted. Then either
    the feature vectors are returned, or a DataFrame with the predicted and
    empirical intensities.
    """

    # Rename ms2pipfeatures_pyx
    if fragmethod == "CID":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_CID
    elif fragmethod == "HCD":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_HCD
    elif fragmethod == "HCDiTRAQ4phospho":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_HCDiTRAQ4phospho
    elif fragmethod == "HCDiTRAQ4":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_HCDiTRAQ4
    elif fragmethod == "HCD":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_HCD
    elif fragmethod == "ETD":
        ms2pipfeatures_pyx = ms2pipfeatures_pyx_ETD

    ms2pipfeatures_pyx.ms2pip_init(bytearray(afile.encode()), bytearray(modfile.encode()), bytearray(modfile2.encode()))

    # transform pandas datastructure into dictionary for easy access
    specdict = data[["spec_id", "peptide", "modifications"]].set_index("spec_id").to_dict()
    peptides = specdict["peptide"]
    modifications = specdict["modifications"]

    # cols contains the names of the computed features
    cols_n = get_feature_names()

    dataresult = pd.DataFrame(columns=["spec_id", "peplen", "charge", "ion", "ionnumber", "mz", "target", "prediction"])
    dataresult["peplen"] = dataresult["peplen"].astype(np.uint8)
    dataresult["charge"] = dataresult["charge"].astype(np.uint8)
    dataresult["ion"] = dataresult["ion"].astype(np.uint8)
    dataresult["ionnumber"] = dataresult["ionnumber"].astype(np.uint8)
    dataresult["mz"] = dataresult["mz"].astype(np.float32)
    dataresult["target"] = dataresult["target"].astype(np.float32)
    dataresult["prediction"] = dataresult["prediction"].astype(np.float32)

    title = ""
    charge = 0
    msms = []
    peaks = []
    f = open(spec_file)
    skip = False
    vectors = []
    pcount = 0
    while 1:
        rows = f.readlines(3000)
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
                    title = row[6:].replace(" ", "")
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

                # convert peptide string to integer list to speed up C code
                peptide = np.array([0] + [a_map[x] for x in peptide] + [0], dtype=np.uint16)

                modpeptide = apply_mods(peptide, mods, PTMmap)

                if 'iTRAQ' in fragmethod:
                    # remove reporter ions
                    for mi, mp in enumerate(msms):
                        if (mp >= 113) & (mp <= 118):
                            peaks[mi] = 0

                # normalize and convert MS2 peaks
                msms = np.array(msms, dtype=np.float32)
                debug = False
                if debug:
                    peaks = np.array(peaks).astype(np.float32)
                else:
                    peaks = peaks / np.sum(peaks)
                    peaks = np.array(np.log2(peaks + 0.001))
                    peaks = peaks.astype(np.float32)

                # get ion mzs
                mzs = ms2pipfeatures_pyx.get_mzs(modpeptide)

                # get targets
                targets = ms2pipfeatures_pyx.get_targets(modpeptide, msms, peaks, float(fragerror))

                if vector_file:
                    tmp = pd.DataFrame(ms2pipfeatures_pyx.get_vector(peptide, modpeptide, charge), columns=cols_n, dtype=np.uint16)
                    # tmp = pd.DataFrame(ms2pipfeatures_pyx.get_vector(peptide, modpeptide, charge), dtype=np.uint16)
                    tmp["psmid"] = [title] * len(tmp)
                    tmp["targetsB"] = targets[0]
                    tmp["targetsY"] = targets[1][::-1]
                    if fragmethod == 'ETD':
                        tmp["targetsC"] = targets[2]
                        tmp["targetsZ"] = targets[3][::-1]
                    vectors.append(tmp)
                else:
                    # predict the b- and y-ion intensities from the peptide
                    predictions = ms2pipfeatures_pyx.get_predictions(peptide, modpeptide, charge)
                    tmp = pd.DataFrame(columns=['spec_id', 'peplen', 'charge', 'ion', 'ionnumber', 'mz', 'target', 'prediction'])
                    num_ions = len(predictions[0])
                    if fragmethod == 'ETD':
                        tmp["ion"] = ['b'] * num_ions + ['y'] * num_ions + ['c'] * num_ions + ['z'] * num_ions
                        tmp["ionnumber"] = list(range(1, num_ions + 1)) * 4
                        tmp["mz"] = mzs[0] + mzs[1] + mzs[2] + mzs[3]
                        tmp["target"] = targets[0] + targets[1] + targets[2] + targets[3]
                        tmp["prediction"] = predictions[0] + predictions[1] + predictions[2] + predictions[3]
                    else:
                        tmp["ion"] = ['b'] * num_ions + ['y'] * num_ions
                        tmp["ionnumber"] = list(range(1, num_ions + 1)) * 2
                        tmp["mz"] = mzs[0] + mzs[1]
                        tmp["target"] = targets[0] + targets[1]
                        tmp["prediction"] = predictions[0] + predictions[1]
                    tmp["spec_id"] = title
                    tmp["peplen"] = len(peptide) - 2
                    tmp["charge"] = charge

                    tmp["peplen"] = tmp["peplen"].astype(np.uint8)
                    tmp["charge"] = tmp["charge"].astype(np.uint8)
                    tmp["ionnumber"] = tmp["ionnumber"].astype(np.uint8)
                    tmp["mz"] = tmp["mz"].astype(np.float32)
                    tmp["target"] = tmp["target"].astype(np.float32)
                    tmp["prediction"] = tmp["prediction"].astype(np.float32)
                    dataresult = dataresult.append(tmp, ignore_index=True)

                pcount += 1
                if (pcount % 500) == 0:
                    sys.stderr.write("w" + str(worker_num) + "(" + str(pcount) + ") ")

    if vector_file:
        df = pd.DataFrame()
        for v in vectors:
            if len(v > 0):
                df = pd.concat([df, v])
            else:
                continue
        return df
    else:
        return dataresult


def get_feature_names():
    """
    feature names for the fixed peptide length feature vectors
    """
    aminos = ["A", "C", "D", "E", "F", "G", "H", "I", "K",
              "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    names = []
    for a in aminos:
        names.append("Ib_" + a)
    for a in aminos:
        names.append("Iy_" + a)
    names += ["pmz", "peplen", "ionnumber", "ionnumber_rel"]

    for c in ["mz", "bas", "heli", "hydro", "pI"]:
        names.append("mean_" + c)

    for c in ["max_{}", "min_{}", "max{}_b", "min{}_b", "max{}_y", "min{}_y"]:
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

    for pos in ["0", "1", "-2", "-1"]:
        for c in ["mz", "bas", "heli", "hydro", "pI", "P", "D", "E", "K", "R"]:
            names.append("loc_" + pos + "_" + c)

    for pos in ["i", "i+1"]:
        for c in ["P", "D", "E", "K", "R"]:
            names.append("loc_" + pos + "_" + c)

    for c in ["bas", "heli", "hydro", "pI", "mz"]:
        for pos in ["i", "i-1", "i+1", "i+2"]:
            names.append("loc_" + pos + "_" + c)

    names.append("charge")

    return names


def get_feature_names_chem(peplen):
    """
    feature names for the fixed peptide length feature vectors
    """
    aminos = ["A", "C", "D", "E", "F", "G", "H", "I", "K",
              "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

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
        rows = f.readlines(1000000)
        if not rows:
            break
        for row in rows:
            if row[0] == "T":
                if row[:5] == "TITLE":
                    titles.append(row.rstrip()[6:].replace(" ", ""))
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
    sys.stdout.write("{} spectra (~{:.2f} per cpu)\n".format(len(titles), np.mean([len(a) for a in split_titles])))

    return split_titles


def apply_mods(peptide, mods, PTMmap):
    """
    Takes a peptide sequence and a set of modifications. Returns the modified
    version of the peptide sequence, c- and n-term modifications. This modified
    version are hard coded in ms2pipfeatures_c.c for now.
    """
    modpeptide = np.array(peptide[:], dtype=np.uint16)

    nptm = 0
    cptm = 0
    if mods != "-":
        l = mods.split("|")
        for i in range(0, len(l), 2):
            tl = l[i + 1]
            if tl in PTMmap:
                modpeptide[int(l[i])] = PTMmap[tl]
            else:
                sys.stderr.write("Unknown modification: {}\n".format(tl))

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
    for i in range(len(pbuffer)):
        f.write(str.encode("{},1,{},{}\n".format(pbuffer[i][0], pbuffer[i][1], pbuffer[i][2])))
    f.close()

    f2 = tempfile.NamedTemporaryFile(delete=False, mode='wb')
    f2.write(str.encode("{}\n".format(len(spbuffer))))
    for i in range(len(spbuffer)):
        f2.write(str.encode("{},1,{},{}\n".format(spbuffer[i][0], spbuffer[i][1], spbuffer[i][2])))
    f2.close()

    return (f.name, f2.name, PTMmap)


def peakcount(x):
    c = 0.
    for i in x:
        if i > -9.95:
            c += 1.
    return c / len(x)


def calc_correlations(df):
    correlations = df.groupby(['spec_id', 'peplen', 'charge', 'ion'])[['target', 'prediction']].corr().iloc[::2]['prediction']
    correlations.index = correlations.index.droplevel(4)
    correlations = correlations.to_frame().reset_index()
    correlations.columns = ['spec_id', 'peplen', 'charge', 'ion', 'pearsonr']
    return correlations


def write_mgf(all_preds, output_filename):
    sys.stdout.write("writing mgf file {}_predictions.mgf...\n".format(output_filename))
    with open("{}_predictions.mgf".format(output_filename), "w+") as mgf_output:
        for sp in all_preds.spec_id.unique():
            tmp = all_preds[all_preds.spec_id == sp]
            tmp = tmp.sort_values("mz")
            mgf_output.write("BEGIN IONS\n")
            mgf_output.write("TITLE=" + str(sp) + "\n")
            mgf_output.write("CHARGE=" + str(tmp.charge[0]) + "\n")
            for i in range(len(tmp)):
                mgf_output.write(
                    str(tmp["mz"][i]) + " " + str(tmp["prediction"][i]) + "\n")
            mgf_output.write("END IONS\n")


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("pep_file", metavar="<peptide file>",
                        help="list of peptides")
    parser.add_argument("-c", metavar="FILE", action="store", dest="config_file", default="config.file",
                        help="config file (by default config.file)")
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


def run(pep_file, spec_file=None, vector_file=None, config_file=None, num_cpu=23, params=None, output_filename=None, datasetname=None, return_results=False):
    # datasetname is needed for Omega compatibility. This can be set to None if a config_file is provided

    # Create a_map:
    # a_map converts the peptide amino acids to integers, note how "L" is removed
    aminos = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "M", "N", "P", "Q",
              "R", "S", "T", "V", "W", "Y"]
    masses = [71.037114, 103.00919, 115.026943, 129.042593, 147.068414,
              57.021464, 137.058912, 113.084064, 128.094963, 131.040485,
              114.042927, 97.052764, 128.058578, 156.101111, 87.032028,
              101.047679, 99.068414, 186.079313, 163.063329, 147.0354]

    a_map = {}
    for i, a in enumerate(aminos):
        a_map[a] = i

    print_logo()

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

    if output_filename is None and not return_results:
        output_filename = '.'.join(pep_file.split('.')[:-1])

    # Check if given fragmethod exists:
    if fragmethod in ["CID", "HCD", "HCDiTRAQ4phospho", "HCDiTRAQ4", "ETD"]:
        print("Using {} models.\n".format(fragmethod))
    else:
        print("Unknown fragmentation method: {}".format(fragmethod))
        exit(1)

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
    data = pd.read_csv(pep_file,
                       sep=" ",
                       index_col=False,
                       dtype={"spec_id": str, "modifications": str})

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
        myPool.close()
        myPool.join()

        sys.stdout.write("merging results...\n")
        all_results = []
        for r in results:
            all_results.append(r.get())
        all_results = pd.concat(all_results)
        # """
        if vector_file:
            sys.stdout.write(
                "writing vector file {}... \n".format(vector_file))
            # write result. write format depends on extension:
            ext = vector_file.split(".")[-1]
            if ext == "pkl":
                all_results.to_pickle(vector_file + ".pkl")
            elif ext == "h5":
                all_results.to_hdf(vector_file, "table")
            # "table" is a tag used to read back the .h5
            else:  # if none of the two, default to .h5
                # all_results.to_hdf(vector_file, "table")
                all_results.to_csv(vector_file)
        else:
            sys.stdout.write("writing file {}_pred_and_emp.csv...\n".format(output_filename))
            all_results.to_csv("{}_pred_and_emp.csv".format(output_filename), index=False)
            sys.stdout.write('computing correlations...\n')
            correlations = calc_correlations(all_results)
            correlations.to_csv("{}_correlations.csv".format(output_filename), index=True)
            """
            corr_boxplot = correlations.plot('hist')
            corr_boxplot = corr_boxplot.get_figure()
            corr_boxplot.suptitle('Pearson corr for ' + spec_file + ' and predictions')
            corr_boxplot.savefig("{}_correlations.png".format(output_filename))
            """

        sys.stdout.write("done! \n")

    else:
        """
        If no mgf file is provided, MS2PIP will generate predicted spectra
        for each peptide in the pep_file
        """
        sys.stdout.write("scanning peptide file... ")

        titles = data.spec_id.tolist()
        split_titles = prepare_titles(titles, num_cpu)
        results = []

        for i in range(num_cpu):
            tmp = split_titles[i]
            results.append(myPool.apply_async(process_peptides, args=(
                i,
                data[data.spec_id.isin(tmp)],
                a_map, afile, modfile, modfile2, PTMmap, fragmethod)))
            """
            process_peptides(
                i,
                data[data.spec_id.isin(tmp)],
                a_map, afile, modfile, modfile2, PTMmap, fragmethod)
            """
        myPool.close()
        myPool.join()

        sys.stdout.write("merging results...\n")

        all_preds = pd.DataFrame()
        for r in results:
            all_preds = all_preds.append(r.get())

        mgf = False  # set to True to write spectrum as mgf file
        if mgf:
            write_mgf(all_preds, output_filename)

        if not return_results:
            sys.stdout.write("writing file {}_predictions.csv...\n".format(output_filename))
            all_preds.to_csv("{}_predictions.csv".format(output_filename), index=False)
            sys.stdout.write("done!\n")
        else:
            return(all_preds)


if __name__ == "__main__":
    pep_file, spec_file, vector_file, config_file, num_cpu = argument_parser()
    params = load_configfile(config_file)
    run(pep_file, spec_file=spec_file, vector_file=vector_file, params=params, num_cpu=num_cpu)
