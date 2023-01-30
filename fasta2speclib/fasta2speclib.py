"""
Create a spectral library starting from a proteome in fasta format.

The script runs through the following steps:
- In silico cleavage of proteins from the fasta file
- Remove peptide redundancy
- Add all variations of variable modifications (max 7 PTMs/peptide)
- Add variations on charge state
- Predict spectra with MS2PIP
- Write to various output file formats
"""


__author__ = "Ralf Gabriels"
__copyright__ = "CompOmics"
__credits__ = ["Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__email__ = "Ralf.Gabriels@ugent.be"


# Native libraries
import argparse
import json
import logging
import multiprocessing
from itertools import product
from math import ceil

# Third party libraries
import numpy as np
import pandas as pd
from Bio import SeqIO
from pyteomics.parser import cleave

from ms2pip.ms2pip_tools import spectrum_output
from ms2pip.ms2pip_tools.get_elude_predictions import get_elude_predictions

# MS2PIP
from ms2pip.ms2pipC import MS2PIP
from ms2pip.retention_time import RetentionTime


def ArgParse():
    parser = argparse.ArgumentParser(
        description="Create an MS2PIP-predicted spectral library, starting from a fasta file."
    )
    parser.add_argument(
        "fasta_filename",
        action="store",
        help="Path to the fasta file containing protein sequences",
    )
    parser.add_argument(
        "-o",
        dest="output_filename",
        action="store",
        help="Name for output file(s) (if not given, derived from input file)",
    )
    parser.add_argument(
        "-c",
        dest="config_filename",
        action="store",
        help="Name of configuration json file (default: fasta2speclib_config.json)",
    )

    args = parser.parse_args()
    return args


def get_params():
    args = ArgParse()

    if not args.config_filename:
        config_filename = "fasta2speclib_config.json"
    else:
        config_filename = args.config_filename

    with open(config_filename, "rt") as config_file:
        params = json.load(config_file)

    params.update(
        {
            "fasta_filename": args.fasta_filename,
            "log_level": logging.INFO,
        }
    )

    if args.output_filename:
        params["output_filename"] = args.output_filename
    else:
        params["output_filename"] = "_".join(
            params["fasta_filename"].split("\\")[-1].split(".")[:-1]
        )

    if not params["num_cpu"]:
        params["num_cpu"] = multiprocessing.cpu_count()

    return params


def prot_to_peprec(protein):
    """Cleave protein and return pd.DataFrame with valid peptides."""

    def validate_peptide(peptide, min_length, max_length):
        """Validate peptide by length and amino acids."""
        peplen = len(peptide)
        return (
            (peplen >= min_length)
            and (peplen <= max_length)
            and not any(aa in peptide for aa in ["B", "J", "O", "U", "X", "Z"])
        )

    params = get_params()

    pep_count = 0
    spec_ids = []
    peptides = []

    for peptide in cleave(
        str(protein.seq), params["cleavage_rule"], params["missed_cleavages"]
    ):
        pep_count += 1
        if validate_peptide(peptide, params["min_peplen"], params["max_peplen"]):
            spec_ids.append("{}_{:03d}".format(protein.id, pep_count))
            peptides.append(peptide)

    return pd.DataFrame(
        {
            "spec_id": spec_ids,
            "peptide": peptides,
            "modifications": "-",
            "charge": np.nan,
        }
    )


def get_protein_list(df):
    peptide_to_prot = {}
    for pi, pep in zip(df["spec_id"], df["peptide"]):
        pi = "_".join(pi.split("_")[0:2])
        if pep in peptide_to_prot.keys():
            peptide_to_prot[pep].append(pi)
        else:
            peptide_to_prot[pep] = [pi]
    df["protein_list"] = [list(set(peptide_to_prot[pep])) for pep in df["peptide"]]
    df = df[~df.duplicated(["peptide", "charge", "modifications"])]
    return df


def get_modifications_by_target(modifications):
    """Restructure modifications configuration to options per side chain or terminus."""
    mods_sidechain = {}
    mods_nterm = {}

    for mod in modifications:
        if mod["fixed"]:
            continue
        elif mod["n_term"]:
            if mod["amino_acid"]:
                if mod["amino_acid"] in mods_nterm:
                    mods_nterm[mod["amino_acid"]].append(mod["name"])
                else:
                    mods_nterm[mod["amino_acid"]] = [mod["name"]]
            else:
                if "any" in mods_nterm:
                    mods_nterm["any"].append(mod["name"])
                else:
                    mods_nterm["any"] = [mod["name"]]
        elif mod["amino_acid"]:
            if mod["amino_acid"] in mods_sidechain:
                mods_sidechain[mod["amino_acid"]].append(mod["name"])
            else:
                mods_sidechain[mod["amino_acid"]] = [mod["name"]]

    for aa, mods in mods_sidechain.items():
        mods.append(None)
    if "any" in mods_nterm:
        mods_nterm["any"].append(None)
    else:
        mods_nterm["any"] = [None]

    return mods_sidechain, mods_nterm


def get_modification_versions(
    peptide, modifications, mods_sidechain, mods_nterm, max_mods=3
):
    """Get MS²PIP modification strings for all potential versions."""
    possibilities_by_site = dict()

    # Assign possible modifications per residue (side chains)
    for pos, aa in enumerate(peptide):
        if aa in mods_sidechain:
            possibilities_by_site[pos + 1] = mods_sidechain[aa]

    # Assign possible modifications for N terminus
    if mods_nterm:
        possibilities_by_site[0] = []
    for site, mods in mods_nterm.items():
        if site == "any":
            possibilities_by_site[0].extend(mods)
        elif peptide[0] == site:
            possibilities_by_site[0].extend(mods)

    # Override with fixed modifications
    for mod in modifications:
        if not mod["fixed"]:
            continue
        if mod["n_term"]:
            if (mod["amino_acid"] and peptide[0] == mod["amino_acid"]) or not mod[
                "amino_acid"
            ]:
                possibilities_by_site[0] = [mod["name"]]
        elif mod["amino_acid"]:
            for pos, aa in enumerate(peptide):
                if aa == mod["amino_acid"]:
                    possibilities_by_site[pos] = [mod["name"]]

    # Get all possible combinations of modifications for all sites
    mod_permutations = list(product(*possibilities_by_site.values()))
    mod_positions = possibilities_by_site.keys()

    # Get MS²PIP modifications strings for each combination
    mod_strings = []
    for perm in mod_permutations:
        mods = sorted(zip(mod_positions, perm))  # Zip permutations with positions
        mods = "|".join(
            f"{m[0]}|{m[1]}" for m in mods if m[1]
        )  # Make str for modified sites
        mod_strings.append(mods)

    # Filter by max modified sites (avoiding combinatorial explosion)
    mod_strings = list(
        filter(lambda x: (x.count("|") + 1) / 2 <= max_mods, mod_strings)
    )

    return mod_strings


def add_mods(tup):
    """
    See fasta2speclib_config.md for more information.
    """
    _, row = tup
    params = get_params()

    # TODO: Do not hardcode max_mods
    # TODO: Do not include fixed modifications in max_mods
    mods_sidechain, mods_nterm = get_modifications_by_target(params["modifications"])
    mod_versions = get_modification_versions(
        row["peptide"], params["modifications"], mods_sidechain, mods_nterm, max_mods=3
    )

    df_out = pd.DataFrame(columns=row.index)
    df_out["modifications"] = ["-" if not mods else mods for mods in mod_versions]
    df_out["spec_id"] = [
        "{}_{:03d}".format(row["spec_id"], i) for i in range(len(mod_versions))
    ]
    df_out["charge"] = row["charge"]
    df_out["peptide"] = row["peptide"]
    if "protein_list" in row.index:
        df_out["protein_list"] = str(row["protein_list"])
    return df_out


def add_charges(df_in):
    params = get_params()
    df_out = pd.DataFrame(columns=df_in.columns)
    for charge in params["charges"]:
        tmp = df_in.copy()
        tmp["spec_id"] = tmp["spec_id"] + "_{}".format(charge)
        tmp["charge"] = charge
        df_out = pd.concat([df_out, tmp], axis=0, ignore_index=True)
    df_out.sort_values(["spec_id", "charge"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


def create_decoy_peprec(
    peprec,
    spec_id_prefix="decoy_",
    keep_cterm_aa=True,
    remove_redundancy=True,
    move_mods=True,
):
    """
    Create decoy peptides by reversing the sequences in a PEPREC DataFrame.

    Keyword arguments:
    spec_id_prefix -- string to prefix the decoy spec_ids (default: 'decoy_')
    keep_cterm_aa -- True if the last amino acid should stay in place (for example to keep tryptic properties) (default: True)
    remove_redundancy -- True if reversed peptides that are also found in the set of normal peptide should be removed (default: True)
    move_mods -- True to move modifications according to reversed sequence (default: True)

    Known issues:
    - C-terminal modifications (with position `-1`) are sorted to the front (eg: `-1|Cterm|0|Nterm|2|NormalPTM`).
    """

    def move_mods(row):
        mods = row["modifications"]
        if type(mods) == str:
            if not mods == "-":
                mods = mods.split("|")
                mods = sorted(
                    zip(
                        [
                            int(p)
                            if (p == "-1" or p == "0")
                            else len(row["peptide"]) - int(p)
                            for p in mods[::2]
                        ],
                        mods[1::2],
                    )
                )
                mods = "|".join(["|".join([str(x) for x in mod]) for mod in mods])
                row["modifications"] = mods
        return row

    peprec_decoy = peprec.copy()
    peprec_decoy["spec_id"] = spec_id_prefix + peprec_decoy["spec_id"].astype(str)

    if keep_cterm_aa:
        peprec_decoy["peptide"] = peprec_decoy["peptide"].apply(
            lambda pep: pep[-2::-1] + pep[-1]
        )
    else:
        peprec_decoy["peptide"] = peprec_decoy["peptide"].apply(lambda pep: pep[-1::-1])

    if remove_redundancy:
        peprec_decoy = peprec_decoy[~peprec_decoy["peptide"].isin(peprec["peptide"])]

    if "protein_list" in peprec_decoy.columns:
        peprec_decoy["protein_list"] = "decoy"

    if move_mods:
        peprec_decoy = peprec_decoy.apply(move_mods, axis=1)

    return peprec_decoy


def remove_from_peprec_filter(peprec_pred, peprec_filter):
    peprec_pred_comb = (
        peprec_pred["modifications"]
        + peprec_pred["peptide"]
        + peprec_pred["charge"].astype(str)
    )
    peprec_filter_comb = (
        peprec_filter["modifications"]
        + peprec_filter["peptide"]
        + peprec_filter["charge"].astype(str)
    )
    return peprec_pred[~peprec_pred_comb.isin(peprec_filter_comb)].copy()


def run_batches(peprec, decoy=False):
    params = get_params()
    if decoy:
        params["output_filename"] += "_decoy"

    ms2pip_params = {
        "ms2pip": {
            "model": params["ms2pip_model"],
            "frag_error": 0.02,
            # Modify fasta2speclib modifications dict to MS2PIP params PTMs entry
            "ptm": [
                "{},{},opt,{}".format(
                    mods["name"], mods["mass_shift"], mods["amino_acid"]
                )
                if not mods["n_term"]
                else "{},{},opt,N-term".format(mods["name"], mods["mass_shift"])
                for mods in params["modifications"]
            ],
            "sptm": [],
            "gptm": [],
        }
    }

    # If add_retention_time, initiate DeepLC
    if params["add_retention_time"]:
        logging.debug("Initializing DeepLC predictor")
        if "deeplc" not in params or not params["deeplc"]:
            params["deeplc"] = {"calibration_file": None}
        if not "n_jobs" in params["deeplc"]:
            params["deeplc"]["n_jobs"] = params["num_cpu"]
        rt_predictor = RetentionTime(config=params)

    # Split up into batches to save memory:
    b_size = params["batch_size"]
    b_count = 0
    num_b_counts = ceil(len(peprec) / b_size)
    for i in range(0, len(peprec), b_size):
        if i + b_size < len(peprec):
            peprec_batch = peprec[i : i + b_size]
        else:
            peprec_batch = peprec[i:]
        b_count += 1
        logging.info(
            "Predicting batch %d of %d, containing %d unmodified peptides",
            b_count,
            num_b_counts,
            len(peprec_batch),
        )

        logging.debug("Adding all modification combinations")
        peprec_mods = pd.DataFrame(columns=peprec_batch.columns)
        with multiprocessing.Pool(params["num_cpu"]) as p:
            peprec_mods = pd.concat(
                [peprec_mods] + p.map(add_mods, peprec_batch.iterrows()),
                ignore_index=True,
            )
        peprec_batch = peprec_mods

        if params["add_retention_time"]:
            logging.info("Adding DeepLC predicted retention times")
            rt_predictor.add_rt_predictions(peprec_batch)
        elif type(params["elude_model_file"]) == str:
            logging.debug("Adding ELUDE predicted retention times")
            peprec_batch["rt"] = get_elude_predictions(
                peprec_batch,
                params["elude_model_file"],
                unimod_mapping={
                    mod["name"]: mod["unimod_accession"]
                    for mod in params["modifications"]
                },
            )

        if type(params["rt_predictions_file"]) == str:
            logging.info("Adding RT predictions from file")
            rt_df = pd.read_csv(params["rt_predictions_file"])
            for col in ["peptide", "modifications", "rt"]:
                assert col in rt_df.columns, (
                    "RT file should contain a `%s` column" % col
                )
            peprec_batch = peprec_batch.merge(
                rt_df, on=["peptide", "modifications"], how="left"
            )
            assert (
                not peprec_batch["rt"].isna().any()
            ), "Not all required peptide-modification combinations could be found in RT file"

        logging.debug("Adding charge states %s", str(params["charges"]))
        peprec_batch = add_charges(peprec_batch)

        if type(params["peprec_filter"]) == str:
            logging.debug("Removing peptides present in peprec filter")
            peprec_filter = pd.read_csv(params["peprec_filter"], sep=" ")
            peprec_batch = remove_from_peprec_filter(peprec_batch, peprec_filter)

        if params["save_peprec"]:
            peprec_batch.to_csv(params["output_filename"] + "_" + str(b_count) + ".csv")

        logging.info("Running MS2PIP for %d peptides", len(peprec_batch))
        ms2pip = MS2PIP(
            peprec_batch,
            num_cpu=params["num_cpu"],
            output_filename=params["output_filename"],
            params=ms2pip_params,
            return_results=True,
        )
        all_preds = ms2pip.run()

        if b_count == 1:
            write_mode = "w"
            append = False
        else:
            write_mode = "a"
            append = True

        if "hdf" in params["output_filetype"]:
            logging.info(
                "Writing predictions to %s_predictions.hdf", params["output_filename"]
            )
            all_preds.astype(str).to_hdf(
                "{}_predictions.hdf".format(params["output_filename"]),
                key="table",
                format="table",
                complevel=3,
                complib="zlib",
                mode=write_mode,
                append=append,
                min_itemsize=50,
            )

        spec_out = spectrum_output.SpectrumOutput(
            all_preds,
            peprec_batch,
            ms2pip_params["ms2pip"],
            output_filename="{}".format(params["output_filename"]),
            write_mode=write_mode,
        )

        if "msp" in params["output_filetype"]:
            logging.info("Writing MSP file")
            spec_out.write_msp()

        if "mgf" in params["output_filetype"]:
            logging.info("Writing MGF file")
            spec_out.write_mgf()

        if "bibliospec" in params["output_filetype"]:
            logging.info("Writing BiblioSpec SSL and MS2 files")
            spec_out.write_bibliospec()

        if "spectronaut" in params["output_filetype"]:
            logging.info("Writing Spectronaut CSV file")
            spec_out.write_spectronaut()

        if "dlib" in params["output_filetype"]:
            logging.info("Writing DLIB SQLite file")
            spec_out.write_dlib()

        del all_preds
        del peprec_batch


def main():
    params = get_params()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=params["log_level"],
    )
    peprec = pd.DataFrame(columns=["spec_id", "peptide", "modifications", "charge"])

    logging.info("Cleaving proteins, adding peptides to peprec")
    with multiprocessing.Pool(params["num_cpu"]) as p:
        peprec = pd.concat(
            [peprec]
            + p.map(prot_to_peprec, SeqIO.parse(params["fasta_filename"], "fasta")),
            ignore_index=True,
        )

    logging.info("Removing peptide redundancy, adding protein list to peptides")
    peprec = get_protein_list(peprec)

    peprec_nonmod = peprec.copy()

    save_peprec = False
    if save_peprec:
        logging.info(
            "Saving non-expanded PEPREC to %s.peprec.hdf", params["output_filename"]
        )
        peprec_nonmod["protein_list"] = [
            "/".join(prot) for prot in peprec_nonmod["protein_list"]
        ]
        peprec_nonmod.astype(str).to_hdf(
            "{}_nonexpanded.peprec.hdf".format(params["output_filename"]),
            key="table",
            format="table",
            complevel=3,
            complib="zlib",
            mode="w",
        )

    if not params["decoy"]:
        del peprec_nonmod

    run_batches(peprec, decoy=False)

    if params["decoy"]:
        logging.info("Reversing sequences for decoy peptides")
        peprec_decoy = create_decoy_peprec(peprec_nonmod, move_mods=False)
        del peprec_nonmod

        logging.info("Predicting spectra for decoy peptides")
        run_batches(peprec_decoy, decoy=True)

    logging.info("fasta2speclib is ready!")


if __name__ == "__main__":
    main()
