"""
Extended PSM Report to PEPREC
extpsmreport_to_peprec.py

Create a PEPREC file out of a PeptideShaker Extended PSM Report. Attention: The
resulting PEPREC file will contain all PSMs and is not filtered on FDR or decoy
hits.
"""

__author__ = "Tim Van Den Bossche"
__credits__ = ["Tim Van Den Bossche", "Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__email__ = "Ralf.Gabriels@ugent.be"


# Native libraries
import argparse

# Third party libraries
import pandas as pd


def modifications(modified_seq):
    # initiate variables for nterm, seq and cterm
    mod_list = list()
    nterm, seq, cterm = modified_seq.split("-")

    # initiatle variable for nterm
    pyro_bool = False

    # initiate variables for seq
    mod_index = 0
    mod_description = False  # to check if it's an amino acid (False) or a description in < ... > (True)

    # check amino terminus for modifications
    if nterm == "ace":
        mod_list.append("0|Acetyl")
    elif nterm == "pyro":
        pyro_bool = True
    elif nterm != "NH2":
        print("Unknown N-terminal modification: {}".format(nterm))

    # check internal sequence
    for char in seq:
        if char == "<":
            mod_peprec = "{}|".format(mod_index)
            mod_name = ""
            mod_description = True
        elif char == ">":
            mod_description = False
            if mod_name == 'ox':
                mod_peprec += 'Oxidizedmethionine'  # allow only oxidation of Met!!
            elif mod_name == 'cmm':
                mod_peprec += 'Carbamidomethylation'
            elif mod_name == 'deam':
                mod_peprec += 'Deamidated'
            else:
                print("Unknown internal modification: {}".format(mod_name))
            mod_list.append("{}".format(mod_peprec))  # peprec format
            mod_peprec = ""

        else:
            if pyro_bool:
                if char == 'C':
                    mod_name = "Pyro-carbamidomethyl"
                elif char == 'Q':
                    mod_name = "Gln->pyro-Glu"
                elif char == 'E':
                    mod_name = "Glu->pyro-Glu"
                elif char == 'P':
                    mod_name = "Pro->pyro-Glu"
                else:
                    print("Unknown N-terminal pyro modification from {}".format(char))
                mod_list.append("1|{}".format(mod_name))
                pyro_bool = False
                mod_index += 1
                mod_name = ""
            else:
                if mod_description:
                    mod_name += char
                else:
                    mod_index += 1

    mods_peprec = "|".join(mod_list)
    if mods_peprec == "":
        mods_peprec = "-"
    return mods_peprec


def ArgParse():
    parser = argparse.ArgumentParser(description='Create a PEPREC file out of a PeptideShaker Extended PSM Report.')
    parser.add_argument("input", metavar="<extended_psm_report>",
                        help="Path to input Extended PSM Report")
    parser.add_argument('-o', '--out', dest='outname', action='store', default='SpecLib',
                        help='Name for output file (default: "SpecLib")')
    args = parser.parse_args()

    return(args)


def main():
    args = ArgParse()

    df = pd.read_excel(str(args.input))

    spec_id = list()
    mods_peprec = list()
    peptide = list()
    charge = list()
    label = list()

    for index, row in df.iterrows():
        spec_id.append(row["Spectrum Title"])
        mods_peprec.append(modifications(row["Modified Sequence"]))
        peptide.append(row["Sequence"])
        charge.append(row["Measured Charge"][:-1])
        if row["Decoy"] == 0:
            label.append(1)
        elif row["Decoy"] == 1:
            label.append(-1)
        else:
            print("No target/decoy label available for {}".format(row["Spectrum Title"]))

    d = {"spec_id": spec_id, "modifications": mods_peprec, "peptide": peptide, "charge": charge, "label":label}
    peprec_df = pd.DataFrame(data=d, columns=["spec_id", "modifications", "peptide", "charge", "label"])
    peprec_df.to_csv('{}.peprec'.format(args.outname), index=False, sep="\t", quotechar="'")

if __name__ == "__main__":
    main()
