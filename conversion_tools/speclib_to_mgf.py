#!/usr/bin/env python3
"""
Convert MSP and SPTXT spectral library files.

Writes three files: mgf with the spectra; PEPREC with the peptide sequences;
meta with additional metainformation.

Arguments:
    arg1 path to spectral library file
    arg2 prefix for spec_id

"""

import re
import sys
import logging


AMINO_MASSES = {
    "A": 71.037114,
    "C": 103.009185,
    "D": 115.026943,
    "E": 129.042593,
    "F": 147.068414,
    "G": 57.021464,
    "H": 137.058912,
    "I": 113.084064,
    "K": 128.094963,
    "L": 113.084064,
    "M": 131.040485,
    "N": 114.042927,
    "P": 97.052764,
    "Q": 128.058578,
    "R": 156.101111,
    "S": 87.032028,
    "T": 101.047679,
    "V": 99.068414,
    "W": 186.079313,
    "Y": 163.063329,
}
PROTON_MASS = 1.007825035
WATER_MASS = 18.010601


def setup_logging():
    """Initiate logging."""
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(module)s %(message)s")
    )
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def parse_peprec_mods(mods, ptm_list):
    """Parse PEPREC modification string out of MSP Mod string."""
    if mods.split("/")[0] != "0":
        num_mods = mods[0]
        mod_list = [mod.split(",") for mod in mods.split("/")[1:]]

        peprec_mods = []
        for location, aa, name in mod_list:
            if not (location == "0" and name == "iTRAQ"):
                location = str(int(location) + 1)
            peprec_mods.append(location)
            peprec_mods.append(name)

            if name not in ptm_list:
                ptm_list[name] = 1
            else:
                ptm_list[name] += 1

        peprec_mods = "|".join(peprec_mods)

    else:
        peprec_mods = "-"

    return peprec_mods


def validate(spec_id, peptide, charge, mods, reported_mw):
    """Validate amino acids and reported peptide mass."""
    invalid_aas = ["B", "J", "O", "U", "X", "Z"]
    if any(aa in invalid_aas for aa in peptide):
        logging.warning("Peptide with non-canonical amino acid found: %s", peptide)

    elif (
        mods.split("/")[0] == "0"
    ):  # Cannot validate mass of peptide with unknown modification
        calculated = WATER_MASS + sum([AMINO_MASSES[x] for x in peptide])
        reported = float(reported_mw) * float(charge) - float(charge) * PROTON_MASS
        if abs(calculated - reported) > 0.5:
            logging.warning(
                "Reported MW does not match calculated mass for spectrum %s", spec_id
            )


def parse_speclib(speclib_filename, title_prefix, speclib_format="msp"):
    """Parse MSP file."""
    filename = ".".join(speclib_filename.split(".")[:-1])
    fpip = open(filename + ".peprec", "w")
    fpip.write("spec_id modifications peptide charge\n")
    fmgf = open(filename + ".mgf", "w")
    fmeta = open(filename + ".meta", "w")

    with open(speclib_filename) as f:
        mod_dict = {}
        spec_id = 1
        peak_sep = None
        peptide = None
        charge = None
        parentmz = None
        mods = None
        purity = None
        HCDenergy = None
        read_spec = False
        mgf = ""

        for row in f:
            if read_spec:
                # Infer peak int/mz separator
                if not peak_sep:
                    if "\t" in row:
                        peak_sep = "\t"
                    elif " " in row:
                        peak_sep = " "
                    else:
                        raise ValueError("Invalid peak separator")

                line = row.rstrip().split(peak_sep)

                # Read all peaks, so save to output files and set read_spec to False
                if row[0].isdigit():
                    # Continue reading spectrum
                    mgf += " ".join([line[0], line[1]]) + "\n"
                    continue

                # Last peak reached, finish up spectrum
                else:
                    validate(spec_id, peptide, charge, mods, parentmz)
                    peprec_mods = parse_peprec_mods(mods, mod_dict)
                    fpip.write(
                        "{}{} {} {} {}\n".format(
                            title_prefix, spec_id, peprec_mods, peptide, charge
                        )
                    )
                    fmeta.write(
                        "{}{} {} {} {} {} {}\n".format(
                            title_prefix,
                            spec_id,
                            charge,
                            peptide,
                            parentmz,
                            purity,
                            HCDenergy,
                        )
                    )

                    buf = "BEGIN IONS\n"
                    buf += "TITLE=" + title_prefix + str(spec_id) + "\n"
                    buf += "CHARGE=" + str(charge) + "\n"
                    buf += "PEPMASS=" + parentmz + "\n"
                    fmgf.write("{}{}END IONS\n\n".format(buf, mgf))

                    spec_id += 1
                    read_spec = False
                    mgf = ""

            if row.startswith("Name:"):
                line = row.rstrip().split(" ")
                tmp = line[1].split("/")
                peptide = tmp[0].replace("(O)", "")
                if speclib_format == "sptxt":
                    peptide = re.sub(r"\[\d*\]|[a-z]", "", peptide)
                charge = tmp[1].split("_")[0]
                continue

            elif row.startswith("Comment:"):
                line = row.rstrip().split(" ")
                for i in range(1, len(line)):
                    if line[i].startswith("Mods="):
                        tmp = line[i].split("=")
                        mods = tmp[1]
                    if line[i].startswith("Parent="):
                        tmp = line[i].split("=")
                        parentmz = tmp[1]
                    if line[i].startswith("Purity="):
                        tmp = line[i].split("=")
                        purity = tmp[1]
                    if line[i].startswith("HCD="):
                        tmp = line[i].split("=")
                        HCDenergy = tmp[1].replace("eV", "")
                continue

            elif row.startswith("Num peaks:") or row.startswith("NumPeaks:"):
                read_spec = True
                continue

    fmgf.close()
    fpip.close()
    fmeta.close()

    return spec_id, mod_dict


def main():
    """Run CLI."""
    # Get arguments
    speclib_filename = sys.argv[1]
    title_prefix = sys.argv[2]

    speclib_ext = speclib_filename.split(".")[-1]
    if speclib_ext.lower() == "sptxt":
        speclib_format = "sptxt"
    elif speclib_ext.lower() == "msp":
        speclib_format = "msp"
    else:
        raise ValueError("Unknown spectral library format: `%s`" % speclib_ext)

    logging.info("Converting %s to MGF, PEPREC and meta file", speclib_filename)

    num_peptides, mod_dict = parse_speclib(
        speclib_filename, title_prefix, speclib_format=speclib_format
    )

    logging.info(
        "Finished!\nSpectral library contains %i peptides and the following modifications: %s",
        num_peptides,
        mod_dict,
    )


if __name__ == "__main__":
    setup_logging()
    main()
