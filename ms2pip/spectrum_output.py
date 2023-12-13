"""
Write spectrum files from MS2PIP predictions.
"""
from __future__ import annotations

import csv
import itertools
import logging
import os
from ast import literal_eval
from functools import wraps
from io import StringIO
from operator import itemgetter
from pathlib import Path
from time import localtime, strftime
from typing import Any, Dict, List

import numpy as np

from ms2pip.result import ProcessingResult

logger = logging.getLogger(__name__)


class InvalidWriteModeError(ValueError):
    pass


    # def _generate_peprec_dict(self, rt_to_seconds=True):
    #     """
    #     Create easy to access dict from all_preds and peprec dataframes
    #     """
    #     peprec_tmp = self.peprec.copy()

    #     if self.has_rt and rt_to_seconds:
    #         peprec_tmp["rt"] = peprec_tmp["rt"] * 60

    #     peprec_tmp.index = peprec_tmp["spec_id"]
    #     peprec_tmp.drop("spec_id", axis=1, inplace=True)

    #     self.peprec_dict = peprec_tmp.to_dict(orient="index")

    # def _generate_preds_dict(self):
    #     """
    #     Create easy to access dict from results data.
    #     """
    #     self.preds_dict = {}

    #     for result in self.results:
    #         spec_id = result.psm_index
    #         if spec_id in self.preds_dict.keys():
    #             for ion, ion_numbers in result.theoretical_mz.items():
    #                 for i, ion_number in enumerate(ion_numbers):
    #                     mz = result.theoretical_mz[ion][i]
    #                     prediction = result.predicted_intensity[ion][i] if result.predicted_intensity else None

    #                     if ion in self.preds_dict[spec_id]["peaks"]:
    #                         self.preds_dict[spec_id]["peaks"][ion].append((ion_number, mz, prediction))
    #                     else:
    #                         self.preds_dict[spec_id]["peaks"][ion] = [(ion_number, mz, prediction)]
    #         else:
    #             self.preds_dict[spec_id] = {
    #                 "peaks": {}
    #             }

    #             if result.psm and result.psm.peptidoform:
    #                 self.preds_dict[spec_id]["charge"] = result.psm.peptidoform.precursor_charge
            
def _tic_normalize(intensities: np.array):
    """Normalize intensities to total ion current (TIC)."""
    return intensities / intensities.sum()

def _basepeak_normalize(intensities: np.array):
    """Normalize intensities to most intense peak."""
    return intensities / intensities.max()

def _get_msp_modifications(sequence, modifications):
    """
    Format modifications in MSP-style, e.g. "1/0,E,Glu->pyro-Glu"
    """

    if isinstance(modifications, str):
        if not modifications or modifications == "-":
            msp_modifications = "0"
        else:
            mods = modifications.split("|")
            mods = [(int(mods[i]), mods[i + 1]) for i in range(0, len(mods), 2)]
            mods = [(x, y) if x == 0 else (x - 1, y) for (x, y) in mods]
            mods = sorted(mods)
            mods = [(str(x), sequence[x], y) for (x, y) in mods]
            msp_modifications = "/".join([",".join(list(x)) for x in mods])
            msp_modifications = f"{len(mods)}/{msp_modifications}"
    else:
        msp_modifications = "0"

    return msp_modifications

def _parse_protein_string(protein_list):
    """
    Parse protein string from list, list string literal, or string.
    """
    if isinstance(protein_list, list):
        protein_string = "/".join(protein_list)
    elif isinstance(protein_list, str):
        try:
            protein_string = "/".join(literal_eval(protein_list))
        except ValueError:
            protein_string = protein_list
    else:
        protein_string = ""
    return protein_string

    # def _get_last_ssl_scannr(self):
    #     """
    #     Return scan number of last line in a Bibliospec SSL file.
    #     """
    #     ssl_filename = "{}_predictions.ssl".format(self.output_filename)
    #     with open(ssl_filename, "rt") as ssl:
    #         for line in ssl:
    #             last_line = line
    #         last_scannr = int(last_line.split("\t")[1])
    #     return last_scannr

    # def _generate_diff_modification_mapping(self, precision):
    #     """
    #     Make modification name -> ssl modification name mapping.
    #     """
    #     self.diff_modification_mapping[precision] = {
    #         ptm.split(",")[0]: "{0:+.{1}f}".format(float(ptm.split(",")[1]), precision)
    #         for ptm in self.params["ptm"]
    #     }

    # def _get_diff_modified_sequence(self, sequence, modifications, precision=1):
    #     """
    #     Build BiblioSpec SSL modified sequence string.
    #     """
    #     pep = list(sequence)
    #     mapping = self.diff_modification_mapping[precision]

    #     for loc, name in zip(modifications.split("|")[::2], modifications.split("|")[1::2]):
    #         # C-term mod
    #         if loc == "-1":
    #             pep[-1] = pep[-1] + "[{}]".format(mapping[name])
    #         # N-term mod
    #         elif loc == "0":
    #             pep[0] = pep[0] + "[{}]".format(mapping[name])
    #         # Normal mod
    #         else:
    #             pep[int(loc) - 1] = pep[int(loc) - 1] + "[{}]".format(mapping[name])
    #     return "".join(pep)

    # @output_format("msp")
    # @writer(
    #     file_suffix="_predictions.msp",
    #     normalization_method="basepeak_10000",
    #     requires_dicts=True,
    #     requires_diff_modifications=False,    
    # )


def write_msp(processing_results: List[ProcessingResult], file_object):
    """Construct MSP string and write to file_object."""
    for result in processing_results:
        sequence = result.psm.peptidoform.sequence
        charge = result.psm.peptidoform.precursor_charge
        mw = result.psm.peptidoform.theoretical_mass
        result_spectrum = result.as_spectra()[0]
        num_peaks = len(result_spectrum.mz)
        intensity_normalized = _basepeak_normalize(result_spectrum.intensity) * 10000

        peaks = zip(result_spectrum.mz, intensity_normalized, result_spectrum.annotations)

        # TODO: change modification info for comment line to use result
        # comment_line = f" Mods={msp_modifications} Parent={prec_mz}"

        # if self.has_protein_list:
        #     protein_list = self.peprec_dict[spec_id]["protein_list"]
        #     protein_string = self._parse_protein_string(protein_list)
        #     comment_line += f' Protein="{protein_string}"'

        # if self.has_rt:
        #     rt = self.peprec_dict[spec_id]["rt"]
        #     comment_line += f" RetentionTime={rt}"

        # comment_line += f' MS2PIP_ID="{spec_id}"'

        out = [
            f"Name: {sequence}/{charge}",
            f"MW: {mw}",
            # f"Comment:{comment_line}",
            f"Num peaks: {num_peaks}",
            [f"{mz}\t{intensity}\t{annotation}/0.0\n" for mz, intensity, annotation in peaks]
        ]
        file_object.writelines(''.join(line) if isinstance(line, list) else line + "\n" for line in out)
        file_object.write('\n')

def write_mgf(processing_results: List[ProcessingResult], file_object):
    """
    Construct MGF string and write to file_object
    """
    
    for result in processing_results:

        sequence = result.psm.peptidoform.sequence
        charge = result.psm.peptidoform.precursor_charge
        mw = result.psm.peptidoform.theoretical_mass

        result_spectrum = result.as_spectra()[0]
        intensity_normalized = _basepeak_normalize(result_spectrum.intensity) * 10000

        peaks = zip(result_spectrum.mz, intensity_normalized)

        out = [
            "BEGIN IONs",
            f"TITLE={result.psm_index} {sequence}/{charge}",
            f"PEPMASS={mw}",
            f"CHARGE={charge}+",
            [f"{mz}\t{intensity}\n" for mz, intensity in peaks],
            "END IONs\n"
        ]
        file_object.writelines(''.join(line) if isinstance(line, list) else line + "\n" for line in out)


    # for spec_id in sorted(self.peprec_dict.keys()):
    #     seq = self.peprec_dict[spec_id]["peptide"]
    #     mods = self.peprec_dict[spec_id]["modifications"]
    #     charge = self.peprec_dict[spec_id]["charge"]
    #     _, prec_mz = self.mods.calc_precursor_mz(seq, mods, charge)
    #     msp_modifications = self._get_msp_modifications(seq, mods)

    #     if self.has_protein_list:
    #         protein_list = self.peprec_dict[spec_id]["protein_list"]
    #         protein_string = self._parse_protein_string(protein_list)
    #     else:
    #         protein_string = ""

    #     out = [
    #         "BEGIN IONS",
    #         f"TITLE={spec_id} {seq}/{charge} {msp_modifications} {protein_string}",
    #         f"PEPMASS={prec_mz}",
    #         f"CHARGE={charge}+",
    #     ]

    #     if self.has_rt:
    #         rt = self.peprec_dict[spec_id]["rt"]
    #         out.append(f"RTINSECONDS={rt}")

    #     out.append(
    #         self._get_msp_peak_annotation(
    #             self.preds_dict[spec_id]["peaks"],
    #             sep=" ",
    #             include_annotations=False,
    #         )
    #     )
    #     out.append("END IONS\n")
    #     file_object.writelines([line for line in out] + ["\n"])

# @output_format("spectronaut")
# @writer(
#     file_suffix="_predictions_spectronaut.csv",
#     normalization_method="tic",
#     requires_dicts=False,
#     requires_diff_modifications=True,
# )
def write_spectronaut(processing_results: List[ProcessingResult], file_obj, header: bool):
    """
    Construct spectronaut DataFrame and write to file_object.
    """
    # if "w" in self.write_mode:
    #     header = True
    # elif "a" in self.write_mode:
    #     header = False
    # else:
    #     raise InvalidWriteModeError(self.write_mode)

    spectronaut_peprec = self.peprec.copy()

    # ModifiedPeptide and PrecursorMz columns
    spectronaut_peprec["ModifiedPeptide"] = spectronaut_peprec.apply(
        lambda row: self._get_diff_modified_sequence(row["peptide"], row["modifications"]),
        axis=1,
    )
    spectronaut_peprec["PrecursorMz"] = spectronaut_peprec.apply(
        lambda row: self.mods.calc_precursor_mz(
            row["peptide"], row["modifications"], row["charge"]
        )[1],
        axis=1,
    )
    spectronaut_peprec["ModifiedPeptide"] = "_" + spectronaut_peprec["ModifiedPeptide"] + "_"

    # Additional columns
    spectronaut_peprec["FragmentLossType"] = "noloss"

    # Retention time
    if "rt" in spectronaut_peprec.columns:
        rt_cols = ["iRT"]
        spectronaut_peprec["iRT"] = spectronaut_peprec["rt"]
    else:
        rt_cols = []

    # ProteinId
    if self.has_protein_list:
        spectronaut_peprec["ProteinId"] = spectronaut_peprec["protein_list"].apply(
            self._parse_protein_string
        )
    else:
        spectronaut_peprec["ProteinId"] = spectronaut_peprec["spec_id"]

    # Rename columns and merge with predictions
    spectronaut_peprec = spectronaut_peprec.rename(
        columns={"charge": "PrecursorCharge", "peptide": "StrippedPeptide"}
    )
    peptide_cols = (
        [
            "ModifiedPeptide",
            "StrippedPeptide",
            "PrecursorCharge",
            "PrecursorMz",
            "ProteinId",
        ]
        + rt_cols
        + ["FragmentLossType"]
    )
    spectronaut_df = spectronaut_peprec[peptide_cols + ["spec_id"]]
    spectronaut_df = self.all_preds.merge(spectronaut_df, on="spec_id")

    # Fragment columns
    spectronaut_df["FragmentCharge"] = (
        spectronaut_df["ion"].str.contains("2").map({True: 2, False: 1})
    )
    spectronaut_df["FragmentType"] = spectronaut_df["ion"].str[0].str.lower()

    # Rename and sort columns
    spectronaut_df = spectronaut_df.rename(
        columns={
            "mz": "FragmentMz",
            "prediction": "RelativeIntensity",
            "ionnumber": "FragmentNumber",
        }
    )
    fragment_cols = [
        "FragmentCharge",
        "FragmentMz",
        "RelativeIntensity",
        "FragmentType",
        "FragmentNumber",
    ]
    spectronaut_df = spectronaut_df[peptide_cols + fragment_cols]
    try:
        spectronaut_df.to_csv(
            file_obj, index=False, header=header, sep=";", lineterminator="\n"
        )
    except TypeError:  # Pandas < 1.5 (Required for Python 3.7 support)
        spectronaut_df.to_csv(
            file_obj, index=False, header=header, sep=";", line_terminator="\n"
        )

    return file_obj

def _write_bibliospec_core(self, file_obj_ssl, file_obj_ms2, start_scannr=0):
    """Construct Bibliospec SSL/MS2 strings and write to file_objects."""

    for i, spec_id in enumerate(sorted(self.preds_dict.keys())):
        scannr = i + start_scannr
        seq = self.peprec_dict[spec_id]["peptide"]
        mods = self.peprec_dict[spec_id]["modifications"]
        charge = self.peprec_dict[spec_id]["charge"]
        prec_mass, prec_mz = self.mods.calc_precursor_mz(seq, mods, charge)
        ms2_filename = os.path.basename(self.output_filename) + "_predictions.ms2"

        peaks = self._get_msp_peak_annotation(
            self.preds_dict[spec_id]["peaks"],
            sep="\t",
            include_annotations=False,
        )

        if isinstance(mods, str) and mods != "-" and mods != "":
            mod_seq = self._get_diff_modified_sequence(seq, mods)
        else:
            mod_seq = seq

        rt = self.peprec_dict[spec_id]["rt"] if self.has_rt else ""

        # TODO: implement csv instead of manual writing
        file_obj_ssl.write(
            "\t".join([ms2_filename, str(scannr), str(charge), mod_seq, "", "", str(rt)])
            + "\n"
        )
        file_obj_ms2.write(
            "\n".join(
                [
                    f"S\t{scannr}\t{prec_mz}",
                    f"Z\t{charge}\t{prec_mass}",
                    f"D\tseq\t{seq}",
                    f"D\tmodified seq\t{mod_seq}",
                    peaks,
                ]
            )
            + "\n"
        )

def _write_general(
    self,
    write_function,
    file_suffix,
    normalization_method,
    requires_dicts,
    requires_diff_modifications,
    diff_modification_precision=1,
):
    """
    General write function to call core write functions.

    Note: Does not work for write_bibliospec and write_dlib functions.
    """

    # Normalize if necessary and make dicts
    if not self.normalization == normalization_method:
        self._normalize_spectra(method=normalization_method)
        if requires_dicts:
            self._generate_preds_dict()
    elif requires_dicts and not self.preds_dict:
        self._generate_preds_dict()
    if requires_dicts and not self.peprec_dict:
        self._generate_peprec_dict()

    if (
        requires_diff_modifications
        and diff_modification_precision not in self.diff_modification_mapping
    ):
        self._generate_diff_modification_mapping(diff_modification_precision)

    # Write to file or stringbuffer
    if self.return_stringbuffer:
        file_object = StringIO()
        logger.info("Writing results to StringIO using %s", write_function.__name__)
    else:
        f_name = self.output_filename + file_suffix
        file_object = open(f_name, self.write_mode)
        logger.info("Writing results to %s", f_name)

    write_function(self, file_object)

    return file_object

# @output_format("bibliospec")
# def write_bibliospec(self):
#     """Write MS2PIP predictions to BiblioSpec/Skyline SSL and MS2 spectral library files."""
#     precision = 1
#     if precision not in self.diff_modification_mapping:
#         self._generate_diff_modification_mapping(precision)

#     # Normalize if necessary and make dicts
#     if not self.normalization == "basepeak_10000":
#         self._normalize_spectra(method="basepeak_10000")
#         self._generate_preds_dict()
#     elif not self.preds_dict:
#         self._generate_preds_dict()
#     if not self.peprec_dict:
#         self._generate_peprec_dict()

#     if self.return_stringbuffer:
#         file_obj_ssl = StringIO()
#         file_obj_ms2 = StringIO()
#     else:
#         file_obj_ssl = open("{}_predictions.ssl".format(self.output_filename), self.write_mode)
#         file_obj_ms2 = open("{}_predictions.ms2".format(self.output_filename), self.write_mode)

#     # If a new file is written, write headers
#     if "w" in self.write_mode:
#         start_scannr = 0
#         ssl_header = [
#             "file",
#             "scan",
#             "charge",
#             "sequence",
#             "score-type",
#             "score",
#             "retention-time",
#             "\n",
#         ]
#         file_obj_ssl.write("\t".join(ssl_header))
#         file_obj_ms2.write(
#             "H\tCreationDate\t{}\n".format(strftime("%Y-%m-%d %H:%M:%S", localtime()))
#         )
#         file_obj_ms2.write("H\tExtractor\tMS2PIP predictions\n")
#     else:
#         # Get last scan number of ssl file, to continue indexing from there
#         # because Bibliospec speclib scan numbers can only be integers
#         start_scannr = self._get_last_ssl_scannr() + 1

#     self._write_bibliospec_core(file_obj_ssl, file_obj_ms2, start_scannr=start_scannr)

#     return file_obj_ssl, file_obj_ms2

# def _write_dlib_metadata(self, connection):
#     from sqlalchemy import select

#     from ms2pip._utils.dlib import DLIB_VERSION, Metadata

#     with connection.begin():
#         version = connection.execute(
#             select([Metadata.c.Value]).where(Metadata.c.Key == "version")
#         ).scalar()
#         if version is None:
#             connection.execute(
#                 Metadata.insert().values(
#                     Key="version",
#                     Value=DLIB_VERSION,
#                 )
#             )

def _write_dlib_entries(self, connection, precision):
    from ms2pip._utils.dlib import Entry

    peptide_to_proteins = set()

    with connection.begin():
        for spec_id, peprec in self.peprec_dict.items():
            seq = peprec["peptide"]
            mods = peprec["modifications"]
            charge = peprec["charge"]

            prec_mass, prec_mz = self.mods.calc_precursor_mz(seq, mods, charge)
            mod_seq = self._get_diff_modified_sequence(seq, mods, precision=precision)

            all_peaks = sorted(
                itertools.chain.from_iterable(self.preds_dict[spec_id]["peaks"].values()),
                key=itemgetter(1),
            )
            mzs = [peak[1] for peak in all_peaks]
            intensities = [peak[2] for peak in all_peaks]

            connection.execute(
                Entry.insert().values(
                    PrecursorMz=prec_mz,
                    PrecursorCharge=charge,
                    PeptideModSeq=mod_seq,
                    PeptideSeq=seq,
                    Copies=1,
                    RTInSeconds=peprec["rt"],
                    Score=0,
                    MassEncodedLength=len(mzs),
                    MassArray=mzs,
                    IntensityEncodedLength=len(intensities),
                    IntensityArray=intensities,
                    SourceFile=self.output_filename,
                )
            )

            if self.has_protein_list:
                protein_list = peprec["protein_list"]
                if isinstance(protein_list, str):
                    protein_list = literal_eval(protein_list)

                for protein in protein_list:
                    peptide_to_proteins.add((seq, protein))

    return peptide_to_proteins

def _write_dlib_peptide_to_protein(self, connection, peptide_to_proteins):
    from ms2pip._utils.dlib import PeptideToProtein

    if not self.has_protein_list:
        return

    with connection.begin():
        sql_peptide_to_proteins = set()
        proteins = {protein for _, protein in peptide_to_proteins}
        for peptide_to_protein in connection.execute(
            PeptideToProtein.select().where(PeptideToProtein.c.ProteinAccession.in_(proteins))
        ):
            sql_peptide_to_proteins.add(
                (
                    peptide_to_protein.PeptideSeq,
                    peptide_to_protein.ProteinAccession,
                )
            )

        peptide_to_proteins.difference_update(sql_peptide_to_proteins)
        for seq, protein in peptide_to_proteins:
            connection.execute(
                PeptideToProtein.insert().values(
                    PeptideSeq=seq, isDecoy=False, ProteinAccession=protein
                )
            )

# @output_format("dlib")
# def write_dlib(self):
#     """Write MS2PIP predictions to a DLIB SQLite file."""
#     from ms2pip._utils.dlib import metadata, open_sqlite

#     normalization = "basepeak_10000"
#     precision = 5
#     if not self.normalization == normalization:
#         self._normalize_spectra(method=normalization)
#         self._generate_preds_dict()
#     if not self.peprec_dict:
#         self._generate_peprec_dict()
#     if precision not in self.diff_modification_mapping:
#         self._generate_diff_modification_mapping(precision)

#     filename = "{}.dlib".format(self.output_filename)
#     logger.info("Writing results to %s", filename)

#     logger.debug(
#         "write mode is ignored for DLIB at the file mode, although append or not is respected"
#     )
#     if "a" not in self.write_mode and os.path.exists(filename):
#         os.remove(filename)

#     if self.return_stringbuffer:
#         raise NotImplementedError("`return_stringbuffer` not implemented for DLIB output.")

#     if not self.has_rt:
#         raise NotImplementedError("Retention times required to write DLIB file.")

#     with open_sqlite(filename) as connection:
#         metadata.create_all()
#         self._write_dlib_metadata(connection)
#         peptide_to_proteins = self._write_dlib_entries(connection, precision)
#         self._write_dlib_peptide_to_protein(connection, peptide_to_proteins)

def write_csv(processing_results: List[ProcessingResult], file_obj):
    """Write processing results to CSV file."""
    fieldnames = [
        "spec_id",
        "charge",
        "ion",
        "ionnumber",
        "mz",
        "prediction",
        "rt"
    ]
    writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for result in processing_results:
        if result.theoretical_mz is not None:
            for ion_type in result.theoretical_mz:
                for i in range(len(result.theoretical_mz[ion_type])):
                    writer.writerow(
                        {
                            "spec_id": result.psm_index,
                            "charge": result.psm.peptidoform.precursor_charge,
                            "ion": ion_type,
                            "ionnumber": i + 1,
                            "mz": "{:.6g}".format(result.theoretical_mz[ion_type][i]),
                            "prediction": "{:.6g}".format(
                                result.predicted_intensity[ion_type][i]
                            ),
                            "rt": result.psm.retention_time # TODO: is always empty now
                        }
                    )


# def write_single_spectrum_csv(spectrum, filepath):
#     """Write a single spectrum to a CSV file."""
#     with open(filepath, "wt") as f:
#         writer = csv.writer(f, delimiter=",", lineterminator="\n")
#         writer.writerow(["mz", "intensity", "annotation"])
#         for mz, intensity, annotation in zip(
#             spectrum.mz,
#             spectrum.intensity,
#             spectrum.annotations,
#         ):
#             writer.writerow([mz, intensity, annotation])


# def write_single_spectrum_png(spectrum, filepath):
#     """Plot a single spectrum and write to a PNG file."""
#     import matplotlib.pyplot as plt
#     import spectrum_utils.plot as sup

#     ax = plt.gca()
#     ax.set_title("MS²PIP prediction for " + str(spectrum.peptidoform))
#     sup.spectrum(spectrum.to_spectrum_utils(), ax=ax)
#     plt.savefig(Path(filepath).with_suffix(".png"))
#     plt.close()
