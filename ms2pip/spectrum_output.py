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


from ms2pip.result import ProcessingResult

logger = logging.getLogger(__name__)


class InvalidWriteModeError(ValueError):
    pass


# Writer decorator
def writer(**kwargs):
    def deco(write_function):
        @wraps(write_function)
        def wrapper(self):
            return self._write_general(write_function, **kwargs)

        return wrapper

    return deco


def output_format(output_format):
    class OutputFormat:
        def __init__(self, fn):
            self.fn = fn
            self.output_format = output_format

        def __set_name__(self, owner, name):
            owner.OUTPUT_FORMATS[self.output_format] = self.fn
            setattr(owner, name, self.fn)

    return OutputFormat


class SpectrumOutput:
    """Write MS2PIP predictions to various output formats."""

    OUTPUT_FORMATS = {}

    def __init__(
        self,
        results: List["ProcessingResult"],
        output_filename="ms2pip_predictions",
        write_mode="wt+",
        return_stringbuffer=False,
        is_log_space=True,
        normalization="basepeak_10000",
    ):
        """
        Write MS2PIP predictions to various output formats.

        Parameters
        ----------
        results:
            List of ProcessingResult objects
        output_filename: str, optional
            path and name for output files, will be suffexed with `_predictions` and the
            relevant file extension (default: ms2pip_predictions)
        write_mode: str, optional
            write mode to use: "wt+" to append to start a new file, "at" to append to an
            existing file (default: "wt+")
        return_stringbuffer: bool, optional
            If True, files are written to a StringIO object, which the write function
            returns. If False, files are written to a file on disk.
        is_log_space: bool, optional
            Set to true if predicted intensities in `all_preds` are in log-space. In that
            case, intensities will first be transformed to "normal"-space.
        normalization: str, optional
            Normalization method to use. Options are "basepeak_10000", "basepeak_1", and
            "tic" (default: "basepeak_10000")

        Example
        -------
        >>> so = ms2pip.spectrum_tools.spectrum_output.SpectrumOutput(
                results
            )
        >>> so.write_msp()
        >>> so.write_spectronaut()

        """

        self.results = results
        self.output_filename = output_filename
        self.write_mode = write_mode
        self.return_stringbuffer = return_stringbuffer
        self.is_log_space = is_log_space
        self.normalization = normalization
        self.preds_dict = None
        # self.peprec_dict = None #TODO: Check if needed

        self.diff_modification_mapping = {}
        self.has_rt = hasattr(self.results[0], "predicted_rt") # Assuming all results have the same attributes
        self.has_protein_list = hasattr(self.results[0].psm, "protein_list") # Assuming all results have the same attributes


        if self.write_mode not in ["wt+", "wt", "at", "w", "a"]:
            raise InvalidWriteModeError(self.write_mode)

        if "a" in self.write_mode and self.return_stringbuffer:
            raise InvalidWriteModeError(self.write_mode)

    def _generate_preds_dict(self):
        """
        Create easy to access dict from ProcessingResult objects
        """
        self.preds_dict = {}

        for result in self.results:
            spec_id = result.psm_index
            if spec_id in self.preds_dict.keys():
                for ion_type in result.theoretical_mz.keys():
                    if ion_type in self.preds_dict[spec_id]["peaks"]:
                        for i in range(len(result.theoretical_mz[ion_type])):
                            self.preds_dict[spec_id]["peaks"][ion_type].append(
                                (
                                    result.theoretical_mz[ion_type][i],
                                    result.predicted_intensity[ion_type][i],
                                )
                            )
                    else:
                        self.preds_dict[spec_id]["peaks"][ion_type] = [
                            (
                                result.theoretical_mz[ion_type][i],
                                result.predicted_intensity[ion_type][i],
                            )
                        ]
            else:
                self.preds_dict[spec_id] = {
                    "peptidoform": result.psm.peptidoform,
                    "charge": result.psm.peptidoform.precursor_charge,
                    "peaks": {
                        ion_type: [
                            (
                                result.theoretical_mz[ion_type][i],
                                result.predicted_intensity[ion_type][i],
                            )
                            for i in range(len(result.theoretical_mz[ion_type]))
                        ]
                        for ion_type in result.theoretical_mz.keys()
                    },
                    "proteins": result.psm.protein_list
                }

    #TODO: Implement normalization
    def _normalize_spectra(self, method="basepeak_10000"):
        """
        Normalize spectra
        """
        if self.is_log_space:
            for result in self.results:
                result.predicted_intensity = {
                    ion_type: [
                    max(((2 ** intensity) - 0.001), 0) for intensity in intensities
                    ]
                    for ion_type, intensities in result.predicted_intensity.items()
                }
            self.is_log_space = False

        if method == "basepeak_10000":
            if self.normalization == "basepeak_10000":
                pass
            elif self.normalization == "basepeak_1":
                for result in self.results:
                    result.predicted_intensity = {
                        ion_type: [
                        intensity * 10000 for intensity in intensities
                        ]
                        for ion_type, intensities in result.predicted_intensity.items()
                    }
            else:
                for result in self.results:
                    for ion_type, intensities in result.predicted_intensity.items():
                        result.predicted_intensity[ion_type] = [
                            (intensity / max(intensities)) * 10000 for intensity in intensities
                        ]
            self.normalization = "basepeak_10000" #needed?

        elif method == "basepeak_1":
            if self.normalization == "basepeak_1":
                pass
            elif self.normalization == "basepeak_10000":
                for result in self.results:
                    result.predicted_intensity = {
                        ion_type: [
                        intensity / 10000 for intensity in intensities
                        ]
                        for ion_type, intensities in result.predicted_intensity.items()
                    }
            else:
                for result in self.results:
                    for ion_type, intensities in result.predicted_intensity.items():
                        result.predicted_intensity[ion_type] = [
                            intensity / max(intensities) for intensity in intensities
                        ]
            self.normalization = "base_peak_1" #needed?

        elif method == "tic":
            if self.normalization != "tic":
                for result in self.results:
                    for ion_type, intensities in result.predicted_intensity.items():
                        result.predicted_intensity[ion_type] = [
                            intensity / sum(intensities) for intensity in intensities
                        ]
            self.normalization = "tic" #needed?

        else:
            raise NotImplementedError

    def _get_msp_peak_annotation(
        self,
        peak_dict,
        sep="\t",
        include_zero=False,
        include_annotations=True,
        intensity_type=float,
    ):
        """
        Get MGF/MSP-like peaklist string
        """
        all_peaks = []
        for ion_type, peaks in peak_dict.items():

            for peak_number, peak in enumerate(peaks):
                if not include_zero and peak[1] == 0:
                    continue
                if include_annotations:
                    all_peaks.append(
                        (
                            peak[0],
                            f'{peak[0]:.6f}{sep}{intensity_type(peak[1])}{sep}"{ion_type.lower()}{peak_number+1}/0.0"',
                        )
                    )
                else:
                    all_peaks.append((peak[0], f"{peak[0]:.6f}{sep}{peak[1]}"))

        all_peaks = sorted(all_peaks, key=itemgetter(0))
        peak_string = "\n".join([peak[1] for peak in all_peaks])

        return peak_string

    def _get_msp_modifications(self, parsed_sequence, pep_properties):
        """
        Format modifications in MSP-style, e.g. "1/0,E,Glu->pyro-Glu" where 1 is the number of modifications,
        0 the position, E the amino acid, and Glu->pyro-Glu the modification.
        """
        mods = []
        counter = 0
        for index, (aa, mod) in enumerate(parsed_sequence):
            if not mod:
                continue
            else:
                counter += 1
                mods.append("{},{},{}".format(index+1, aa, mod[0].name)) #Assuming only one mod per amino acid, which I guess is okay?
        n_term_mod = pep_properties['n_term']
        c_term_mod = pep_properties['c_term']
        if n_term_mod:
            counter += 1
            mods.append("0,{},{}".format(parsed_sequence[0][0], n_term_mod))
        if c_term_mod:
            counter += 1
            mods.append("-1,{},{}".format(parsed_sequence[-1][0], c_term_mod))

        if counter == 0:
            return "0"
        else:
            return f"{counter}/{'/'.join(sorted(mods))}"

    def _parse_protein_string(self, protein_list):
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

    def _get_last_ssl_scannr(self):
        """
        Return scan number of last line in a Bibliospec SSL file.
        """
        ssl_filename = "{}_predictions.ssl".format(self.output_filename)
        with open(ssl_filename, "rt") as ssl:
            for line in ssl:
                last_line = line
            last_scannr = int(last_line.split("\t")[1])
        return last_scannr

    def _generate_diff_modification_mapping(self, precision):
        """
        Make modification name -> ssl modification name mapping.
        """
        self.diff_modification_mapping[precision] = {
            ptm.split(",")[0]: "{0:+.{1}f}".format(float(ptm.split(",")[1]), precision)
            for ptm in self.params["ptm"]
        }

    def _get_diff_modified_sequence(self, sequence, modifications, precision=1):
        """
        Build BiblioSpec SSL modified sequence string.
        """
        pep = list(sequence)
        mapping = self.diff_modification_mapping[precision]

        for loc, name in zip(modifications.split("|")[::2], modifications.split("|")[1::2]):
            # C-term mod
            if loc == "-1":
                pep[-1] = pep[-1] + "[{}]".format(mapping[name])
            # N-term mod
            elif loc == "0":
                pep[0] = pep[0] + "[{}]".format(mapping[name])
            # Normal mod
            else:
                pep[int(loc) - 1] = pep[int(loc) - 1] + "[{}]".format(mapping[name])
        return "".join(pep)

    def write_results(self, output_formats: List[str]) -> Dict[str, Any]:
        """
        Write MS2PIP predictions in output formats defined by output_formats.
        """
        results = {}
        for output_format in output_formats:
            output_format = output_format.lower()
            writer = self.OUTPUT_FORMATS[output_format]
            results[output_format] = writer(self)
        return results

    @output_format("msp")
    @writer(
        file_suffix="_predictions.msp",
        normalization_method="basepeak_10000",
        requires_dicts=True,
        requires_diff_modifications=False,
    )
    def write_msp(self, file_object):
        """
        Construct MSP string and write to file_object.
        """

        for spec_id in sorted(self.preds_dict.keys()):
            seq = self.preds_dict[spec_id]["peptidoform"].sequence
            pep_parsed_seq = self.preds_dict[spec_id]["peptidoform"].parsed_sequence
            pep_properties = self.preds_dict[spec_id]["peptidoform"].properties

            charge = self.preds_dict[spec_id]["charge"]
            prec_mz = self.preds_dict[spec_id]["peptidoform"].theoretical_mz
            prec_mass = self.preds_dict[spec_id]["peptidoform"].theoretical_mass
            msp_modifications = self._get_msp_modifications(pep_parsed_seq, pep_properties)
            num_peaks = sum(
                [len(peaklist) for _, peaklist in self.preds_dict[spec_id]["peaks"].items()]
            )

            comment_line = f" Mods={msp_modifications} Parent={prec_mz}"

            if self.has_protein_list:
                protein_list = self.preds_dict[spec_id]["proteins"]
                protein_string = self._parse_protein_string(protein_list)
                comment_line += f' Protein="{protein_string}"'

            if self.has_rt:
                rt = self.preds_dict[spec_id]['predicted_rt']
                comment_line += f" RetentionTime={rt}"

            comment_line += f' MS2PIP_ID="{spec_id}"'

            out = [
                f"Name: {seq}/{charge}",
                f"MW: {prec_mass}",
                f"Comment:{comment_line}",
                f"Num peaks: {num_peaks}",
                self._get_msp_peak_annotation(
                    self.preds_dict[spec_id]["peaks"],
                    sep="\t",
                    include_annotations=True,
                    intensity_type=int,
                ),
            ]

            file_object.writelines([line + "\n" for line in out] + ["\n"])

    @output_format("mgf")
    @writer(
        file_suffix="_predictions.mgf",
        normalization_method="basepeak_10000",
        requires_dicts=True,
        requires_diff_modifications=False,
    )
    def write_mgf(self, file_object):
        """
        Construct MGF string and write to file_object
        """
        for spec_id in sorted(self.peprec_dict.keys()):
            seq = self.peprec_dict[spec_id]["peptide"]
            mods = self.peprec_dict[spec_id]["modifications"]
            charge = self.peprec_dict[spec_id]["charge"]
            _, prec_mz = self.mods.calc_precursor_mz(seq, mods, charge)
            msp_modifications = self._get_msp_modifications(seq, mods)

            if self.has_protein_list:
                protein_list = self.peprec_dict[spec_id]["protein_list"]
                protein_string = self._parse_protein_string(protein_list)
            else:
                protein_string = ""

            out = [
                "BEGIN IONS",
                f"TITLE={spec_id} {seq}/{charge} {msp_modifications} {protein_string}",
                f"PEPMASS={prec_mz}",
                f"CHARGE={charge}+",
            ]

            if self.has_rt:
                rt = self.peprec_dict[spec_id]["rt"]
                out.append(f"RTINSECONDS={rt}")

            out.append(
                self._get_msp_peak_annotation(
                    self.preds_dict[spec_id]["peaks"],
                    sep=" ",
                    include_annotations=False,
                )
            )
            out.append("END IONS\n")
            file_object.writelines([line + "\n" for line in out])

    @output_format("spectronaut")
    @writer(
        file_suffix="_predictions_spectronaut.csv",
        normalization_method="tic",
        requires_dicts=False,
        requires_diff_modifications=True,
    )
    def write_spectronaut(self, file_obj):
        """
        Construct spectronaut DataFrame and write to file_object.
        """
        if "w" in self.write_mode:
            header = True
        elif "a" in self.write_mode:
            header = False
        else:
            raise InvalidWriteModeError(self.write_mode)

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
        # if requires_dicts and not self.peprec_dict:
        #     self._generate_peprec_dict()

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

    @output_format("bibliospec")
    def write_bibliospec(self):
        """Write MS2PIP predictions to BiblioSpec/Skyline SSL and MS2 spectral library files."""
        precision = 1
        if precision not in self.diff_modification_mapping:
            self._generate_diff_modification_mapping(precision)

        # Normalize if necessary and make dicts
        if not self.normalization == "basepeak_10000":
            self._normalize_spectra(method="basepeak_10000")
            self._generate_preds_dict()
        elif not self.preds_dict:
            self._generate_preds_dict()
        if not self.peprec_dict:
            self._generate_peprec_dict()

        if self.return_stringbuffer:
            file_obj_ssl = StringIO()
            file_obj_ms2 = StringIO()
        else:
            file_obj_ssl = open("{}_predictions.ssl".format(self.output_filename), self.write_mode)
            file_obj_ms2 = open("{}_predictions.ms2".format(self.output_filename), self.write_mode)

        # If a new file is written, write headers
        if "w" in self.write_mode:
            start_scannr = 0
            ssl_header = [
                "file",
                "scan",
                "charge",
                "sequence",
                "score-type",
                "score",
                "retention-time",
                "\n",
            ]
            file_obj_ssl.write("\t".join(ssl_header))
            file_obj_ms2.write(
                "H\tCreationDate\t{}\n".format(strftime("%Y-%m-%d %H:%M:%S", localtime()))
            )
            file_obj_ms2.write("H\tExtractor\tMS2PIP predictions\n")
        else:
            # Get last scan number of ssl file, to continue indexing from there
            # because Bibliospec speclib scan numbers can only be integers
            start_scannr = self._get_last_ssl_scannr() + 1

        self._write_bibliospec_core(file_obj_ssl, file_obj_ms2, start_scannr=start_scannr)

        return file_obj_ssl, file_obj_ms2

    def _write_dlib_metadata(self, connection):
        from sqlalchemy import select

        from ms2pip._utils.dlib import DLIB_VERSION, Metadata

        with connection.begin():
            version = connection.execute(
                select([Metadata.c.Value]).where(Metadata.c.Key == "version")
            ).scalar()
            if version is None:
                connection.execute(
                    Metadata.insert().values(
                        Key="version",
                        Value=DLIB_VERSION,
                    )
                )

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

    @output_format("dlib")
    def write_dlib(self):
        """Write MS2PIP predictions to a DLIB SQLite file."""
        from ms2pip._utils.dlib import metadata, open_sqlite

        normalization = "basepeak_10000"
        precision = 5
        if not self.normalization == normalization:
            self._normalize_spectra(method=normalization)
            self._generate_preds_dict()
        if not self.peprec_dict:
            self._generate_peprec_dict()
        if precision not in self.diff_modification_mapping:
            self._generate_diff_modification_mapping(precision)

        filename = "{}.dlib".format(self.output_filename)
        logger.info("Writing results to %s", filename)

        logger.debug(
            "write mode is ignored for DLIB at the file mode, although append or not is respected"
        )
        if "a" not in self.write_mode and os.path.exists(filename):
            os.remove(filename)

        if self.return_stringbuffer:
            raise NotImplementedError("`return_stringbuffer` not implemented for DLIB output.")

        if not self.has_rt:
            raise NotImplementedError("Retention times required to write DLIB file.")

        with open_sqlite(filename) as connection:
            metadata.create_all()
            self._write_dlib_metadata(connection)
            peptide_to_proteins = self._write_dlib_entries(connection, precision)
            self._write_dlib_peptide_to_protein(connection, peptide_to_proteins)

    def get_normalized_predictions(self, normalization_method="tic"):
        """Return normalized copy of predictions."""
        self._normalize_spectra(method=normalization_method)
        return self.all_preds.copy()

    @output_format("csv")
    def write_csv(self):
        """Write MS2PIP predictions to CSV."""

        self._normalize_spectra(method="tic")

        # Write to file or stringbuffer
        if self.return_stringbuffer:
            file_object = StringIO()
            logger.info("Writing results to StringIO using %s", "write_csv")
        else:
            f_name = "{}_predictions.csv".format(self.output_filename)
            file_object = open(f_name, self.write_mode)
            logger.info("Writing results to %s", f_name)

        try:
            self.all_preds.to_csv(
                file_object, float_format="%.6g", index=False, lineterminator="\n"
            )
        except TypeError:  # Pandas < 1.5 (Required for Python 3.7 support)
            self.all_preds.to_csv(
                file_object, float_format="%.6g", index=False, line_terminator="\n"
            )
        return file_object


def write_single_spectrum_csv(spectrum, filepath):
    """Write a single spectrum to a CSV file."""
    with open(filepath, "wt") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")
        writer.writerow(["mz", "intensity", "annotation"])
        for mz, intensity, annotation in zip(
            spectrum.mz,
            spectrum.intensity,
            spectrum.annotations,
        ):
            writer.writerow([mz, intensity, annotation])


def write_single_spectrum_png(spectrum, filepath):
    """Plot a single spectrum and write to a PNG file."""
    import matplotlib.pyplot as plt
    import spectrum_utils.plot as sup

    ax = plt.gca()
    ax.set_title("MS²PIP prediction for " + str(spectrum.peptidoform))
    sup.spectrum(spectrum.to_spectrum_utils(), ax=ax)
    plt.savefig(Path(filepath).with_suffix(".png"))
    plt.close()
