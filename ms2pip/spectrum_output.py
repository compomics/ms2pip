"""
Write spectrum files from MS2PIP predictions.
"""

from __future__ import annotations

import csv
import itertools
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from io import StringIO
from pathlib import Path
from time import localtime, strftime
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
from psm_utils import PSM, Peptidoform
from pyteomics import proforma
from sqlalchemy import engine, select

from ms2pip._utils import dlib
from ms2pip.result import ProcessingResult


def _peptidoform_str_without_charge(peptidoform: Peptidoform) -> str:
    """Get peptidoform string without charge."""
    return re.sub(r"\/\d+$", "", str(peptidoform))


def _unlogarithmize(intensities: np.array) -> np.array:
    """Undo logarithmic transformation of intensities."""
    return (2**intensities) - 0.001


def _tic_normalize(intensities: np.array):
    """Normalize intensities to total ion current (TIC)."""
    return intensities / intensities.sum()


def _basepeak_normalize(intensities: np.array, basepeak: Optional[float] = None) -> np.array:
    """Normalize intensities to most intense peak."""
    if not basepeak:
        basepeak = intensities.max()
    return intensities / basepeak


class _Writer(ABC):
    """Abstract base class for writing spectrum files."""

    def __init__(self, file: Union[str, Path, StringIO], write_mode: str = "w"):
        self.ssl_file = file
        self.write_mode = write_mode

        self._open_file = None

    def __enter__(self):
        """Open file in context manager."""
        if isinstance(self.ssl_file, (str, Path)):
            self.ssl_file = Path(self.ssl_file)
            self._open_file = open(self.ssl_file, self.write_mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._open_file:
            self._open_file.close()

    @property
    def _file_object(self):
        """Get file object from file path or StringIO."""
        if self._open_file:
            return self._open_file
        else:
            if isinstance(self.ssl_file, StringIO):
                return self.ssl_file
            elif isinstance(self.ssl_file, (str, Path)):
                raise TypeError("Use `with` statement to open file from path.")
            else:
                raise TypeError("Unsupported type for `file`.")

    def write(self, processing_results: List[ProcessingResult]):
        """Write multiple processing results to file."""
        for result in processing_results:
            self._write_result(result)

    @abstractmethod
    def _write_result(self, result: ProcessingResult):
        """Write single processing result to file."""
        ...


class TSV(_Writer):
    """Write TSV files from MS2PIP processing results."""

    field_names = [
        "psm_index",
        "ion_type",
        "ion_number",
        "mz",
        "predicted",
        "observed",
        "rt",
    ]

    def write(self, processing_results: List[ProcessingResult]):
        """Write multiple processing results to file."""
        writer = csv.DictWriter(self._file_object, fieldnames=self.field_names, delimiter="\t")
        if self.write_mode == "w":
            writer.writeheader()
        for result in processing_results:
            self._write_result(result, writer)

    def _write_result(self, result: ProcessingResult, writer: csv.DictWriter):
        """Write single processing result to file."""
        # Only write results with predictions or observations
        if not result.theoretical_mz:
            return

        for ion_type in result.theoretical_mz:
            for i in range(len(result.theoretical_mz[ion_type])):
                writer.writerow(self._write_row(result, ion_type, i))

    @staticmethod
    def _write_row(result: ProcessingResult, ion_type: str, ion_index: int):
        """Write single row for TSV file."""
        return {
            "psm_index": result.psm_index,
            "ion_type": ion_type,
            "ion_number": ion_index + 1,
            "mz": "{:.10g}".format(result.theoretical_mz[ion_type][ion_index]),
            "predicted": "{:.10g}".format(result.predicted_intensity[ion_type][ion_index])
            if result.predicted_intensity
            else None,
            "observed": "{:.10g}".format(result.observed_intensity[ion_type][ion_index])
            if result.observed_intensity
            else None,
            "rt": result.psm.retention_time if result.psm.retention_time else None,
        }


class MSP(_Writer):
    """Write MSP files from MS2PIP processing results."""

    def write(self, results: List[ProcessingResult]):
        """Write multiple processing results to file."""
        for result in results:
            self._write_result(result)

    def _write_result(self, result: ProcessingResult):
        """Write single processing result to file."""
        predicted_spectrum = result.as_spectra()[0]
        intensity_normalized = _basepeak_normalize(predicted_spectrum.intensity) * 1e4
        peaks = zip(predicted_spectrum.mz, intensity_normalized, predicted_spectrum.annotations)

        # Header
        lines = [
            f"Name: {result.psm.peptidoform.sequence}/{result.psm.get_precursor_charge()}",
            f"MW: {result.psm.peptidoform.theoretical_mass}",
            self._format_comment_line(result.psm),
            f"Num peaks: {len(predicted_spectrum.mz)}",
        ]

        # Peaks
        lines.extend(
            f"{mz:.10g}\t{intensity:.10g}\t{annotation}/0.0" for mz, intensity, annotation in peaks
        )

        # Write to file
        self._file_object.writelines(line + "\n" for line in lines)
        self._file_object.write("\n")

    @staticmethod
    def _format_modifications(peptidoform: Peptidoform):
        """Format modifications in MSP-style string, e.g. ``Mods=1/0,E,Glu->pyro-Glu``."""

        def _format_single_modification(
            amino_acid: str,
            position: int,
            modifications: Optional[List[proforma.ModificationBase]],
        ) -> Union[str, None]:
            """Get modification label from :py:class:`proforma.ModificationBase` list."""
            if not modifications:
                return None
            if len(modifications) > 1:
                raise ValueError("Multiple modifications per amino acid not supported.")
            modification = modifications[0]
            return f"{position},{amino_acid},{modification.name}"

        sequence_mods = [
            _format_single_modification(aa, pos + 1, mods)
            for pos, (aa, mods) in enumerate(peptidoform.parsed_sequence)
        ]
        n_term = _format_single_modification(
            peptidoform.sequence[0], 0, peptidoform.properties["n_term"]
        )
        c_term = _format_single_modification(
            peptidoform.sequence[-1], -1, peptidoform.properties["c_term"]
        )

        mods = [mod for mod in [n_term] + sequence_mods + [c_term] if mod is not None]

        if not mods:
            return "Mods=0"
        else:
            return f"Mods={len(mods)}/{'/'.join(sorted(mods))}"

    @staticmethod
    def _format_parent_mass(peptidoform: Peptidoform) -> str:
        """Format parent mass as string."""
        return f"Parent={peptidoform.theoretical_mz}"

    @staticmethod
    def _format_protein_string(psm: PSM) -> Union[str, None]:
        """Format protein list as string."""
        if psm.protein_list:
            return f"Protein={','.join(psm.protein_list)}"
        else:
            return None

    @staticmethod
    def _format_retention_time(psm: PSM) -> Union[str, None]:
        """Format retention time as string."""
        if psm.retention_time:
            return f"RetentionTime={psm.retention_time}"
        else:
            return None

    @staticmethod
    def _format_identifier(psm: PSM) -> str:
        """Format MS2PIP ID as string."""
        return f"SpectrumIdentifier={psm.spectrum_id}"

    @staticmethod
    def _format_comment_line(psm: PSM) -> str:
        """Format comment line for MSP file."""
        comments = " ".join(
            filter(
                None,
                [
                    MSP._format_modifications(psm.peptidoform),
                    MSP._format_parent_mass(psm.peptidoform),
                    MSP._format_protein_string(psm),
                    MSP._format_retention_time(psm),
                    MSP._format_identifier(psm),
                ],
            )
        )
        return f"Comment: {comments}"


class MGF(_Writer):
    """Write MGF files from MS2PIP processing results."""

    def write(self, results: List[ProcessingResult]):
        """Write multiple processing results to file."""
        for result in results:
            self._write_result(result)

    def _write_result(self, result: ProcessingResult):
        """Write single processing result to file."""
        predicted_spectrum = result.as_spectra()[0]
        intensity_normalized = _basepeak_normalize(predicted_spectrum.intensity) * 1e4
        peaks = zip(predicted_spectrum.mz, intensity_normalized)

        # Header
        lines = [
            "BEGIN IONS",
            f"TITLE={result.psm.peptidoform}",
            f"PEPMASS={result.psm.peptidoform.theoretical_mz}",
            f"CHARGE={result.psm.get_precursor_charge()}+",
            f"SCANS={result.psm.spectrum_id}",
            f"RTINSECONDS={result.psm.retention_time}" if result.psm.retention_time else None,
        ]

        # Peaks
        lines.extend(f"{mz:.10g} {intensity:.10g}" for mz, intensity in peaks)

        # Write to file
        self._file_object.writelines(line + "\n" for line in lines if line)
        self._file_object.write("END IONS\n")


class Spectronaut(_Writer):
    """Write Spectronaut files from MS2PIP processing results."""

    field_names = [
        "ModifiedPeptide",
        "StrippedPeptide",
        "PrecursorCharge",
        "PrecursorMz",
        "IonMobility",
        "iRT",
        "ProteinId",
        "RelativeFragmentIntensity",
        "FragmentMz",
        "FragmentType",
        "FragmentNumber",
        "FragmentCharge",
        "FragmentLossType",
    ]

    def write(self, processing_results: List[ProcessingResult]):
        """Write multiple processing results to file."""
        writer = csv.DictWriter(self._file_object, fieldnames=self.field_names, delimiter="\t")
        if self.write_mode == "w":
            writer.writeheader()
        for result in processing_results:
            self._write_result(result, writer)

    def _write_result(self, result: ProcessingResult, writer: csv.DictWriter):
        """Write single processing result to file."""
        # Only write results with predictions
        if result.predicted_intensity is None:
            return
        psm_info = self._process_psm(result.psm)
        for fragment_info in self._yield_fragment_info(result):
            writer.writerow({**psm_info, **fragment_info})

    @staticmethod
    def _process_psm(psm: PSM) -> Dict[str, Any]:
        """Process PSM to Spectronaut format."""
        return {
            "ModifiedPeptide": _peptidoform_str_without_charge(psm.peptidoform),
            "StrippedPeptide": psm.peptidoform.sequence,
            "PrecursorCharge": psm.get_precursor_charge(),
            "PrecursorMz": f"{psm.peptidoform.theoretical_mz:.10g}",
            "IonMobility": f"{psm.ion_mobility:.10g}" if psm.ion_mobility else None,
            "iRT": f"{psm.retention_time:.10g}" if psm.retention_time else None,
            "ProteinId": "".join(psm.protein_list) if psm.protein_list else None,
        }

    @staticmethod
    def _yield_fragment_info(result: ProcessingResult) -> Generator[Dict[str, Any], None, None]:
        """Yield fragment information for a processing result."""
        # Normalize intensities
        intensities = {
            ion_type: _unlogarithmize(intensities)
            for ion_type, intensities in result.predicted_intensity.items()
        }
        max_intensity = max(itertools.chain(*intensities.values()))
        intensities = {
            ion_type: _basepeak_normalize(intensities[ion_type], basepeak=max_intensity)
            for ion_type in intensities
        }
        for ion_type in result.predicted_intensity:
            fragment_type = ion_type[0].lower()
            fragment_charge = ion_type[1:] if len(ion_type) > 1 else "1"
            for ion_index, (intensity, mz) in enumerate(
                zip(intensities[ion_type], result.theoretical_mz[ion_type])
            ):
                yield {
                    "RelativeFragmentIntensity": f"{intensity:.10g}",
                    "FragmentMz": f"{mz:.10g}",
                    "FragmentType": fragment_type,
                    "FragmentNumber": ion_index + 1,
                    "FragmentCharge": fragment_charge,
                    "FragmentLossType": "noloss",
                }


class Bibliospec(_Writer):
    """Write Bibliospec files from MS2PIP processing results."""

    ssl_field_names = [
        "file",
        "scan",
        "charge",
        "sequence",
        "score-type",
        "score",
        "retention-time",
    ]

    def __init__(
        self,
        ssl_file: Union[str, Path, StringIO],
        ms2_file: Union[str, Path, StringIO],
        write_mode: str = "w",
    ):
        """
        Write Bibliospec files from MS2PIP processing results.

        Parameters
        ----------
        ssl_file : Union[str, Path, StringIO]
            Path to SSL file or StringIO object.
        ms2_file : Union[str, Path, StringIO]
            Path to MS2 file or StringIO object.
        write_mode : str
            Write mode for files. Default is "w".
        """

        self.ssl_file = ssl_file
        self.ms2_file = ms2_file
        self.write_mode = write_mode

        self._open_ssl_file = None
        self._open_ms2_file = None

    def __enter__(self):
        """Open file in context manager."""
        if isinstance(self.ssl_file, (str, Path)):
            self.ssl_file = Path(self.ssl_file)
            self._open_ssl_file = open(self.ssl_file, self.write_mode)
        if isinstance(self.ms2_file, (str, Path)):
            self.ms2_file = Path(self.ms2_file)
            self._open_ms2_file = open(self.ms2_file, self.write_mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._open_ssl_file:
            self._open_ssl_file.close()
        if self._open_ms2_file:
            self._open_ms2_file.close()

    @property
    def _ssl_file_object(self):
        """Get SSL file object from file path or StringIO."""
        if self._open_ssl_file:
            return self._open_ssl_file
        else:
            if isinstance(self.ssl_file, StringIO):
                return self.ssl_file
            elif isinstance(self.ssl_file, (str, Path)):
                raise TypeError("Use `with` statement to open file from path.")
            else:
                raise TypeError("Unsupported type for `file`.")

    @property
    def _ms2_file_object(self):
        """Get MS2 file object from file path or StringIO."""
        if self._open_ms2_file:
            return self._open_ms2_file
        else:
            if isinstance(self.ms2_file, StringIO):
                return self.ms2_file
            elif isinstance(self.ms2_file, (str, Path)):
                raise TypeError("Use `with` statement to open file from path.")
            else:
                raise TypeError("Unsupported type for `file`.")

    def write(self, processing_results: List[ProcessingResult]):
        """Write multiple processing results to file."""
        # Create CSV writer
        ssl_dict_writer = csv.DictWriter(
            self._ssl_file_object, fieldnames=self.ssl_field_names, delimiter="\t"
        )

        # Write headers
        if self.write_mode == "w":
            ssl_dict_writer.writeheader()
            self._write_ms2_header()
            start_scan_number = 0
        elif self.write_mode == "a":
            start_scan_number = self._get_last_ssl_scan_number(self.ssl_file) + 1
        else:
            raise ValueError(f"Unsupported write mode: {self.write_mode}")

        # Write results
        for i, result in enumerate(processing_results):
            scan_number = start_scan_number + i
            modified_sequence = self._format_modified_sequence(result.psm.peptidoform)
            self._write_result(result, modified_sequence, scan_number, ssl_dict_writer)

    def _write_ms2_header(self):
        """Write header to MS2 file."""
        self._ms2_file_object.write(
            f"H\tCreationDate\t{strftime('%Y-%m-%d %H:%M:%S', localtime())}\n"
        )
        self._ms2_file_object.write("H\tExtractor\tMS2PIP predictions\n")

    def _write_result(
        self,
        result: ProcessingResult,
        modified_sequence: str,
        scan_number: int,
        writer: csv.DictWriter,
    ):
        """Write single processing result to files."""
        self._write_result_to_ssl(result, modified_sequence, scan_number, writer)
        self._write_result_to_ms2(result, modified_sequence, scan_number)

    def _write_result_to_ssl(
        self,
        result: ProcessingResult,
        modified_sequence: str,
        scan_number: int,
        writer: csv.DictWriter,
    ):
        """Write single processing result to the SSL file."""
        writer.writerow(
            {
                "file": self.ms2_file.name if isinstance(self.ms2_file, Path) else "file.ms2",
                "scan": scan_number,
                "charge": result.psm.get_precursor_charge(),
                "sequence": modified_sequence,
                "score-type": None,
                "score": None,
                "retention-time": result.psm.retention_time if result.psm.retention_time else None,
            }
        )

    def _write_result_to_ms2(
        self, result: ProcessingResult, modified_sequence: str, scan_number: int
    ):
        """Write single processing result to the MS2 file."""
        predicted_spectrum = result.as_spectra()[0]
        intensity_normalized = _basepeak_normalize(predicted_spectrum.intensity) * 1e4
        peaks = zip(predicted_spectrum.mz, intensity_normalized)

        # Header
        lines = [
            f"S\t{scan_number}\t{result.psm.peptidoform.theoretical_mz}",
            f"Z\t{result.psm.get_precursor_charge()}\t{result.psm.peptidoform.theoretical_mass}",
            f"D\tseq\t{result.psm.peptidoform.sequence}",
            f"D\tmodified seq\t{modified_sequence}",
        ]

        # Peaks
        lines.extend(f"{mz:.10g}\t{intensity:.10g}" for mz, intensity in peaks)

        # Write to file
        self._ms2_file_object.writelines(line + "\n" for line in lines)
        self._ms2_file_object.write("\n")

    @staticmethod
    def _format_modified_sequence(peptidoform: Peptidoform) -> str:
        """Format modified sequence as string for Spectronaut."""
        modification_dict = defaultdict(list)
        for term, position in [("n_term", 0), ("c_term", len(peptidoform) - 1)]:
            if peptidoform.properties[term]:
                modification_dict[position].extend(peptidoform.properties[term])
        for position, (_, mods) in enumerate(peptidoform.parsed_sequence):
            if mods:
                modification_dict[position].extend(mods)
        return "".join(
            [
                f"{aa}{''.join([f'[{mod.mass:+.1f}]' for mod in modification_dict[position]])}"
                for position, aa in enumerate(peptidoform.sequence)
            ]
        )

    @staticmethod
    def _get_last_ssl_scan_number(ssl_file: Union[str, Path, StringIO]):
        """Read scan number of last line in a Bibliospec SSL file."""
        if isinstance(ssl_file, StringIO):
            ssl_file.seek(0)
            for line in ssl_file:
                last_line = line
        elif isinstance(ssl_file, (str, Path)):
            with open(ssl_file, "rt") as ssl:
                for line in ssl:
                    last_line = line
        else:
            raise TypeError("Unsupported type for `ssl_file`.")
        return int(last_line.split("\t")[1])


class DLIB(_Writer):
    """
    Write DLIB files from MS2PIP processing results.

    See https://bitbucket.org/searleb/encyclopedia/wiki/EncyclopeDIA%20File%20Formats for
    documentation on the DLIB format.

    """

    def write(self, processing_results: List[ProcessingResult]):
        """Write MS2PIP predictions to a DLIB SQLite file."""
        with dlib.open_sqlite(self.file) as connection:
            dlib.metadata.create_all()
            self._write_metadata(connection)
            self._write_entries(processing_results, connection, self.file)
            self._write_peptide_to_protein(processing_results, connection)

    def _write_result(self, result: ProcessingResult):
        """Write single processing result to file."""
        ...

    @staticmethod
    def _format_modified_sequence(peptidoform: Peptidoform) -> str:
        """Format modified sequence as string for DLIB."""
        # TODO: Implement
        # From the EncyclopeDIA DLIB documentation:
        # PeptideModSeq has strings like "QKEC[+57.0214635]SDK" to indicate PTMs. PTMs are always
        # encoded as delta masses (including fixed PTMs such as carbamidomethylation). Sites can
        # only have one PTM mass, so compound masses are allowed, such as
        # "M[+58.00548]ELS[+79.966331]C[+57.0214635]PGSR", where +58.00548 indicates both
        # acetylation and oxidation. N- and C-terminus PTMs should be annotated on the first or
        # last amino acid in the peptide, respectively. Metabolic labels can be incorporated in
        # the same way, for example EC[+57.0214635]SDK[+8.014199].
        raise NotImplementedError

    @staticmethod
    def _write_metadata(connection: engine.Connection):
        with connection.begin():
            version = connection.execute(
                select([dlib.Metadata.c.Value]).where(dlib.Metadata.c.Key == "version")
            ).scalar()
            if version is None:
                connection.execute(
                    dlib.Metadata.insert().values(
                        Key="version",
                        Value=dlib.DLIB_VERSION,
                    )
                )

    @staticmethod
    def _write_entries(
        processing_results: List[ProcessingResult],
        connection: engine.Connection,
        output_filename: str,
    ):
        with connection.begin():
            for result in processing_results:
                if not result.psm.retention_time:
                    raise ValueError("Retention time required to write DLIB file.")

                spectrum = result.as_spectra()[0]
                intensity_normalized = _basepeak_normalize(spectrum.intensity) * 1e4
                n_peaks = len(spectrum.mz)

                connection.execute(
                    dlib.Entry.insert().values(
                        PrecursorMz=result.psm.precursor_mz,
                        PrecursorCharge=result.psm.get_precursor_charge(),
                        PeptideModSeq=DLIB._format_modified_sequence(result.psm.peptidoform),
                        PeptideSeq=result.psm.peptidoform.sequence,
                        Copies=1,
                        RTInSeconds=result.psm.retention_time,
                        Score=0,
                        MassEncodedLength=n_peaks,
                        MassArray=spectrum.mz,
                        IntensityEncodedLength=n_peaks,
                        IntensityArray=intensity_normalized,
                        SourceFile=output_filename,
                    )
                )

    @staticmethod
    def _write_peptide_to_protein(results: List[ProcessingResult], connection: engine.Connection):
        from ms2pip._utils.dlib import PeptideToProtein

        peptide_to_proteins = {
            (result.psm.peptidoform.sequence, protein)
            for result in results
            for protein in result.psm.protein_list
        }

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
