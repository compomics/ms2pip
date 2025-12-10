"""
Write spectrum files from MS²PIP prediction results.


Examples
--------

The simplest way to write MS²PIP predictions to a file is to use the :py:func:`write_spectra`
function:

>>> from ms2pip import predict_single, write_spectra
>>> results = [predict_single("ACDE/2")]
>>> write_spectra("/path/to/output/filename", results, "mgf")

Specific writer classes can also be used directly. Writer classes should be used in a context
manager to ensure the file is properly closed after writing. The following example writes MS²PIP
predictions to a TSV file:

>>> from ms2pip import predict_single
>>> results = [predict_single("ACDE/2")]
>>> with TSV("output.tsv") as writer:
...     writer.write(results)

Results can be written to the same file sequentially:

>>> results_2 = [predict_single("PEPTIDEK/2")]
>>> with TSV("output.tsv", write_mode="a") as writer:
...     writer.write(results)
...     writer.write(results_2)

Results can be written to an existing file using the append mode:

>>> with TSV("output.tsv", write_mode="a") as writer:
...     writer.write(results_2)


"""

from __future__ import annotations

import csv
import itertools
import logging
import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from io import StringIO
from pathlib import Path
from os import PathLike
from time import localtime, strftime
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
from psm_utils import PSM, Peptidoform
from pyteomics import proforma
from sqlalchemy import select
from sqlalchemy.engine import Connection

from ms2pip._utils import dlib
from ms2pip.result import ProcessingResult

LOGGER = logging.getLogger(__name__)


def write_spectra(
    filename: Union[str, PathLike],
    processing_results: List[ProcessingResult],
    file_format: str = "tsv",
    write_mode: str = "w",
):
    """
    Write MS2PIP processing results to a supported spectrum file format.

    Parameters
    ----------
    filename
        Output filename without file extension.
    processing_results
        List of :py:class:`ms2pip.result.ProcessingResult` objects.
    file_format
        File format to write. See :py:attr:`FILE_FORMATS` for available formats.
    write_mode
        Write mode for file. Default is ``w`` (write). Use ``a`` (append) to add to existing file.

    """
    with SUPPORTED_FORMATS[file_format](filename, write_mode) as writer:
        LOGGER.info(f"Writing to {writer.filename}")
        writer.write(processing_results)


class _Writer(ABC):
    """Abstract base class for writing spectrum files."""

    suffix = ""

    def __init__(self, filename: Union[str, PathLike], write_mode: str = "w"):
        self.filename = Path(filename).with_suffix(self.suffix)
        self.write_mode = write_mode

        self._open_file = None

    def __enter__(self):
        """Open file in context manager."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close file in context manager."""
        self.close()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.filename, self.write_mode})"

    def open(self):
        """Open file."""
        if self._open_file:
            self.close()
        self._open_file = open(self.filename, self.write_mode)

    def close(self):
        """Close file."""
        if self._open_file:
            self._open_file.close()
            self._open_file = None

    @property
    def _file_object(self):
        """Get open file object."""
        if self._open_file:
            return self._open_file
        else:
            warnings.warn(
                "Opening file outside of context manager. Manually close file after use."
            )
            self.open()
            return self._open_file

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

    suffix = ".tsv"
    field_names = [
        "psm_index",
        "ion_type",
        "ion_number",
        "mz",
        "predicted",
        "observed",
        "rt",
        "im",
    ]

    def write(self, processing_results: List[ProcessingResult]):
        """Write multiple processing results to file."""
        writer = csv.DictWriter(
            self._file_object, fieldnames=self.field_names, delimiter="\t", lineterminator="\n"
        )
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
            "mz": "{:.8f}".format(result.theoretical_mz[ion_type][ion_index]),
            "predicted": "{:.8f}".format(result.predicted_intensity[ion_type][ion_index])
            if result.predicted_intensity
            else None,
            "observed": "{:.8f}".format(result.observed_intensity[ion_type][ion_index])
            if result.observed_intensity
            else None,
            "rt": result.psm.retention_time if result.psm.retention_time else None,
            "im": result.psm.ion_mobility if result.psm.ion_mobility else None,
        }


class MSP(_Writer):
    """Write MSP files from MS2PIP processing results."""

    suffix = ".msp"

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
            f"{mz:.8f}\t{intensity:.8f}\t{annotation}/0.0" for mz, intensity, annotation in peaks
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
                raise ValueError("Multiple modifications per amino acid not supported in MSP.")
            modification = modifications[0]
            try:
                return f"{position},{amino_acid},{modification.name}"
            except AttributeError:  # MassModification has no attribute `name`
                return f"{position},{amino_acid},{modification.value}"

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
            return f"Mods={len(mods)}/{'/'.join(mods)}"

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
    def _format_ion_mobility(psm: PSM) -> Union[str, None]:
        """Format ion mobility as string."""
        if psm.ion_mobility:
            return f"IonMobility={psm.ion_mobility}"
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
                    MSP._format_ion_mobility(psm),
                    MSP._format_identifier(psm),
                ],
            )
        )
        return f"Comment: {comments}"


class MGF(_Writer):
    """
    Write MGF files from MS2PIP processing results.

    See http://www.matrixscience.com/help/data_file_help.html for documentation on the MGF format.
    """

    suffix = ".mgf"

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
            f"ION_MOBILITY={result.psm.ion_mobility}" if result.psm.ion_mobility else None,
        ]

        # Peaks
        lines.extend(f"{mz:.8f} {intensity:.8f}" for mz, intensity in peaks)

        # Write to file
        self._file_object.writelines(line + "\n" for line in lines if line)
        self._file_object.write("END IONS\n\n")


class Spectronaut(_Writer):
    """Write Spectronaut files from MS2PIP processing results."""

    suffix = ".spectronaut.tsv"
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
        writer = csv.DictWriter(
            self._file_object, fieldnames=self.field_names, delimiter="\t", lineterminator="\n"
        )
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
            "PrecursorMz": f"{psm.peptidoform.theoretical_mz:.8f}",
            "IonMobility": f"{psm.ion_mobility:.8f}" if psm.ion_mobility else None,
            "iRT": f"{psm.retention_time:.8f}" if psm.retention_time else None,
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
                    "RelativeFragmentIntensity": f"{intensity:.8f}",
                    "FragmentMz": f"{mz:.8f}",
                    "FragmentType": fragment_type,
                    "FragmentNumber": ion_index + 1,
                    "FragmentCharge": fragment_charge,
                    "FragmentLossType": "noloss",
                }


class Bibliospec(_Writer):
    """
    Write Bibliospec SSL and MS2 files from MS2PIP processing results.

    Bibliospec SSL and MS2 files are also compatible with Skyline. See
    https://skyline.ms/wiki/home/software/BiblioSpec/page.view?name=BiblioSpec%20input%20and%20output%20file%20formats
    for documentation on the Bibliospec file formats.

    """

    ssl_suffix = ".ssl"
    ms2_suffix = ".ms2"
    ssl_field_names = [
        "file",
        "scan",
        "charge",
        "sequence",
        "score-type",
        "score",
        "retention-time",
        "ion-mobility",
    ]

    def __init__(self, filename: Union[str, PathLike], write_mode: str = "w"):
        super().__init__(filename, write_mode)
        self.ssl_file = self.filename.with_suffix(self.ssl_suffix)
        self.ms2_file = self.filename.with_suffix(self.ms2_suffix)

        self._open_ssl_file = None
        self._open_ms2_file = None

    def open(self):
        """Open files."""
        self._open_ssl_file = open(self.ssl_file, self.write_mode)
        self._open_ms2_file = open(self.ms2_file, self.write_mode)

    def close(self):
        """Close files."""
        if self._open_ssl_file:
            self._open_ssl_file.close()
            self._open_ssl_file = None
        if self._open_ms2_file:
            self._open_ms2_file.close()
            self._open_ms2_file = None

    @property
    def _ssl_file_object(self):
        """Get open SSL file object."""
        if self._open_ssl_file:
            return self._open_ssl_file
        else:
            warnings.warn(
                "Opening file outside of context manager. Manually close file after use."
            )
            self.open()
            return self._open_ssl_file

    @property
    def _ms2_file_object(self):
        """Get open MS2 file object."""
        if self._open_ms2_file:
            return self._open_ms2_file
        else:
            warnings.warn(
                "Opening file outside of context manager. Manually close file after use."
            )
            self.open()
            return self._open_ms2_file

    def write(self, processing_results: List[ProcessingResult]):
        """Write multiple processing results to file."""
        # Create CSV writer
        ssl_dict_writer = csv.DictWriter(
            self._ssl_file_object,
            fieldnames=self.ssl_field_names,
            delimiter="\t",
            lineterminator="\n",
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
                "ion-mobility": result.psm.ion_mobility if result.psm.ion_mobility else None,
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
        lines.extend(f"{mz:.8f}\t{intensity:.8f}" for mz, intensity in peaks)

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
    def _get_last_ssl_scan_number(ssl_file: Union[str, PathLike, StringIO]):
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

    See `EncyclopeDIA File Formats <https://bitbucket.org/searleb/encyclopedia/wiki/EncyclopeDIA%20File%20Formats>`_
    for documentation on the DLIB format.

    """

    suffix = ".dlib"

    def open(self):
        """Open file."""
        if self.write_mode == "w":
            self._open_file = self.filename.unlink(missing_ok=True)
        self._open_file = dlib.open_sqlite(self.filename)

    def write(self, processing_results: List[ProcessingResult]):
        """Write MS2PIP predictions to a DLIB SQLite file."""
        connection = self._file_object
        dlib.metadata.create_all(connection.engine)
        self._write_metadata(connection)
        self._write_entries(processing_results, connection, self.filename)
        self._write_peptide_to_protein(processing_results, connection)

    def _write_result(self, result: ProcessingResult): ...

    @staticmethod
    def _format_modified_sequence(peptidoform: Peptidoform) -> str:
        """Format modified sequence as string for DLIB."""
        # Sum all sequential mass shifts for each position
        masses = [
            sum(mod.mass for mod in mods) if mods else 0 for _, mods in peptidoform.parsed_sequence
        ]

        # Add N- and C-terminal modifications
        for term, position in [("n_term", 0), ("c_term", len(peptidoform) - 1)]:
            if peptidoform.properties[term]:
                masses[position] += sum(mod.mass for mod in peptidoform.properties[term])

        # Format modified sequence
        return "".join(
            [
                f"{aa}[{mass:+.6f}]" if mass else aa
                for aa, mass in zip(peptidoform.sequence, masses)
            ]
        )

    @staticmethod
    def _write_metadata(connection: Connection):
        """Write metadata to DLIB SQLite file."""
        with connection.begin():
            version = connection.execute(
                select(dlib.Metadata.c.Value).where(dlib.Metadata.c.Key == "version")
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
        connection: Connection,
        output_filename: Union[str, PathLike],
    ):
        """Write spectra to DLIB SQLite file."""
        with connection.begin():
            for result in processing_results:
                if not result.psm.retention_time:
                    raise ValueError("Retention time required to write DLIB file.")

                spectrum = result.as_spectra()[0]
                intensity_normalized = _basepeak_normalize(spectrum.intensity) * 1e4
                n_peaks = len(spectrum.mz)

                connection.execute(
                    dlib.Entry.insert().values(
                        PrecursorMz=result.psm.peptidoform.theoretical_mz,
                        PrecursorCharge=result.psm.get_precursor_charge(),
                        PeptideModSeq=DLIB._format_modified_sequence(result.psm.peptidoform),
                        PeptideSeq=result.psm.peptidoform.sequence,
                        Copies=1,
                        RTInSeconds=result.psm.retention_time,
                        Score=0,
                        MassEncodedLength=n_peaks,
                        MassArray=spectrum.mz.tolist(),
                        IntensityEncodedLength=n_peaks,
                        IntensityArray=intensity_normalized.tolist(),
                        SourceFile=str(output_filename),
                    )
                )

    @staticmethod
    def _write_peptide_to_protein(results: List[ProcessingResult], connection: Connection):
        """Write peptide-to-protein mappings to DLIB SQLite file."""
        peptide_to_proteins = {
            (result.psm.peptidoform.sequence, protein)
            for result in results
            if result.psm.protein_list
            for protein in result.psm.protein_list
        }

        with connection.begin():
            sql_peptide_to_proteins = set()
            proteins = {protein for _, protein in peptide_to_proteins}
            for peptide_to_protein in connection.execute(
                select(dlib.PeptideToProtein).where(
                    dlib.PeptideToProtein.c.ProteinAccession.in_(proteins)
                )
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
                    dlib.PeptideToProtein.insert().values(
                        PeptideSeq=seq, isDecoy=False, ProteinAccession=protein
                    )
                )


SUPPORTED_FORMATS = {
    "tsv": TSV,
    "msp": MSP,
    "mgf": MGF,
    "spectronaut": Spectronaut,
    "bibliospec": Bibliospec,
    "dlib": DLIB,
}


def _peptidoform_str_without_charge(peptidoform: Peptidoform) -> str:
    """Get peptidoform string without charge."""
    return re.sub(r"\/\d+$", "", str(peptidoform))


def _unlogarithmize(intensities: np.array) -> np.array:
    """Undo logarithmic transformation of intensities."""
    return (2**intensities) - 0.001


def _basepeak_normalize(intensities: np.array, basepeak: Optional[float] = None) -> np.array:
    """Normalize intensities to most intense peak."""
    if not basepeak:
        basepeak = intensities.max()
    return intensities / basepeak
