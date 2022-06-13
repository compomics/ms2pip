"""Read MS2 spectra."""

from pathlib import Path
from typing import Generator, List

import numpy as np
from pyteomics import mzml, mgf

from ms2pip.exceptions import (
    UnsupportedSpectrumFiletypeError,
    InvalidSpectrumError,
    EmptySpectrumError,
)


class Spectrum:
    def __init__(self, title, charge, pepmass, msms, peaks) -> None:
        """Minimal information on observed MS2 spectrum."""
        self.title = str(title)
        self.charge = int(charge)
        self.pepmass = float(pepmass)
        self.msms = np.array(msms, dtype=np.float32)
        self.peaks = np.array(peaks, dtype=np.float32)

        self.tic = np.sum(self.peaks)

        if len(self.msms) != len(self.peaks):
            raise InvalidSpectrumError(
                "Inconsistent number of m/z and intensity values."
            )

    def __repr__(self) -> str:
        return "{}.{}({})".format(
            self.__class__.__module__,
            self.__class__.__qualname__,
            f"title='{self.title}'",
        )

    def validate_spectrum_content(self) -> None:
        """Raise EmptySpectrumError if no peaks are present."""
        if (len(self.peaks) == 0) or (len(self.msms) == 0):
            raise EmptySpectrumError()

    def remove_reporter_ions(self, label_type=None) -> None:
        """Remove reporter ions."""
        if label_type == "iTRAQ":
            for mi, mp in enumerate(self.msms):
                if (mp >= 113) & (mp <= 118):
                    self.peaks[mi] = 0

        # TMT6plex: 126.1277, 127.1311, 128.1344, 129.1378, 130.1411, 131.1382
        elif label_type == "TMT":
            for mi, mp in enumerate(self.msms):
                if (mp >= 125) & (mp <= 132):
                    self.peaks[mi] = 0

    def remove_precursor(self, tolerance=0.02) -> None:
        """Remove precursor peak."""
        for mi, mp in enumerate(self.msms):
            if (mp >= self.pepmass - tolerance) & (mp <= self.pepmass + tolerance):
                self.peaks[mi] = 0

    def tic_norm(self) -> None:
        """Normalize spectrum to total ion current."""
        self.peaks = self.peaks / self.tic

    def log2_transform(self) -> None:
        """Log2-tranform spectrum."""
        self.peaks = np.log2(self.peaks + 0.001)


def scan_spectrum_file(filename) -> List[str]:
    """Iterate over MGF file and return list with all spectrum titles."""
    titles = []
    f = open(filename)
    while 1:
        rows = f.readlines(10000)
        if not rows:
            break
        for row in rows:
            if row[0] == "T":
                if row[:5] == "TITLE":
                    titles.append(row.rstrip()[6:])
    f.close()
    return titles


def read_mgf_legacy(
    spec_file, peptide_titles: List[str] = None
) -> Generator[Spectrum, None, None]:
    """
    Read MGF file (legacy code).

    Parameters
    ----------
    spec_file: str
        Path to MGF file.
    peptide_titles: list[str], optional
        List with peptide `spec_id` values which correspond to MGF TITLE field
        values.

    """

    # Initiate properties for first spectrum
    title = ""
    pepmass = 0
    charge = 0
    msms = []
    peaks = []

    f = open(spec_file)
    skip = False

    # Iterate over spectra
    with open(spec_file, "rt") as f:
        while True:
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
                        if peptide_titles and title not in peptide_titles:
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
                elif row[0] == "P":
                    if row[:7] == "PEPMASS":
                        pepmass = float(row.split("=")[1].split(" ")[0])
                elif row[:8] == "END IONS":
                    yield Spectrum(title, charge, pepmass, msms, peaks)


def read_mgf(
    spec_file, peptide_titles: List[str] = None
) -> Generator[Spectrum, None, None]:
    """
    Read MGF file.

    Parameters
    ----------
    spec_file: str
        Path to MGF file.
    peptide_titles: list[str], optional
        List with peptide `spec_id` values which correspond to MGF TITLE field
        values.

    """
    # TODO check most optimal mgf.read options
    with mgf.read(spec_file) as mgf_file:
        for spectrum in mgf_file:
            spec_id = spectrum["params"]["title"]
            if peptide_titles and spec_id not in peptide_titles:
                continue
            peaks = spectrum["intensity array"]
            msms = spectrum["m/z array"]
            precursor_mz = spectrum["params"]["pepmass"][0]
            precursor_charge = spectrum["params"]["charge"][0]
            parsed_spectrum = Spectrum(
                spec_id, precursor_charge, precursor_mz, msms, peaks
            )
            yield parsed_spectrum


def read_mzml(
    spec_file, peptide_titles: List[str] = None
) -> Generator[Spectrum, None, None]:
    """
    Read mzML file.

    Parameters
    ----------
    spec_file: str
        Path to mzML file.
    peptide_titles: list[str], optional
        List with peptide `spec_id` values which correspond to mzML spectrum id
        values.

    """

    with mzml.read(spec_file) as mzml_file:
        for spectrum in mzml_file:
            if spectrum["ms level"] == 2:
                spec_id = spectrum["id"]
                if peptide_titles and spec_id not in peptide_titles:
                    continue
                peaks = spectrum["intensity array"]
                msms = spectrum["m/z array"]
                precursor = spectrum["precursorList"]["precursor"][0][
                    "selectedIonList"
                ]["selectedIon"][0]
                precursor_mz = precursor["selected ion m/z"]
                precursor_charge = precursor["charge state"]
                parsed_spectrum = Spectrum(
                    spec_id, precursor_charge, precursor_mz, msms, peaks
                )
                yield parsed_spectrum


def read_spectrum_file(
    spec_file, peptide_titles: List[str] = None
) -> Generator[Spectrum, None, None]:
    """
    Read MGF or mzML file; infer type from filename extension.

    Parameters
    ----------
    spec_file: str
        Path to mzML file.
    peptide_titles: list[str], optional
        List with peptide `spec_id` values which correspond to mzML spectrum id
        values.

    """

    filetype = Path(spec_file).suffix.lower()

    if filetype == ".mzml":
        for spectrum in read_mzml(spec_file, peptide_titles):
            yield spectrum

    elif filetype == ".mgf":
        for spectrum in read_mgf(spec_file, peptide_titles):
            yield spectrum

    else:
        raise UnsupportedSpectrumFiletypeError(filetype)
