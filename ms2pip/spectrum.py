"""Read MS2 spectra."""

from typing import Generator, List

import numpy as np

from ms2pip.exceptions import InvalidSpectrumError, EmptySpectrumError


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
          if (mp >= self.pepmass-tolerance) & (mp <= self.pepmass+tolerance):
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
                    titles.append(
                        row.rstrip()[6:]
                    )
    f.close()
    return titles


def read_mgf(
    spec_file, peptide_titles: List[str] = None
) -> Generator[Spectrum, None, None]:
    """
    Read MGF file (legacy code).

    Parameters
    ----------
    spec_file: str
        Path to MGF file.
    peptide_titles: list[str], optional
        List with peptide `spec_id` values which corresond to MGF TITLE field
        values.

    """
    if not peptide_titles:
        peptide_titles = []

    # Initiate spectrum properties
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
                        if title not in peptide_titles:
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
