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


def read_mgf(spec_file) -> Generator[Spectrum, None, None]:
    """
    Read MGF file.

    Parameters
    ----------
    spec_file: str
        Path to MGF file.

    """
    with mgf.read(
        spec_file,
        convert_arrays=1,
        read_charges=False,
        read_ions=False,
        dtype=np.float32,
        use_index=False,
    ) as mgf_file:
        for spectrum in mgf_file:
            spec_id = spectrum["params"]["title"]
            peaks = spectrum["intensity array"]
            msms = spectrum["m/z array"]
            precursor_mz = spectrum["params"]["pepmass"][0]
            precursor_charge = spectrum["params"]["charge"][0]
            parsed_spectrum = Spectrum(
                spec_id, precursor_charge, precursor_mz, msms, peaks
            )
            yield parsed_spectrum


def read_mzml(spec_file) -> Generator[Spectrum, None, None]:
    """
    Read mzML file.

    Parameters
    ----------
    spec_file: str
        Path to mzML file.

    """
    with mzml.read(
        spec_file,
        read_schema=False,
        iterative=True,
        use_index=False,
        dtype=np.float32,
        decode_binary=True,
    ) as mzml_file:
        for spectrum in mzml_file:
            if spectrum["ms level"] == 2:
                spec_id = spectrum["id"]
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


def read_spectrum_file(spec_file) -> Generator[Spectrum, None, None]:
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
        for spectrum in read_mzml(spec_file):
            yield spectrum
    elif filetype == ".mgf":
        for spectrum in read_mgf(spec_file):
            yield spectrum
    else:
        raise UnsupportedSpectrumFiletypeError(filetype)
