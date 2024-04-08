"""Read MS2 spectra."""

from pathlib import Path
from typing import Generator

import numpy as np
from ms2rescore_rs import get_ms2_spectra

from ms2pip.exceptions import UnsupportedSpectrumFiletypeError
from ms2pip.spectrum import ObservedSpectrum


def read_spectrum_file(spectrum_file: str) -> Generator[ObservedSpectrum, None, None]:
    """
    Read MS2 spectra from a supported file format; inferring the type from the filename extension.

    Parameters
    ----------
    spectrum_file
        Path to MGF or mzML file.

    Yields
    ------
    ObservedSpectrum

    Raises
    ------
    UnsupportedSpectrumFiletypeError
        If the file extension is not supported.

    """
    file_extension = Path(spectrum_file).suffix.lower()
    if file_extension not in [".mgf", ".mzml", ".d"] and not _is_minitdf(spectrum_file):
        raise UnsupportedSpectrumFiletypeError(file_extension)

    for spectrum in get_ms2_spectra(spectrum_file):
        yield ObservedSpectrum(
            mz=np.array(spectrum.mz, dtype=np.float32),
            intensity=np.array(spectrum.intensity, dtype=np.float32),
            identifier=str(spectrum.identifier),
            precursor_mz=float(spectrum.precursor.mz),
            precursor_charge=float(spectrum.precursor.charge),
            retention_time=float(spectrum.precursor.rt),
        )


def _is_minitdf(spectrum_file: str) -> bool:
    """
    Check if the spectrum file is a Bruker miniTDF folder.

    A Bruker miniTDF folder has no fixed name, but contains files matching the patterns
    ``*ms2spectrum.bin`` and ``*ms2spectrum.parquet``.
    """
    files = set(Path(spectrum_file).glob("*ms2spectrum.bin"))
    files.update(Path(spectrum_file).glob("*ms2spectrum.parquet"))
    return len(files) >= 2
