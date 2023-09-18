"""Read MS2 spectra."""

from pathlib import Path
from typing import Generator

import numpy as np
from pyteomics import mgf, mzml

from ms2pip.exceptions import UnsupportedSpectrumFiletypeError
from ms2pip.spectrum import ObservedSpectrum


def read_mgf(spectrum_file: str) -> Generator[ObservedSpectrum, None, None]:
    """
    Read MS2 spectra from MGF file.

    Parameters
    ----------
    spectrum_file
        Path to MGF file.

    """
    with mgf.read(
        str(spectrum_file),
        convert_arrays=1,
        read_charges=False,
        read_ions=False,
        dtype=np.float32,
        use_index=False,
    ) as mgf_file:
        for spectrum in mgf_file:
            try:
                precursor_charge = spectrum["params"]["charge"][0]
            except KeyError:
                precursor_charge = None
            try:
                precursor_mz = spectrum["params"]["pepmass"][0]
            except KeyError:
                precursor_mz = None
            spectrum = ObservedSpectrum(
                identifier=spectrum["params"]["title"],
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
            )
            yield spectrum


def read_mzml(spectrum_file: str) -> Generator[ObservedSpectrum, None, None]:
    """
    Read MS2 spectra from mzML file.

    Parameters
    ----------
    spectrum_file
        Path to mzML file.

    """
    with mzml.read(
        str(spectrum_file),
        read_schema=False,
        iterative=True,
        use_index=False,
        dtype=np.float32,
    ) as mzml_file:
        for spectrum in mzml_file:
            if spectrum["ms level"] == 2:
                precursor = spectrum["precursorList"]["precursor"][0]["selectedIonList"][
                    "selectedIon"
                ][0]
                try:
                    precursor_charge = precursor["charge state"]
                except KeyError:
                    precursor_charge = None
                spectrum = ObservedSpectrum(
                    identifier=spectrum["id"],
                    mz=spectrum["m/z array"],
                    intensity=spectrum["intensity array"],
                    precursor_mz=precursor["selected ion m/z"],
                    precursor_charge=precursor_charge,
                )
                yield spectrum


def read_spectrum_file(spectrum_file: str) -> Generator[ObservedSpectrum, None, None]:
    """
    Read MS2 spectra from MGF or mzML file; inferring the type from the filename extension.

    Parameters
    ----------
    spectrum_file
        Path to MGF or mzML file.

    """
    filetype = Path(spectrum_file).suffix.lower()
    if filetype == ".mzml":
        for spectrum in read_mzml(spectrum_file):
            yield spectrum
    elif filetype == ".mgf":
        for spectrum in read_mgf(spectrum_file):
            yield spectrum
    else:
        raise UnsupportedSpectrumFiletypeError(filetype)
