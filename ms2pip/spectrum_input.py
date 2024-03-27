"""Read MS2 spectra."""

from pathlib import Path
from typing import Generator

import numpy as np
try:
    import timsrust_pyo3 as timsrust
    _has_timsrust = True
except ImportError:
    _has_timsrust = False
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
                mz=spectrum["m/z array"].astype(np.float32),
                intensity=spectrum["intensity array"].astype(np.float32),
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
                    mz=spectrum["m/z array"].astype(np.float32),
                    intensity=spectrum["intensity array"].astype(np.float32),
                    precursor_mz=precursor["selected ion m/z"],
                    precursor_charge=precursor_charge,
                )
                yield spectrum


def read_tdf(spectrum_file: str) -> Generator[ObservedSpectrum, None, None]:
    """
    Read MS2 DDA spectra from .d folder.

    Parameters
    ----------
    spectrum_file
        Path to .d folder.

    """
    if not _has_timsrust:
        raise ImportError(
            "Optional dependency timsrust_pyo3 required for .d spectrum file support. Reinstall "
            "ms2pip with `pip install ms2pip[tdf]` and try again."
        )
    reader = timsrust.TimsReader(spectrum_file)
    for spectrum in reader.read_all_spectra():
        spectrum = ObservedSpectrum(
            identifier=spectrum.index,
            mz=np.asarray(spectrum.mz_values, dtype=np.float32),
            intensity=np.asarray(spectrum.intensities, dtype=np.float32),
            precursor_mz=spectrum.precursor.mz,
            precursor_charge=spectrum.precursor.charge
            )
        yield spectrum


def read_spectrum_file(spectrum_file: str) -> Generator[ObservedSpectrum, None, None]:
    """
    Read MS2 spectra from MGF or mzML file or .d folder; inferring the type from the filename extension.

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
    elif filetype == ".d":
        for spectrum in read_tdf(spectrum_file):
            yield spectrum
    else:
        raise UnsupportedSpectrumFiletypeError(filetype)
