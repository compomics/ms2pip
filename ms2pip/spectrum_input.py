"""Read MS2 spectra."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
from ms2rescore_rs import get_ms2_spectra  # type: ignore[ty:unresolved-import]

import ms2pip.exceptions as exceptions
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
    try:
        spectra = get_ms2_spectra(str(spectrum_file))
    except ValueError as e:
        raise exceptions.UnsupportedSpectrumFiletypeError(Path(spectrum_file).suffixes) from e

    for spectrum in spectra:
        obs_spectrum = ObservedSpectrum(
            mz=np.array(spectrum.mz, dtype=np.float32),
            intensity=np.array(spectrum.intensity, dtype=np.float32),
            identifier=str(spectrum.identifier),
            precursor_mz=float(spectrum.precursor.mz),
            precursor_charge=int(spectrum.precursor.charge),
            retention_time=float(spectrum.precursor.rt),
        )
        # Workaround for mobiusklein/mzdata#3
        if (
            obs_spectrum.identifier == ""
            or obs_spectrum.mz.shape[0] == 0
            or obs_spectrum.intensity.shape[0] == 0
        ):
            continue
        yield obs_spectrum
