"""MS2 spectrum handling."""

from __future__ import annotations

import warnings
from typing import Annotated

import numpy as np
from psm_utils import Peptidoform
from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator


def _coerce_peptidoform(v):
    if v is None or isinstance(v, Peptidoform):
        return v
    elif isinstance(v, str):
        return Peptidoform(v)
    raise ValueError("Peptidoform must be a string, a Peptidoform object, or None.")


_PeptidoformField = Annotated[Peptidoform | None, BeforeValidator(_coerce_peptidoform)]


class Spectrum(BaseModel):
    """
    MS2 spectrum.

    Parameters
    ----------
    mz
        Array of m/z values.
    intensity
        Array of intensity values.
    annotations
        Array of peak annotations.
    identifier
        Spectrum identifier.
    peptidoform
        Peptidoform.
    precursor_mz
        Precursor m/z.
    precursor_charge
        Precursor charge.
    retention_time
        Retention time.
    mass_tolerance
        Mass tolerance for spectrum annotation.
    mass_tolerance_unit
        Unit of mass tolerance for spectrum annotation.
    """

    mz: np.ndarray
    intensity: np.ndarray
    annotations: np.ndarray | None = None
    identifier: str | None = None
    peptidoform: _PeptidoformField = None
    precursor_mz: float | None = None
    precursor_charge: int | None = None
    retention_time: float | None = None
    mass_tolerance: float | None = None
    mass_tolerance_unit: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        return "{}.{}({})".format(
            self.__class__.__module__,
            self.__class__.__qualname__,
            f"identifier='{self.identifier}'",
        )

    @model_validator(mode="after")
    @classmethod
    def check_array_lengths(cls, data):
        if len(data.mz) != len(data.intensity):
            raise ValueError("Array lengths do not match.")
        if data.annotations is not None:
            if len(data.annotations) != len(data.intensity):
                raise ValueError("Array lengths do not match.")
        return data

    @property
    def tic(self):
        """Total ion current."""
        return np.sum(self.intensity)

    def remove_reporter_ions(self, label_type=None) -> None:
        """Set the intensity of reporter ions to 0."""
        # TODO: Consider using the exact m/z values instead of a range.
        if label_type == "iTRAQ":
            for i, mz in enumerate(self.mz):
                if (mz >= 113) & (mz <= 118):
                    self.intensity[i] = 0

        # TMT6plex: 126.1277, 127.1311, 128.1344, 129.1378, 130.1411, 131.1382
        elif label_type == "TMT":
            for i, mz in enumerate(self.mz):
                if (mz >= 125) & (mz <= 132):
                    self.intensity[i] = 0

    def remove_precursor(self, tolerance=0.02) -> None:
        """Set the intensity of the precursor peak to 0."""
        if not self.precursor_mz:
            raise ValueError("Precursor m/z must be set.")
        for i, mz in enumerate(self.mz):
            if (mz >= self.precursor_mz - tolerance) & (mz <= self.precursor_mz + tolerance):
                self.intensity[i] = 0

    def tic_norm(self) -> None:
        """Normalize spectrum to total ion current."""
        self.intensity = self.intensity / self.tic

    def log2_transform(self) -> None:
        """Log2-transform spectrum."""
        self.intensity = np.log2(self.intensity + 0.001)

    def inverse_log2_transform(self) -> None:
        """Undo log2 transformation of intensities (inverse of :meth:`log2_transform`)."""
        self.intensity = (2**self.intensity) - 0.001

    def clip_intensity(self, min_intensity=0.0) -> None:
        """Clip intensity values."""
        self.intensity = np.clip(self.intensity, min_intensity, None)

    def to_spectrum_utils(self):
        """
        Convert to spectrum_utils.spectrum.MsmsSpectrum.

        Notes
        -----
        - Requires spectrum_utils to be installed.
        - If the ``precursor_mz`` or ``precursor_charge`` attributes are not set, the theoretical
          m/z and precursor charge of the ``peptidoform`` attribute are used, if present.
          Otherwise, ``ValueError`` is raised.

        """
        try:
            import spectrum_utils.spectrum as sus
        except ImportError as e:
            raise ImportError("Optional dependency spectrum_utils not installed.") from e

        if self.precursor_charge:
            precursor_charge = self.precursor_charge
        else:
            if not self.peptidoform:
                raise ValueError("`precursor_charge` or `peptidoform` must be set.")
            precursor_charge = self.peptidoform.precursor_charge
            if precursor_charge is None:
                raise ValueError("Peptidoform charge state is not set.")

        if self.precursor_mz:
            precursor_mz_float = float(self.precursor_mz)
        else:
            if not self.peptidoform:
                raise ValueError("`precursor_mz` or `peptidoform` must be set.")
            elif not self.peptidoform.theoretical_mz:
                raise ValueError(
                    "Peptidoform theoretical m/z could not be calculated; ensure the charge state "
                    " is set."
                )
            else:
                warnings.warn("precursor_mz not set, using theoretical precursor m/z.")
                precursor_mz_float = float(self.peptidoform.theoretical_mz)

        spectrum = sus.MsmsSpectrum(
            identifier=self.identifier if self.identifier else "spectrum",
            precursor_mz=precursor_mz_float,
            precursor_charge=precursor_charge,
            mz=self.mz,
            intensity=self.intensity,
            retention_time=self.retention_time if self.retention_time is not None else 0.0,
        )
        if (
            self.peptidoform
            and self.mass_tolerance is not None
            and self.mass_tolerance_unit is not None
        ):
            spectrum.annotate_proforma(
                str(self.peptidoform),
                self.mass_tolerance,
                self.mass_tolerance_unit,
            )
        return spectrum


class ObservedSpectrum(Spectrum):
    """Observed MS2 spectrum."""

    pass


class PredictedSpectrum(Spectrum):
    """Predicted MS2 spectrum."""

    mass_tolerance: float | None = 0.001
    mass_tolerance_unit: str | None = "Da"
