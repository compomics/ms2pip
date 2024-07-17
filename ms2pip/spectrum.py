"""MS2 spectrum handling."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
from psm_utils import Peptidoform
from pydantic import model_validator, field_validator, ConfigDict, BaseModel
try:
    import spectrum_utils.spectrum as sus
    import spectrum_utils.plot as sup
except ImportError:
    sus = None
    sup = None


class Spectrum(BaseModel):
    """MS2 spectrum."""

    mz: np.ndarray
    intensity: np.ndarray
    annotations: Optional[np.ndarray] = None
    identifier: Optional[str] = None
    peptidoform: Optional[Union[Peptidoform, str]] = None
    precursor_mz: Optional[float] = None
    precursor_charge: Optional[int] = None
    retention_time: Optional[float] = None
    mass_tolerance: Optional[float] = None
    mass_tolerance_unit: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(__pydantic_self__, **data: Any) -> None:
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
        super().__init__(**data)

    def __repr__(self) -> str:
        return "{}.{}({})".format(
            self.__class__.__module__,
            self.__class__.__qualname__,
            f"identifier='{self.identifier}'",
        )

    @model_validator(mode="after")
    @classmethod
    def check_array_lengths(cls, data: dict):
        if len(data.mz) != len(data.intensity):
            raise ValueError("Array lengths do not match.")
        if data.annotations is not None:
            if len(data.annotations) != len(data.intensity):
                raise ValueError("Array lengths do not match.")
        return data

    @field_validator("peptidoform")
    @classmethod
    def check_peptidoform(cls, value):
        if not value or isinstance(value, Peptidoform):
            return value
        elif isinstance(value, str):
            return Peptidoform(value)
        else:
            raise ValueError("Peptidoform must be a string, a Peptidoform object, or None.")

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
        """Log2-tranform spectrum."""
        self.intensity = np.log2(self.intensity + 0.001)

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
        if not sus:
            raise ImportError("Optional dependency spectrum_utils not installed.")

        if self.precursor_charge:
            precursor_charge = self.precursor_charge
        else:
            if not self.peptidoform:
                raise ValueError("`precursor_charge` or `peptidoform` must be set.")
            else:
                precursor_charge = self.peptidoform.precursor_charge

        if self.precursor_mz:
            precursor_mz = self.precursor_mz
        else:
            if not self.peptidoform:
                raise ValueError("`precursor_mz` or `peptidoform` must be set.")
            else:
                warnings.warn("precursor_mz not set, using theoretical precursor m/z.")
                precursor_mz = self.peptidoform.theoretical_mz

        spectrum = sus.MsmsSpectrum(
            identifier=self.identifier if self.identifier else "spectrum",
            precursor_mz=precursor_mz,
            precursor_charge=precursor_charge,
            mz=self.mz,
            intensity=self.intensity,
            retention_time=self.retention_time,
        )
        if self.peptidoform:
            spectrum.annotate_proforma(
                str(self.peptidoform), self.mass_tolerance, self.mass_tolerance_unit
            )
        return spectrum


class ObservedSpectrum(Spectrum):
    """Observed MS2 spectrum."""

    pass


class PredictedSpectrum(Spectrum):
    """Predicted MS2 spectrum."""

    mass_tolerance: Optional[float] = 0.001
    mass_tolerance_unit: Optional[str] = "Da"

    pass
