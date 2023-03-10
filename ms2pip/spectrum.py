"""MS2 spectrum handling."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, root_validator


class Spectrum(BaseModel):
    """MS2 spectrum."""

    identifier: str
    mz: np.ndarray
    intensity: np.ndarray
    annotations: Optional[np.ndarray] = None
    peptidoform: Optional[str] = None
    precursor_mz: Optional[float] = None
    precursor_charge: Optional[int] = None
    retention_time: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(__pydantic_self__, **data: Any) -> None:
        """
        MS2 spectrum.

        Parameters
        ----------
        identifier
            Spectrum identifier.
        mz
            Array of m/z values.
        intensity
            Array of intensity values.
        annotations
            Array of peak annotations.
        peptidoform
            Peptidoform annotation.
        precursor_mz
            Precursor m/z.
        precursor_charge
            Precursor charge.
        retention_time
            Retention time.

        """
        super().__init__(**data)

    def __repr__(self) -> str:
        return "{}.{}({})".format(
            self.__class__.__module__,
            self.__class__.__qualname__,
            f"title='{self.identifier}'",
        )

    @root_validator()
    def check_array_lengths(cls, values):
        if len(values["mz"]) != len(values["intensity"]):
            raise ValueError("Array lengths do not match.")
        if values["annotations"] is not None:
            if len(values["annotations"]) != len(values["intensity"]):
                raise ValueError("Array lengths do not match.")
        return values

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

    def to_spectrum_utils(self):
        """Convert to spectrum_utils.MsmsSpectrum."""
        if not self.precursor_charge or not self.precursor_mz:
            raise ValueError("Precursor charge and m/z must be set.")
        from spectrum_utils.spectrum import MsmsSpectrum

        spectrum = MsmsSpectrum(
            identifier=str(self.peptidoform) if self.peptidoform else "spectrum",
            precursor_mz=self.precursor_mz,
            precursor_charge=self.precursor_charge,
            mz=self.mz,
            intensity=self.intensity,
            retention_time=self.retention_time,
        )
        if self.peptidoform:
            spectrum.annotate_proforma(self.peptidoform)
        return spectrum


class ObservedSpectrum(Spectrum):
    """Observed MS2 spectrum."""

    pass


class PredictedSpectrum(Spectrum):
    """Predicted MS2 spectrum."""

    pass
