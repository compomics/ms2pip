"""Definition and handling of MS²PIP results."""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional, Tuple
from logging import getLogger

import numpy as np
from psm_utils import PSM
from pydantic import BaseModel, ConfigDict

try:
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus
except ImportError:
    sus = None
    sup = None

from ms2pip.spectrum import ObservedSpectrum, PredictedSpectrum

logger = getLogger(__name__)

class ProcessingResult(BaseModel):
    """Result of processing a single PSM."""

    psm_index: int
    psm: Optional[PSM] = None
    theoretical_mz: Optional[Dict[str, np.ndarray]] = None
    predicted_intensity: Optional[Dict[str, np.ndarray]] = None
    observed_intensity: Optional[Dict[str, np.ndarray]] = None
    correlation: Optional[float] = None
    feature_vectors: Optional[np.ndarray] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(__pydantic_self__, **data: Any) -> None:
        """Result of processing a single PSM."""
        super().__init__(**data)

    def as_spectra(self) -> Tuple[Optional[PredictedSpectrum], Optional[ObservedSpectrum]]:
        """Convert result to predicted and observed spectra."""
        if not self.theoretical_mz:
            raise ValueError("Theoretical m/z values required to convert to spectra.")
        mz = np.concatenate([i for i in self.theoretical_mz.values()])
        annotations = np.concatenate(
            [
                [ion_type + str(i + 1) for i in range(len(peaks))]
                for ion_type, peaks in self.theoretical_mz.items()
            ]
        )
        peak_order = np.argsort(mz)

        if self.predicted_intensity:
            pred_int = np.concatenate([i for i in self.predicted_intensity.values()])
            pred_int = (2**pred_int) - 0.001  # Unlog intensities
            predicted = PredictedSpectrum(
                mz=mz[peak_order],
                intensity=pred_int[peak_order],
                annotations=annotations[peak_order],
                peptidoform=self.psm.peptidoform if self.psm else None,
                precursor_charge=self.psm.peptidoform.precursor_charge if self.psm else None,
            )
        else:
            predicted = None

        if self.observed_intensity:
            obs_int = np.concatenate([i for i in self.observed_intensity.values()])
            obs_int = (2**obs_int) - 0.001  # Unlog intensities
            observed = ObservedSpectrum(
                mz=mz[peak_order],
                intensity=obs_int[peak_order],
                annotations=annotations[peak_order],
                peptidoform=self.psm.peptidoform if self.psm else None,
                precursor_charge=self.psm.peptidoform.precursor_charge if self.psm else None,
            )
        else:
            observed = None

        return predicted, observed

    def plot_spectra(self):
        """
        Plot predicted and observed spectra.

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        -----
        Requires optional dependency ``spectrum_utils`` to be installed.

        """
        predicted, observed = (
            spec.to_spectrum_utils() if spec else None for spec in self.as_spectra()
        )
        if predicted and observed:
            ax = sup.mirror(observed, predicted)
            ax.set_title(
                f"Observed (top) and predicted (bottom) spectra for {self.psm.peptidoform}"
            )
        elif predicted:
            ax = sup.spectrum(predicted)
            ax.set_title(f"Predicted spectrum for {self.psm.peptidoform}")
        elif observed:
            ax = sup.spectrum(observed)
            ax.set_title(f"Observed spectrum for {self.psm.peptidoform}")
        else:
            raise ValueError("No spectra to plot.")
        return ax


def calculate_correlations(results: List[ProcessingResult]) -> None:
    """Calculate and add Pearson correlations to list of results."""
    for result in results:
        pred_int = np.concatenate([i for i in result.predicted_intensity.values()])
        obs_int = np.concatenate([i for i in result.observed_intensity.values()])
        result.correlation = np.corrcoef(pred_int, obs_int)[0][1]


def write_correlations(results: List["ProcessingResult"], output_file: str) -> None:
    """Write correlations to CSV file."""
    with open(output_file, "wt") as f:
        fieldnames = ["psm_index", "correlation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({"psm_index": result.psm_index, "correlation": result.correlation})
