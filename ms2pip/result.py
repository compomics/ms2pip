"""Definition and handling of MS²PIP results."""

from __future__ import annotations

import csv
from logging import getLogger
from pathlib import Path
import numpy as np
from psm_utils import PSM
from pydantic import BaseModel, ConfigDict

try:
    import spectrum_utils.plot as sup
except ImportError:
    sup = None  # type: ignore[ty:invalid-assignment]

from ms2pip.correlation import pearson
from ms2pip.spectrum import ObservedSpectrum, PredictedSpectrum

logger = getLogger(__name__)


class ProcessingResult(BaseModel):
    """
    Result of processing a single PSM.

    Parameters
    ----------
    psm_index
        Index of the PSM in the input list.
    psm
        The PSM object.
    theoretical_mz
        Dict mapping ion type to theoretical m/z array.
    predicted_intensity
        Dict mapping ion type to predicted intensity array.
    observed_intensity
        Dict mapping ion type to observed intensity array.
    correlation
        Pearson correlation between predicted and observed intensities.
    feature_vectors
        Feature vectors for model training.
    """

    psm_index: int
    psm: PSM
    theoretical_mz: dict[str, np.ndarray] | None = None
    predicted_intensity: dict[str, np.ndarray] | None = None
    observed_intensity: dict[str, np.ndarray] | None = None
    correlation: float | None = None
    feature_vectors: np.ndarray | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def as_spectra(self) -> tuple[PredictedSpectrum | None, ObservedSpectrum | None]:
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
            pred_int = np.concatenate(list(self.predicted_intensity.values()))
            predicted = PredictedSpectrum(
                mz=mz[peak_order],
                intensity=pred_int[peak_order],
                annotations=annotations[peak_order],
                peptidoform=self.psm.peptidoform if self.psm else None,
                precursor_charge=self.psm.peptidoform.precursor_charge if self.psm else None,
            )
            predicted.inverse_log2_transform()
        else:
            predicted = None

        if self.observed_intensity:
            obs_int = np.concatenate(list(self.observed_intensity.values()))
            observed = ObservedSpectrum(
                mz=mz[peak_order],
                intensity=obs_int[peak_order],
                annotations=annotations[peak_order],
                peptidoform=self.psm.peptidoform if self.psm else None,
                precursor_charge=self.psm.peptidoform.precursor_charge if self.psm else None,
            )
            observed.inverse_log2_transform()
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


def calculate_correlations(results: list[ProcessingResult]) -> None:
    """Calculate and add Pearson correlations to list of results."""
    # TODO: Consider nan values? https://github.com/CompOmics/ms2pip/pull/214/changes#diff-5f77421a48cf8f17c5b83ed031897ce8076e2f52c0028aa6f4294a34ba3b3305R115-R123
    for result in results:
        if result.predicted_intensity is None or result.observed_intensity is None:
            continue
        pred_int = np.concatenate(list(result.predicted_intensity.values()))
        obs_int = np.concatenate(list(result.observed_intensity.values()))
        result.correlation = pearson(pred_int, obs_int)


def write_correlations(results: list[ProcessingResult], output_file: str | Path) -> None:
    """Write correlations to CSV file."""
    with open(output_file, "wt") as f:
        fieldnames = ["psm_index", "correlation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({"psm_index": result.psm_index, "correlation": result.correlation})
