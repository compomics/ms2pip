"""Definition and handling of MS²PIP results."""
from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel
from psm_utils import PSM

from ms2pip.spectrum import ObservedSpectrum, PredictedSpectrum


class ProcessingResult(BaseModel):
    """Result of processing a single PSM."""

    psm: PSM
    theoretical_mz: Optional[Dict[str, np.ndarray]] = None
    predicted_intensity: Optional[Dict[str, np.ndarray]] = None
    observed_intensity: Optional[Dict[str, np.ndarray]] = None
    correlation: Optional[float] = None
    feature_vectors: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True

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
            pred_int = (2 ** pred_int[peak_order]) - 0.001  # Unlog intensities
            predicted = PredictedSpectrum(
                identifier=self.psm_id,
                mz=mz[peak_order],
                intensity=pred_int[peak_order],
                annotations=annotations[peak_order],
            )
        else:
            predicted = None

        if self.observed_intensity:
            obs_int = np.concatenate([i for i in self.observed_intensity.values()])
            obs_int = (2 ** pred_int[peak_order]) - 0.001  # Unlog intensities
            observed = ObservedSpectrum(
                identifier=self.psm_id,
                mz=mz[peak_order],
                intensity=obs_int[peak_order],
                annotations=annotations[peak_order],
            )
        else:
            observed = None

        return predicted, observed


def calculate_correlations(results: List[ProcessingResult]) -> None:
    """Calculate and add Pearson correlations to list of results."""
    for result in results:
        pred_int = np.concatenate([i for i in result.predicted_intensity.values()])
        obs_int = np.concatenate([i for i in result.observed_intensity.values()])
        result.correlation = np.corrcoef(pred_int, obs_int)[0][1]


def results_to_csv(results: List["ProcessingResult"], output_file: str) -> None:
    """Write processing results to CSV file."""
    with open(output_file, "wt") as f:
        fieldnames = [
            # "psm_id",
            "ion_type",
            "ion_number",
            "mz",
            "predicted",
            "observed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for ion_type in result.theoretical_mz:
                for i in range(len(result.theoretical_mz[ion_type])):
                    writer.writerow(
                        {
                            # "psm_id": result.psm_id,
                            "ion_type": ion_type,
                            "ion_number": i + 1,
                            "mz": "{:.6g}".format(result.theoretical_mz[ion_type][i]),
                            "predicted": "{:.6g}".format(result.predicted_intensity[ion_type][i]),
                            "observed": "{:.6g}".format(result.observed_intensity[ion_type][i])
                            if result.observed_intensity
                            else None,
                        }
                    )


def correlations_to_csv(results: List["ProcessingResult"], output_file: str) -> None:
    """Write correlations to CSV file."""
    with open(output_file, "wt") as f:
        fieldnames = ["psm_id", "correlation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({"psm_id": result.psm_id, "correlation": result.correlation})
