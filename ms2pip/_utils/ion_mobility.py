"""Module for ion mobility prediction with IM²Deep."""

import logging

import pandas as pd
from psm_utils import PSMList

logger = logging.getLogger(__name__)


class IonMobility:
    """Predict ion mobility using IM2Deep."""

    def __init__(self, processes=1) -> None:
        # Lazy import to avoid loading loading heavy dependencies when not needed
        try:
            from im2deep.im2deep import predict_ccs  # noqa: F401

            self.predict_fn = predict_ccs
            self.processes = processes
        except ImportError as e:
            raise ImportError(
                "The 'im2deep' package is required for ion mobility prediction."
            ) from e

    def add_im_predictions(self, psm_list: PSMList) -> None:
        """Add ion mobility predictions to the PSMList."""
        logger.info("Predicting ion mobility...")
        predictions: pd.Series = self.predict_fn(
            psm_list, write_output=False, n_jobs=self.processes, ion_mobility=True
        )
        psm_list["ion_mobility"] = predictions.values
