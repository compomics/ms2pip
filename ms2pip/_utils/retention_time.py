import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psm_utils.io.peptide_record as peptide_record

logger = logging.getLogger(__name__)

# Reduce Tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class RetentionTime:
    def __init__(self, predictor="deeplc", config=None, processes=None):
        """
        Initialize peptide retention time predictor

        Parameters
        ----------
        predictor: str, optional
            Retention time predictor to employ. Currently only 'deeplc' is supported.
        config: dict, optional
            Dictionary with configuration. Requires 'deeplc' top-level key for
            DeepLC predictions.
        """
        self.predictor = predictor
        self.deeplc_predictor = None

        if not config:
            self.config = dict()
        else:
            self.config = config

        if "deeplc" not in self.config:
            self.config["deeplc"] = {
                "verbose": False,
                "calibration_file": None,
                "n_jobs": processes,
            }

    def _get_irt_peptides(self):
        """
        Return DeepLC DataFrame with iRT peptides
        """
        irt_peptides = {
            "LGGNEQVTR": -24.92,
            "GAGSSEPVTGLDAK": 0.00,
            "VEATFGVDESNAK": 12.39,
            "YILAGVENSK": 19.79,
            "TPVISGGPYEYR": 28.71,
            "TPVITGAPYEYR": 33.38,
            "DGLDAASYYAPVR": 42.26,
            "ADVTPADFSEWSK": 54.62,
            "GTFIIDPGGVIR": 70.52,
            "GTFIIDPAAVIR": 87.23,
            "LFLQFGAQGSPFLK": 100.00,
        }

        irt_df = pd.DataFrame.from_dict(irt_peptides, orient="index")
        irt_df = irt_df.reset_index()
        irt_df.columns = ["seq", "tr"]
        irt_df["modifications"] = ""

        return irt_df

    def _init_deeplc(self):
        """
        Initialize DeepLC: import, configurate and calibrate
        """
        # Only import if DeepLC will be used, otherwise lots of extra heavy
        # dependencies (e.g. Tensorflow) are imported as well
        import deeplc

        deeplc_params = self.config["deeplc"]
        if "calibration_file" in deeplc_params and deeplc_params["calibration_file"]:
            cal_df = pd.read_csv(deeplc_params["calibration_file"], sep=",")
        else:
            cal_df = self._get_irt_peptides()
            deeplc_params["split_cal"] = 9  # Only 11 iRT peptides, so use 9 splits
            best_model_for_irt = (
                Path(deeplc.__file__).parent
                / "mods/full_hc_PXD005573_mcp_cb975cfdd4105f97efa0b3afffe075cc.hdf5"
            )
            if best_model_for_irt.exists():
                deeplc_params["path_model"] = str(best_model_for_irt)

            # PyGAM results in strange calibration on limited set of iRT peptides
            deeplc_params["pygam_calibration"] = False

        # Remove from deeplc_params to control calibration here instead of in DeepLC
        if "calibration_file" in deeplc_params:
            del deeplc_params["calibration_file"]

        self.deeplc_predictor = deeplc.DeepLC(**deeplc_params)

        logger.info("Calibrating DeepLC...")
        self.deeplc_predictor.calibrate_preds(seq_df=cal_df)

    def _prepare_deeplc_peptide_df(self):
        """
        Prepare DeepLC peptide DataFrame
        """
        column_map = {"peptide": "seq", "modifications": "modifications"}
        peprec = peptide_record.to_dataframe(self.psm_list)[column_map.keys()]
        self.deeplc_pep_df = peprec.rename(columns=column_map)

    def _run_deeplc(self):
        """Run DeepLC."""
        logger.info("Predicting retention times with DeepLC...")
        self.deeplc_preds = self.deeplc_predictor.make_preds(
            seq_df=self.deeplc_pep_df.fillna("")
        )

    def _parse_deeplc_preds(self):
        """Add DeepLC predictions to peprec DataFrame."""
        self.psm_list["retention_time"] = np.array(self.deeplc_preds, dtype=np.float32)

    def _predict_deeplc(self):
        """Predict retention times using DeepLC."""
        if not self.deeplc_predictor:
            self._init_deeplc()
        self._prepare_deeplc_peptide_df()
        self._run_deeplc()
        self._parse_deeplc_preds()

    def add_rt_predictions(self, psm_list):
        """
        Run RT predictor and add predictions to a PSMList.

        psm_list: PSMList
        """
        self.psm_list = psm_list

        if self.predictor == "deeplc":
            self._predict_deeplc()
        else:
            raise NotImplementedError(self.predictor)
