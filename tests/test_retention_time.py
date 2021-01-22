import os

import numpy as np
import pandas as pd

from ms2pip.retention_time import RetentionTime
from ms2pip.config_parser import ConfigParser


TEST_DIR = os.path.dirname(__file__)

class TestRetentionTime:
    def test_prepare_deeplc_peptide_df(self):
        peprec = pd.read_csv(os.path.join(TEST_DIR, "test_data/test.peprec"), sep=" ")
        config = {
            "deeplc": {
                "calibration_file": False,
                "verbose": False,
                "path_model": False,
                "split_cal": 25,
                "batch_num": 350000,
            }
        }

        rt_predictor = RetentionTime(config=config)
        rt_predictor.peprec = peprec
        rt_predictor._prepare_deeplc_peptide_df()
        dlc_df = rt_predictor.deeplc_pep_df

        assert dlc_df.equals(
            pd.DataFrame({
                "seq": {0: "ACDE", 1: "ACDEFGHI", 2: "ACDEFGHIKMNPQ"},
                "modifications": {0: np.nan, 1: np.nan, 2: np.nan},
            })
        )
