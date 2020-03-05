import pandas as pd

# from deeplc import DeepLC


class RetentionTime:
    def __init__(self, peprec, config=None):
        """
        Add peptide retention time predictions to peprec.

        Parameters
        ----------
        peprec: pd.DataFrame
            MS2PIP-style peprec DataFrame with peptides for which to predict retention
            times
        config: dict, optional
            Dictionary with configuration. E.g. requires 'deeplc' top-level key for
            DeepLC predictions.
        """
        self.peprec = peprec
        if not config:
            self.config = dict()
        else:
            self.config = config

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

    def _prepare_deeplc_peptide_df(self):
        column_map = {"peptide": "seq", "modifications": "modifications"}
        self.deeplc_pep_df = self.peprec[column_map.keys()].copy()
        self.deeplc_pep_df.rename(columns=column_map, inplace=True)

    def _run_deeplc(self):
        # Only import if DeepLC will be used, otherwise lot's of extra heavy
        # dependencies (e.g. Tensorflow) are imported as well
        from deeplc import DeepLC

        if "deeplc" in self.config:
            deeplc_params = self.config["deeplc"]
            calibration_file = deeplc_params["calibration_file"]
            del deeplc_params["calibration_file"]
        else:
            deeplc_params = {"verbose": False}
            calibration_file = None

        # TODO: Remove when fixed upstream in DeepLC
        if not calibration_file:
            deeplc_params["split_cal"] = 9

        dlc = DeepLC(**deeplc_params)

        if calibration_file:
            cal_df = pd.read_csv(calibration_file, sep=",")
        else:
            dlc.calibrate_preds(seq_df=self._get_irt_peptides())

        self.deeplc_preds = dlc.make_preds(seq_df=self.deeplc_pep_df.fillna(""))

    def _parse_deeplc_preds(self):
        self.peprec["rt"] = self.deeplc_preds

    def _predict_deeplc(self):
        self._prepare_deeplc_peptide_df()
        self._run_deeplc()
        self._parse_deeplc_preds()

    def add_rt_predictions(self, predictor="deeplc"):
        if predictor == "deeplc":
            self._predict_deeplc()
        else:
            raise NotImplementedError(predictor)
