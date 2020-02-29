import pandas as pd

from ms2pip.ms2pip_tools.spectrum_output import SpectrumOutput

def test_spectrum_output():
    pass

    """
    peprec = pd.read_pickle("peprec_prot_test_HCDpeprec.pkl")
    all_preds = pd.read_pickle("peprec_prot_test_HCDall_preds.pkl")

    params = {
        "ptm": [
            "Oxidation,15.994915,opt,M",
            "Carbamidomethyl,57.021464,opt,C",
            "Glu->pyro-Glu,-18.010565,opt,E",
            "Gln->pyro-Glu,-17.026549,opt,Q",
            "Acetyl,42.010565,opt,N-term",
        ],
        "sptm": [],
        "gptm": [],
        "model": "HCD",
        "frag_error": "0.02",
        "out": "csv",
    }

    for i in range(2):
        peprec_tmp = peprec.sample(5).copy()
        all_preds_tmp = all_preds[all_preds["spec_id"].isin(peprec_tmp["spec_id"])].copy()

        write_mode = 'wt+' if i == 0 else 'at'

        so = SpectrumOutput(
            all_preds_tmp,
            peprec_tmp,
            params,
            output_filename="new",
            write_mode=write_mode,
            return_stringbuffer=False,
        )

        so.write_mgf()
        so.write_msp()
        so.write_spectronaut()
        so.write_bibliospec()
    """