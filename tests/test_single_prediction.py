from ms2pip.exceptions import (InvalidModificationFormattingError,
                               InvalidPeptideError)
from ms2pip.single_prediction import SinglePrediction

MOD_TEST_CASES = {
    "": True,
    "-": True,
    "17|Cmm": True,
    "3|Cmm|11|Cmm": True,
    "7|Cm-:?M": True,
    "10|Cmm|": True,
    "-1|Oxidation": True,
    "|10|Cmm": False,
    "1|Pyro_glu ": False,
    "6|Cmm |25|Cmm": False,
    "221|Cmm": False,
    "-4|Oxidation": False,
}

PEPTIDE_TEST_CASES = {
    "HKA": True,
    "SETAPLAPTIPAPAEK": True,
    "TICIETIKGTCWQTVIDGR": True,
    "LASYAVBYR": False,
    "SPLTICYPEYTGSNTYEEAAAYIQCQFEDLNRR": True,
    "-KAVASVAK-": False,
    "VM(ox)SAFVEIIFDNRLPIDKEEVSLR": False,
    "HK": False,
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA": False,
}


class TestSinglePrediction:
    # TODO: Implement tests
    pass
