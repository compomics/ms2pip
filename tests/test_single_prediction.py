from ms2pip.single_prediction import SinglePrediction
from ms2pip.exceptions import InvalidPeptideError, InvalidModificationFormattingError

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
    def test_validate_mod_string(self):
        for mod_string, is_valid in MOD_TEST_CASES.items():
            try:
                SinglePrediction._validate_mod_string(mod_string)
            except InvalidModificationFormattingError:
                passed = False
            else:
                passed = True
            assert passed == is_valid, f"{mod_string} incorrectly marked as {passed}"

    def test_validate_sequence(self):
        for peptide, is_valid in PEPTIDE_TEST_CASES.items():
            try:
                SinglePrediction._validate_sequence(peptide)
            except InvalidPeptideError:
                passed = False
            else:
                passed = True
            assert passed == is_valid, f"{peptide} incorrectly marked as {passed}"
