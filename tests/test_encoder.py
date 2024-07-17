import pytest
from psm_utils import Peptidoform, PSM, PSMList

from ms2pip._utils.encoder import Encoder


class TestEncoder:
    def test_from_peptidoform(self):
        test_cases = [
            # Peptidoform, {(target, label): (amino_acid, amino_acid_id, mass_shift)}
            ("ACDEK", {}),
            ("AC[+57.021464]DEK", {("C", "+57.021464"): ("C", 1, 57.021464)}),
            ("AC[U:4]", {("C", "UNIMOD:4"): ("C", 1, 57.021464)}),
            ("AC[formula:H3C2NO]", {("C", "Formula:H3C2NO"): ("C", 1, 57.021464)}),
            ("[Acetyl]-ACDE", {("n_term", "Acetyl"): ("n_term", -1, 42.010565)}),
            ("ACDE-[Amidated]", {("c_term", "Amidated"): ("c_term", -2, -0.984016)}),
            (
                "AC[+57.021464]DE-[Amidated]",
                {
                    ("C", "+57.021464"): ("C", 1, 57.021464),
                    ("c_term", "Amidated"): ("c_term", -2, -0.984016),
                },
            ),
            (
                "[Acetyl]-AC[+57.021464]DE",
                {
                    ("n_term", "Acetyl"): ("n_term", -1, 42.010565),
                    ("C", "+57.021464"): ("C", 1, 57.021464),
                },
            ),
        ]

        for peptidoform, expected_mods in test_cases:
            encoder = Encoder.from_peptidoform(Peptidoform(peptidoform))
            for key, modification in encoder.modifications.items():
                for item_key, expected_item in zip(
                    ["amino_acid", "amino_acid_id", "mass_shift"], expected_mods[key]
                ):
                    if isinstance(expected_item, float):
                        assert modification[item_key] == pytest.approx(expected_item)
                    else:
                        assert modification[item_key] == expected_item

    def test_from_psm_list(self):
        psm_list = PSMList(psm_list=[
            PSM(peptidoform="AC[+57.021464]DEK", spectrum_id=0),
            PSM(peptidoform="AC[U:4]", spectrum_id=1),
            PSM(peptidoform="AC[formula:H3C2NO]", spectrum_id=2),
            PSM(peptidoform="[Acetyl]-ACDE", spectrum_id=3),
            PSM(peptidoform="ACDE-[Amidated]",spectrum_id= 4)
        ])
        expected = {
            ("C", "+57.021464"): {
                "mod_id": 38,
                "mass_shift": 57.021464,
                "amino_acid": "C",
                "amino_acid_id": 1,
            },
            ("C", "UNIMOD:4"): {
                "mod_id": 39,
                "mass_shift": 57.021464,
                "amino_acid": "C",
                "amino_acid_id": 1,
            },
            ("C", "Formula:H3C2NO"): {
                "mod_id": 40,
                "mass_shift": 57.02146372057,
                "amino_acid": "C",
                "amino_acid_id": 1,
            },
            ("n_term", "Acetyl"): {
                "mod_id": 41,
                "mass_shift": 42.010565,
                "amino_acid": "n_term",
                "amino_acid_id": -1,
            },
            ("c_term", "Amidated"): {
                "mod_id": 42,
                "mass_shift": -0.984016,
                "amino_acid": "c_term",
                "amino_acid_id": -2,
            },
        }

        encoder = Encoder.from_psm_list(psm_list)
        for modification_key, modification_dict in encoder.modifications.items():
            for item_key, expected_item in expected[modification_key].items():
                if isinstance(expected_item, float):
                    assert modification_dict[item_key] == pytest.approx(expected_item)
                else:
                    assert modification_dict[item_key] == expected_item
