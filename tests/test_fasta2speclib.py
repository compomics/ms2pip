"""Tests for fasta2speclib."""

from pyteomics.fasta import Protein

from fasta2speclib.fasta2speclib import Fasta2SpecLib, ModificationConfig, Peptide

MODIFICATION_CONFIG = [
    {
        "name": "Oxidation",
        "mass_shift": 15.9994,
        "amino_acid": "M",
    },
    {
        "name": "Carbamidomethyl",
        "mass_shift": 57.0513,
        "amino_acid": "C",
        "fixed": True,
    },
    {
        "name": "Glu->pyro-Glu",
        "mass_shift": -18.010565,
        "amino_acid": "E",
        "peptide_n_term": True,
    },
    {
        "name": "Acetyl",
        "mass_shift": 42.010565,
        "amino_acid": None,
        "protein_n_term": True,
    },
]
MODIFICATION_CONFIG = [ModificationConfig(**mod) for mod in MODIFICATION_CONFIG]


def test_get_modifications_by_target():
    modifications_by_target = Fasta2SpecLib._get_modifications_by_target(MODIFICATION_CONFIG)
    assert modifications_by_target["sidechain"] == {"M": [None] + MODIFICATION_CONFIG[0:1]}
    assert modifications_by_target["peptide_n_term"] == {"E": [None] + MODIFICATION_CONFIG[2:3]}
    assert modifications_by_target["peptide_c_term"] == {}
    assert modifications_by_target["protein_n_term"] == {"any": [None] + MODIFICATION_CONFIG[3:4]}
    assert modifications_by_target["protein_c_term"] == {}


def test_get_modification_versions():
    modification_config = [
        ModificationConfig(
            **{
                "name": "Oxidation",
                "mass_shift": 15.9994,
                "amino_acid": "M",
            }
        ),
        ModificationConfig(
            **{
                "name": "Carbamidomethyl",
                "mass_shift": 57.0513,
                "amino_acid": "C",
                "fixed": True,
            }
        ),
        ModificationConfig(
            **{
                "name": "Glu->pyro-Glu",
                "mass_shift": -18.010565,
                "amino_acid": "E",
                "protein_n_term": True,
            }
        ),
    ]
    modifications_by_target = Fasta2SpecLib._get_modifications_by_target(modification_config)

    test_cases = [
        ("ADEF", {""}),  # None
        ("ACDE", {"2|Carbamidomethyl"}),  # Single fixed
        ("ACCDE", {"2|Carbamidomethyl|3|Carbamidomethyl"}),  # Double fixed
        ("ADME", {"", "3|Oxidation"}),  # Single variable
        (
            "ADMME",
            {"", "3|Oxidation", "4|Oxidation", "3|Oxidation|4|Oxidation"},
        ),  # Double variable
        (
            "ADMMME",
            {
                "",
                "3|Oxidation",
                "4|Oxidation",
                "5|Oxidation",
                "3|Oxidation|4|Oxidation",
                "4|Oxidation|5|Oxidation",
                "3|Oxidation|5|Oxidation",
            },
        ),  # More than maximum simultaneous mods should be ignored
        ("EDEF", {"", "0|Glu->pyro-Glu"}),  # N-term and AA-specific
    ]

    for peptide, expected_output in test_cases:
        output = Fasta2SpecLib._get_modification_versions(
            Peptide(sequence=peptide, is_n_term=True, proteins=[]),
            modification_config,
            modifications_by_target,
            max_variable_modifications=2,
        )
        assert set(output) == expected_output


def test_digest_protein():
    test_input = {
        "protein": Protein(
            description="P12345",
            sequence="MYSSCSLLQRLVWFPFLALVATQLLFIRNVSSLNLTNEYLHHKCLVSEGKYKPGSKYEYI",
        ),
        "min_length": 8,
        "max_length": 30,
        "cleavage_rule": "trypsin",
        "missed_cleavages": 2,
        "semi_specific": False,
    }

    test_output = [
        Peptide(
            sequence="MYSSCSLLQR",
            proteins=["P12345"],
            modification_options=None,
            is_n_term=True,
            is_c_term=False,
        ),
        Peptide(
            sequence="MYSSCSLLQRLVWFPFLALVATQLLFIR",
            proteins=["P12345"],
            modification_options=None,
            is_n_term=True,
            is_c_term=False,
        ),
        Peptide(
            sequence="LVWFPFLALVATQLLFIR",
            proteins=["P12345"],
            modification_options=None,
            is_n_term=False,
            is_c_term=False,
        ),
        Peptide(
            sequence="NVSSLNLTNEYLHHK",
            proteins=["P12345"],
            modification_options=None,
            is_n_term=False,
            is_c_term=False,
        ),
        Peptide(
            sequence="NVSSLNLTNEYLHHKCLVSEGK",
            proteins=["P12345"],
            modification_options=None,
            is_n_term=False,
            is_c_term=False,
        ),
        Peptide(
            sequence="NVSSLNLTNEYLHHKCLVSEGKYKPGSK",
            proteins=["P12345"],
            modification_options=None,
            is_n_term=False,
            is_c_term=False,
        ),
        Peptide(
            sequence="CLVSEGKYKPGSK",
            proteins=["P12345"],
            modification_options=None,
            is_n_term=False,
            is_c_term=False,
        ),
        Peptide(
            sequence="CLVSEGKYKPGSKYEYI",
            proteins=["P12345"],
            modification_options=None,
            is_n_term=False,
            is_c_term=True,
        ),
        Peptide(
            sequence="YKPGSKYEYI",
            proteins=["P12345"],
            modification_options=None,
            is_n_term=False,
            is_c_term=True,
        ),
    ]

    assert test_output == Fasta2SpecLib._digest_protein(**test_input)
