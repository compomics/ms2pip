"""Tests for fasta2speclib."""

from fasta2speclib.fasta2speclib import (
    get_modification_versions,
    get_modifications_by_target,
)


def test_get_modification_versions():
    modification_config = [
        {
            "name": "Oxidation",
            "unimod_accession": 35,
            "mass_shift": 15.9994,
            "amino_acid": "M",
            "n_term": False,
            "fixed": False,
        },
        {
            "name": "Carbamidomethyl",
            "unimod_accession": 4,
            "mass_shift": 57.0513,
            "amino_acid": "C",
            "n_term": False,
            "fixed": True,
        },
        {
            "name": "Glu->pyro-Glu",
            "unimod_accession": 27,
            "mass_shift": -18.010565,
            "amino_acid": "E",
            "n_term": True,
            "fixed": False,
        },
    ]
    mods_sidechain, mods_nterm = get_modifications_by_target(modification_config)

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
        output = get_modification_versions(
            peptide, modification_config, mods_sidechain, mods_nterm, max_mods=2
        )
        assert set(output) == expected_output
