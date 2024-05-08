from ms2pip import search_space

OXIDATION = search_space.ModificationConfig(
    label="Oxidation",
    amino_acid="M",
)
CARBAMIDOMETHYL = search_space.ModificationConfig(
    label="Carbamidomethyl",
    amino_acid="C",
    fixed=True,
)
PYROGLU = search_space.ModificationConfig(
    label="Glu->pyro-Glu",
    amino_acid="E",
    peptide_n_term=True,
)
ACETYL = search_space.ModificationConfig(
    label="Acetyl",
    amino_acid=None,
    protein_n_term=True,
)
PHOSPHO = search_space.ModificationConfig(
    label="Phospho",
    amino_acid="T",
    fixed=False,
)

MODIFICATION_CONFIG = [OXIDATION, CARBAMIDOMETHYL, PYROGLU, ACETYL]


def test_restructure_modifications_by_target():
    test_cases = [
        {
            "modifications": [PHOSPHO, ACETYL],
            "expected": {
                "sidechain": {"T": [PHOSPHO]},
                "peptide_n_term": {},
                "peptide_c_term": {},
                "protein_n_term": {"any": [ACETYL]},
                "protein_c_term": {},
            },
        },
        {
            "modifications": [CARBAMIDOMETHYL, ACETYL],
            "expected": {
                "sidechain": {},
                "peptide_n_term": {},
                "peptide_c_term": {},
                "protein_n_term": {"any": [ACETYL]},
                "protein_c_term": {},
            },
        },
    ]

    for case in test_cases:
        test_out = search_space._restructure_modifications_by_target(case["modifications"])
        assert test_out == case["expected"]


def test_get_peptidoform_modification_versions():
    test_cases = [
        # None
        {
            "sequence": "PEPTIDE",
            "modifications": [],
            "expected": [{}],
        },
        # Single fixed
        {
            "sequence": "ACDE",
            "modifications": [CARBAMIDOMETHYL],
            "expected": [{1: CARBAMIDOMETHYL}],
        },
        # Double fixed
        {
            "sequence": "ACCDE",
            "modifications": [CARBAMIDOMETHYL],
            "expected": [{1: CARBAMIDOMETHYL, 2: CARBAMIDOMETHYL}],
        },
        # Single variable
        {
            "sequence": "ADME",
            "modifications": [OXIDATION],
            "expected": [{}, {2: OXIDATION}],
        },
        # Double variable
        {
            "sequence": "ADMME",
            "modifications": [OXIDATION],
            "expected": [{}, {2: OXIDATION}, {3: OXIDATION}, {2: OXIDATION, 3: OXIDATION}],
        },
        # More than maximum simultaneous mods should be ignored
        {
            "sequence": "ADMMME",
            "modifications": [OXIDATION],
            "expected": [
                {},
                {2: OXIDATION},
                {3: OXIDATION},
                {4: OXIDATION},
                {2: OXIDATION, 3: OXIDATION},
                {2: OXIDATION, 4: OXIDATION},
                {3: OXIDATION, 4: OXIDATION},
            ],
        },
        # N-term and AA-specific
        {
            "sequence": "EDEF",
            "modifications": [PYROGLU],
            "expected": [{}, {"N": PYROGLU}],
        },
        {
            "sequence": "PEPTIDE",
            "modifications": [PHOSPHO, ACETYL],
            "expected": [{}, {3: PHOSPHO}, {"N": ACETYL}, {"N": ACETYL, 3: PHOSPHO}],
        },
        {
            "sequence": "ACDEK",
            "modifications": [CARBAMIDOMETHYL, ACETYL],
            "expected": [
                {1: CARBAMIDOMETHYL},
                {1: CARBAMIDOMETHYL, "N": ACETYL},
            ],
        },
    ]

    for case in test_cases:
        peptide = search_space._PeptidoformSearchSpace(
            sequence=case["sequence"], proteins=[], is_n_term=True
        )
        modifications_by_target = search_space._restructure_modifications_by_target(
            case["modifications"]
        )
        test_out = search_space._get_peptidoform_modification_versions(
            peptide,
            case["modifications"],
            modifications_by_target,
            max_variable_modifications=2,
        )

        assert test_out == case["expected"]


def test_get_modifications_by_target():
    modifications_by_target = search_space._restructure_modifications_by_target(
        MODIFICATION_CONFIG
    )
    assert modifications_by_target["sidechain"] == {"M": MODIFICATION_CONFIG[0:1]}
    assert modifications_by_target["peptide_n_term"] == {"E": MODIFICATION_CONFIG[2:3]}
    assert modifications_by_target["peptide_c_term"] == {}
    assert modifications_by_target["protein_n_term"] == {"any": MODIFICATION_CONFIG[3:4]}
    assert modifications_by_target["protein_c_term"] == {}


class TestProteomeSearchSpace:
    def test_digest_fasta(self):
        test_input = {
            "fasta_file": "tests/test_data/test.fasta",
            "min_length": 8,
            "max_length": 30,
            "cleavage_rule": "trypsin",
            "missed_cleavages": 2,
            "semi_specific": False,
        }

        test_output = [
            search_space._PeptidoformSearchSpace(
                sequence="MYSSCSLLQR",
                proteins=["P12345"],
                modification_options=[],
                is_n_term=True,
                is_c_term=False,
            ),
            search_space._PeptidoformSearchSpace(
                sequence="MYSSCSLLQRLVWFPFLALVATQLLFIR",
                proteins=["P12345"],
                modification_options=[],
                is_n_term=True,
                is_c_term=False,
            ),
            search_space._PeptidoformSearchSpace(
                sequence="LVWFPFLALVATQLLFIR",
                proteins=["P12345"],
                modification_options=[],
                is_n_term=False,
                is_c_term=False,
            ),
            search_space._PeptidoformSearchSpace(
                sequence="NVSSLNLTNEYLHHK",
                proteins=["P12345"],
                modification_options=[],
                is_n_term=False,
                is_c_term=False,
            ),
            search_space._PeptidoformSearchSpace(
                sequence="NVSSLNLTNEYLHHKCLVSEGK",
                proteins=["P12345"],
                modification_options=[],
                is_n_term=False,
                is_c_term=False,
            ),
            search_space._PeptidoformSearchSpace(
                sequence="NVSSLNLTNEYLHHKCLVSEGKYKPGSK",
                proteins=["P12345"],
                modification_options=[],
                is_n_term=False,
                is_c_term=False,
            ),
            search_space._PeptidoformSearchSpace(
                sequence="CLVSEGKYKPGSK",
                proteins=["P12345"],
                modification_options=[],
                is_n_term=False,
                is_c_term=False,
            ),
            search_space._PeptidoformSearchSpace(
                sequence="CLVSEGKYKPGSKYEYI",
                proteins=["P12345"],
                modification_options=[],
                is_n_term=False,
                is_c_term=True,
            ),
            search_space._PeptidoformSearchSpace(
                sequence="YKPGSKYEYI",
                proteins=["P12345"],
                modification_options=[],
                is_n_term=False,
                is_c_term=True,
            ),
        ]

        sp = search_space.ProteomeSearchSpace(**test_input)
        sp._digest_fasta()
        assert test_output == sp._peptidoform_spaces
