"""Search space for in silico spectral library generation."""

from __future__ import annotations

import multiprocessing
import multiprocessing.dummy
from collections import defaultdict
from functools import cmp_to_key, partial
from itertools import chain, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pyteomics.fasta
from psm_utils import PSMList
from pydantic import BaseModel, field_validator, model_validator
from pyteomics.parser import icleave
from rich.progress import track


class ModificationConfig(BaseModel):
    """Configuration for a single modification in the search space."""

    label: str
    amino_acid: Optional[str] = None
    peptide_n_term: Optional[bool] = False
    protein_n_term: Optional[bool] = False
    peptide_c_term: Optional[bool] = False
    protein_c_term: Optional[bool] = False
    fixed: Optional[bool] = False

    @model_validator(mode="after")
    def _modification_must_have_target(self):
        target_fields = [
            "amino_acid",
            "peptide_n_term",
            "protein_n_term",
            "peptide_c_term",
            "protein_c_term",
        ]
        if not any(getattr(self, t) for t in target_fields):
            raise ValueError("Modifications must have a target (amino acid or N/C-term).")
        return self


class PeptidoformSearchSpace(BaseModel):
    """Peptidoform search space for a given amino acid sequence."""

    sequence: str
    proteins: List[str]
    is_n_term: Optional[bool] = None
    is_c_term: Optional[bool] = None
    # TODO Can be changed to whichever is convenient
    modification_options: List[Dict[int, ModificationConfig]] = None
    charge_options: List[int] = None

    # TODO
    def into_psm_list(self) -> PSMList:
        """Convert PeptidoformSpace to PSMList with given charge and modification."""
        if not self.charge_options:
            raise ValueError("Peptide charge options not defined.")
        if not self.modification_options:
            raise ValueError("Peptide modification options not defined.")

        raise NotImplementedError("Method not implemented yet.")


DEFAULT_MODIFICATIONS = [
    ModificationConfig(
        name="Oxidation",
        unimod_accession=35,
        mass_shift=15.994915,
        amino_acid="M",
    ),
    ModificationConfig(
        name="Carbamidomethyl",
        mass_shift=57.021464,
        unimod_accession=4,
        amino_acid="C",
        fixed=True,
    ),
]


class ProteomeSearchSpace(BaseModel):
    """Search space for in silico spectral library generation."""

    fasta_file: Path
    min_length: int = 8
    max_length: int = 30
    min_precursor_mz: Optional[float] = None
    max_precursor_mz: Optional[float] = None
    cleavage_rule: str = "trypsin"
    missed_cleavages: int = 2
    semi_specific: bool = False
    add_decoys: bool = False
    modifications: List[ModificationConfig] = DEFAULT_MODIFICATIONS
    max_variable_modifications: int = 3
    charges: List[int] = [2, 3]

    @field_validator("modifications")
    @classmethod
    def _validate_modifications(cls, v):
        if all(isinstance(m, ModificationConfig) for m in v):
            return v
        elif all(isinstance(m, dict) for m in v):
            return [ModificationConfig(**modification) for modification in v]
        else:
            raise ValueError(
                "Modifications should be a list of dicts or ModificationConfig objects."
            )

    @model_validator(mode="after")
    def _validate_unspecific_cleavage(self):
        """Validate and configure unspecific cleavage settings."""
        # `unspecific` is not an option in pyteomics.parser.icleave, so we configure
        # the settings for unspecific cleavage manually.
        if self.cleavage_rule == "unspecific":
            self.missed_cleavages = self.max_length
            self.cleavage_rule = r"(?<=[A-Z])"
        return self

    @classmethod
    def from_any(cls, _input: Union[dict, str, Path, ProteomeSearchSpace]) -> ProteomeSearchSpace:
        """Create ProteomeSearchSpace from various input types."""
        if isinstance(_input, ProteomeSearchSpace):
            return _input
        elif isinstance(_input, (str, Path)):
            with open(_input, "rt") as f:
                return cls.model_validate_json(f.read())
        elif isinstance(_input, dict):
            return cls.model_validate(_input)
        else:
            raise ValueError("Search space must be a dict, str, Path, or ProteomeSearchSpace.")

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._peptidoform_space: List[PeptidoformSearchSpace] = []

    def generate_psms(self, processes=1) -> PSMList:
        """Generate PSMs from search space."""
        if not self._peptidoform_space:
            self.build_search_space(processes)

        # TODO Implement filtering by precursor m/z on PSMs
        return chain.from_iterable([psm.into_psm_list() for psm in self._peptidoform_space])

    def build_search_space(self, processes=1):
        """Build peptide search space from FASTA file."""
        self.digest_fasta(processes)
        self.remove_redundancy()
        self.add_modifications(processes)
        self.add_charges()

    def digest_fasta(self, processes=1):
        """Digest FASTA file to peptides and populate search space."""
        n_proteins = _count_fasta_entries(self.fasta_file)
        if self.add_decoys:
            fasta_db = pyteomics.fasta.decoy_db(
                self.fasta_file,
                mode="reverse",
                decoy_only=False,
                keep_nterm=True,
            )
        else:
            fasta_db = pyteomics.fasta.FASTA(self.fasta_file)
            n_proteins *= 2

        # Read proteins and digest to peptides
        with _get_pool(processes) as pool:
            partial_digest_protein = partial(
                _digest_single_protein,
                min_length=self.min_length,
                max_length=self.max_length,
                cleavage_rule=self.cleavage_rule,
                missed_cleavages=self.missed_cleavages,
                semi_specific=self.semi_specific,
            )
            results = track(
                pool.imap(partial_digest_protein, fasta_db),
                total=n_proteins,
                description="Digesting proteins...",
                transient=True,
            )
            self._peptidoform_space = list(chain.from_iterable(results))

    def remove_redundancy(self):
        """Remove redundancy in peptides and combine protein lists."""
        peptide_dict = dict()
        for peptide in track(
            self._peptidoform_space,
            description="Removing peptide redundancy...",
            transient=True,
        ):
            if peptide.sequence in peptide_dict:
                peptide_dict[peptide.sequence].proteins.extend(peptide.proteins)
            else:
                peptide_dict[peptide.sequence] = peptide

        # Overwrite with non-redundant peptides
        self._peptidoform_space = list(peptide_dict.values())

    def add_modifications(self, processes=1):
        """Add modifications to peptides in search space."""
        modifications_by_target = _restructure_modifications_by_target(self.modifications)
        modification_options = []
        with _get_pool(processes) as pool:
            partial_get_modification_versions = partial(
                _get_modification_versions,
                modifications=self.modifications,
                modifications_by_target=modifications_by_target,
                max_variable_modifications=self.max_variable_modifications,
            )
            modification_options = pool.imap(
                partial_get_modification_versions, self._peptidoform_space
            )
            for pep, mod_opt in track(
                zip(self._peptidoform_space, modification_options),
                description="Adding modifications...",
                total=len(self._peptidoform_space),
                transient=True,
            ):
                pep.modification_options = mod_opt

    def add_charges(self):
        """Add charge permutations to peptides in search space."""
        for peptide in track(
            self._peptidoform_space,
            description="Adding charge permutations...",
            transient=True,
        ):
            peptide.charge_options = self.charges

    # TODO
    def filter_precursor_mz(self, processes=1):
        """Filter peptides based on precursor m/z."""
        raise NotImplementedError()


def _digest_single_protein(
    protein: pyteomics.fasta.Protein,
    min_length: int = 8,
    max_length: int = 30,
    cleavage_rule: str = "trypsin",
    missed_cleavages: int = 2,
    semi_specific: bool = False,
) -> List[PeptidoformSearchSpace]:
    """Digest protein sequence and return a list of validated peptides."""

    def valid_residues(sequence: str) -> bool:
        return not any(aa in sequence for aa in ["B", "J", "O", "U", "X", "Z"])

    def parse_peptide(
        start_position: int,
        sequence: str,
        protein: pyteomics.fasta.Protein,
    ) -> PeptidoformSearchSpace:
        """Parse result from parser.icleave into Peptide."""
        return PeptidoformSearchSpace(
            sequence=sequence,
            # Assumes protein ID is description until first space
            proteins=[protein.description.split(" ")[0]],
            is_n_term=start_position == 0,
            is_c_term=start_position + len(sequence) == len(protein.sequence),
        )

    peptides = [
        parse_peptide(start, seq, protein)
        for start, seq in icleave(
            protein.sequence,
            cleavage_rule,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            semi=semi_specific,
        )
        if valid_residues(seq)
    ]

    return peptides


def _count_fasta_entries(filename: Path) -> int:
    """Count the number of entries in a FASTA file."""
    with open(filename, "rt") as f:
        count = 0
        for line in f:
            if line[0] == ">":
                count += 1
    return count


def _restructure_modifications_by_target(
    modifications: List[ModificationConfig],
) -> Dict[str, Dict[str, List[ModificationConfig]]]:
    """Restructure variable modifications to options per side chain or terminus."""
    modifications_by_target = {
        "sidechain": defaultdict(lambda: [None]),
        "peptide_n_term": defaultdict(lambda: [None]),
        "peptide_c_term": defaultdict(lambda: [None]),
        "protein_n_term": defaultdict(lambda: [None]),
        "protein_c_term": defaultdict(lambda: [None]),
    }

    def add_mod(mod, target, amino_acid):
        if amino_acid:
            modifications_by_target[target][amino_acid].append(mod)
        else:
            modifications_by_target[target]["any"].append(mod)

    for mod in modifications:
        if mod.fixed:
            continue
        if mod.peptide_n_term:
            add_mod(mod, "peptide_n_term", mod.amino_acid)
        elif mod.peptide_c_term:
            add_mod(mod, "peptide_c_term", mod.amino_acid)
        elif mod.protein_n_term:
            add_mod(mod, "protein_n_term", mod.amino_acid)
        elif mod.protein_c_term:
            add_mod(mod, "protein_c_term", mod.amino_acid)
        else:
            add_mod(mod, "sidechain", mod.amino_acid)

    return {k: dict(v) for k, v in modifications_by_target.items()}


# TODO: Refactor for v4.0.0
def _get_modification_versions(
    peptide: PeptidoformSearchSpace,
    modifications: List[ModificationConfig],
    modifications_by_target: Dict[str, Dict[str, List[ModificationConfig]]],
    max_variable_modifications: int = 3,
) -> Dict[Union[str, int], List[str]]:
    """
    Get all potential combinations of modifications for a peptide sequence.

    Examples
    --------
    >>> peptide = PeptidoformSpace(sequence="PEPTIDE", proteins=["PROTEIN"])
    >>> modifications = [
    ...     ModificationConfig(label="Phospho", amino_acid="T", fixed=False),
    ...     ModificationConfig(label="Acetyl", peptide_n_term=True, fixed=False),
    ... ]
    >>> modifications_by_target = {
    ...     "sidechain": {"S": [modifications[0]]},
    ...     "peptide_n_term": {"any": [modifications[1]]},
    ...     "peptide_c_term": {"any": []},
    ...     "protein_n_term": {"any": []},
    ...     "protein_c_term": {"any": []},
    ... }
    >>> _get_modification_versions(peptide, modifications, modifications_by_target)
    [{}, {3: 'Phospho'}, {0: 'Acetyl'}, {0: 'Acetyl', 3: 'Phospho'}]

    """
    possibilities_by_site = defaultdict(list)

    # Generate dictionary of positions per amino acid
    pos_dict = defaultdict(list)
    for pos, aa in enumerate(peptide.sequence):
        pos_dict[aa].append(pos + 1)
    # Map modifications to positions
    for aa in set(pos_dict).intersection(set(modifications_by_target["sidechain"])):
        possibilities_by_site.update(
            {pos: modifications_by_target["sidechain"][aa] for pos in pos_dict[aa]}
        )

    # Assign possible modifications per terminus
    for terminus, position, specificity in [
        ("peptide_n_term", 0, None),
        ("peptide_c_term", -1, None),
        ("protein_n_term", 0, "is_n_term"),
        ("protein_c_term", -1, "is_c_term"),
    ]:
        if specificity is None or getattr(peptide, specificity):
            for site, mods in modifications_by_target[terminus].items():
                if site == "any" or peptide.sequence[position] == site:
                    possibilities_by_site[position].extend(mods)

    # Override with fixed modifications
    for mod in modifications:
        aa = mod.amino_acid
        # Skip variable modifications
        if not mod.fixed:
            continue
        # Assign if specific aa matches or if no aa is specified for each terminus
        for terminus, position, specificity in [
            ("peptide_n_term", 0, None),
            ("peptide_c_term", -1, None),
            ("protein_n_term", 0, "is_n_term"),
            ("protein_c_term", -1, "is_c_term"),
        ]:
            if getattr(mod, terminus):  # Mod has this terminus
                if specificity is None or getattr(peptide, specificity):  # Specificity matches
                    if not aa or (aa and peptide.sequence[position] == aa):  # Aa matches
                        possibilities_by_site[position] = [mod]  # Override with fixed mod
                break  # Allow `else: if amino_acid` if no terminus matches
        # Assign if fixed modification is not terminal and specific aa matches
        else:
            if aa:
                for pos in pos_dict[aa]:
                    possibilities_by_site[pos] = [mod]

    # Get all possible combinations of modifications for all sites
    mod_permutations = product(*possibilities_by_site.values())
    mod_positions = possibilities_by_site.keys()

    # Filter by max modified sites (avoiding combinatorial explosion)
    mod_permutations = filter(
        lambda mods: sum([1 for m in mods if m is not None and not m.fixed])
        <= max_variable_modifications,
        mod_permutations,
    )

    def _compare_minus_one_larger(a, b):
        """Custom comparison function where `-1` is always larger."""
        if a[0] == -1:
            return 1
        elif b[0] == -1:
            return -1
        else:
            return a[0] - b[0]

    # Get MS²PIP modifications strings for each combination
    mod_strings = []
    for p in mod_permutations:
        if p == [""]:
            mod_strings.append("-")
        else:
            mods = sorted(zip(mod_positions, p), key=cmp_to_key(_compare_minus_one_larger))
            mod_strings.append("|".join(f"{p}|{m.name}" for p, m in mods if m))

    return mod_strings


def _get_pool(processes: int) -> Union[multiprocessing.Pool, multiprocessing.dummy.Pool]:
    """Get a multiprocessing pool with the given number of processes."""
    if processes > 1:
        return multiprocessing.Pool(processes)
    else:
        return multiprocessing.dummy.Pool(processes)
