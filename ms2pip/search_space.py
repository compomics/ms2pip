"""
Define and build the search space for in silico spectral library generation.

This module defines the search space for in silico spectral library generation as a
:py:class:`~ProteomeSearchSpace` object. Variable and fixed modifications can be configured
as :py:class:`~ModificationConfig` objects.

The peptide search space can be built from a protein FASTA file and a set of parameters, which can
then be converted to a :py:class:`psm_utils.PSMList` object for use in :py:mod:`ms2pip`. All
parameters are listed below at :py:class:`~ProteomeSearchSpace` and can be passed as a
dictionary, a JSON file, or as a :py:class:`~ProteomeSearchSpace` object. For example:

.. code-block:: json

   {
     "fasta_file": "test.fasta",
     "min_length": 8,
     "max_length": 30,
     "cleavage_rule": "trypsin",
     "missed_cleavages": 2,
     "semi_specific": false,
     "add_decoys": true,
     "modifications": [
       {
         "label": "UNIMOD:Oxidation",
         "amino_acid": "M"
       },
       {
         "label": "UNIMOD:Carbamidomethyl",
         "amino_acid": "C",
         "fixed": true
       }
     ],
     "max_variable_modifications": 3,
     "charges": [2, 3]
   }


For an unspecific protein digestion, the cleavage rule can be set to ``unspecific``. This will
result in a cleavage rule that allows cleavage after any amino acid with an unlimited number of
allowed missed cleavages.

To disable protein digestion when the FASTA file contains peptides, set the cleavage rule to
``-``. This will treat each line in the FASTA file as a separate peptide sequence, but still
allow for modifications and charges to be added.


Examples
--------
>>> from ms2pip.search_space import ProteomeSearchSpace, ModificationConfig
>>> search_space = ProteomeSearchSpace(
...     fasta_file="tests/data/test_proteins.fasta",
...     min_length=8,
...     max_length=30,
...     cleavage_rule="trypsin",
...     missed_cleavages=2,
...     semi_specific=False,
...     modifications=[
...         ModificationConfig(label="UNIMOD:Oxidation", amino_acid="M"),
...         ModificationConfig(label="UNIMOD:Carbamidomethyl", amino_acid="C", fixed=True),
...     ],
...     charges=[2, 3],
... )
>>> psm_list = search_space.into_psm_list()

>>> from ms2pip.search_space import ProteomeSearchSpace
>>> search_space = ProteomeSearchSpace.from_any("tests/data/test_search_space.json")
>>> psm_list = search_space.into_psm_list()

"""

from __future__ import annotations

import multiprocessing
import multiprocessing.dummy
from collections import defaultdict
from functools import partial
from itertools import chain, combinations, product
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import pyteomics.fasta
from psm_utils import PSM, PSMList
from pydantic import BaseModel, field_validator, model_validator
from pyteomics.parser import icleave
from rich.progress import track

logger = getLogger(__name__)


class ModificationConfig(BaseModel):
    """Configuration for a single modification in the search space."""

    label: str
    amino_acid: Optional[str] = None
    peptide_n_term: Optional[bool] = False
    protein_n_term: Optional[bool] = False
    peptide_c_term: Optional[bool] = False
    protein_c_term: Optional[bool] = False
    fixed: Optional[bool] = False

    def __init__(self, **data: Any):
        """
        Configuration for a single modification in the search space.

        Parameters
        ----------
        label
            Label of the modification. This can be any valid ProForma 2.0 label.
        amino_acid
            Amino acid target of the modification. :py:obj:`None` if the modification is not
            specific to an amino acid. Default is None.
        peptide_n_term
            Whether the modification occurs only on the peptide N-terminus. Default is False.
        protein_n_term
            Whether the modification occurs only on the protein N-terminus. Default is False.
        peptide_c_term
            Whether the modification occurs only on the peptide C-terminus. Default is False.
        protein_c_term
            Whether the modification occurs only on the protein C-terminus. Default is False.
        fixed
            Whether the modification is fixed. Default is False.

        """
        super().__init__(**data)

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


DEFAULT_MODIFICATIONS = [
    ModificationConfig(
        label="UNIMOD:Oxidation",
        amino_acid="M",
    ),
    ModificationConfig(
        label="UNIMOD:Carbamidomethyl",
        amino_acid="C",
        fixed=True,
    ),
]


class ProteomeSearchSpace(BaseModel):
    """Search space for in silico spectral library generation."""

    fasta_file: Path
    min_length: int = 8
    max_length: int = 30
    min_precursor_mz: Optional[float] = 0
    max_precursor_mz: Optional[float] = np.inf
    cleavage_rule: str = "trypsin"
    missed_cleavages: int = 2
    semi_specific: bool = False
    add_decoys: bool = False
    modifications: List[ModificationConfig] = DEFAULT_MODIFICATIONS
    max_variable_modifications: int = 3
    charges: List[int] = [2, 3]

    def __init__(self, **data: Any):
        """
        Search space for in silico spectral library generation.

        Parameters
        ----------
        fasta_file
            Path to FASTA file with protein sequences.
        min_length
            Minimum peptide length. Default is 8.
        max_length
            Maximum peptide length. Default is 30.
        min_precursor_mz
            Minimum precursor m/z for peptides. Default is 0.
        max_precursor_mz
            Maximum precursor m/z for peptides. Default is np.inf.
        cleavage_rule
            Cleavage rule for peptide digestion. Default is "trypsin".
        missed_cleavages
            Maximum number of missed cleavages. Default is 2.
        semi_specific
            Allow semi-specific cleavage. Default is False.
        add_decoys
            Add decoy sequences to search space. Default is False.
        modifications
            List of modifications to consider. Default is oxidation of M and carbamidomethylation
            of C.
        max_variable_modifications
            Maximum number of variable modifications per peptide. Default is 3.
        charges
            List of charges to consider. Default is [2, 3].

        """

        super().__init__(**data)
        self._peptidoform_spaces: List[_PeptidoformSearchSpace] = []

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

    def __len__(self):
        if not self._peptidoform_spaces:
            raise ValueError("Search space must be built before length can be determined.")
        return sum(len(pep_space) for pep_space in self._peptidoform_spaces)

    @classmethod
    def from_any(cls, _input: Union[dict, str, Path, ProteomeSearchSpace]) -> ProteomeSearchSpace:
        """
        Create ProteomeSearchSpace from various input types.

        Parameters
        ----------
        _input
            Search space parameters as a dictionary, a path to a JSON file, an existing
            :py:class:`ProteomeSearchSpace` object.

        """
        if isinstance(_input, ProteomeSearchSpace):
            return _input
        elif isinstance(_input, (str, Path)):
            with open(_input, "rt") as f:
                return cls.model_validate_json(f.read())
        elif isinstance(_input, dict):
            return cls.model_validate(_input)
        else:
            raise ValueError("Search space must be a dict, str, Path, or ProteomeSearchSpace.")

    def build(self, processes: int = 1):
        """
        Build peptide search space from FASTA file.

        Parameters
        ----------
        processes : int
            Number of processes to use for parallelization.

        """
        processes = processes if processes else multiprocessing.cpu_count()
        self._digest_fasta(processes)
        self._remove_redundancy()
        self._add_modifications(processes)
        self._add_charges()

    def __iter__(self) -> Generator[PSM, None, None]:
        """
        Generate PSMs from search space.

        If :py:meth:`build` has not been called, the search space will first be built with the
        given parameters.

        Parameters
        ----------
        processes : int
            Number of processes to use for parallelization.

        """
        # Build search space if not already built
        if not self._peptidoform_spaces:
            raise ValueError("Search space must be built before PSMs can be generated.")

        spectrum_id = 0
        for pep_space in self._peptidoform_spaces:
            for pep in pep_space:
                yield PSM(
                    peptidoform=pep,
                    spectrum_id=spectrum_id,
                    protein_list=pep_space.proteins,
                )
                spectrum_id += 1

    def filter_psms_by_mz(self, psms: PSMList) -> PSMList:
        """Filter PSMs by precursor m/z range."""
        return PSMList(
            psm_list=[
                psm
                for psm in psms
                if self.min_precursor_mz <= psm.peptidoform.theoretical_mz <= self.max_precursor_mz
            ]
        )

    def _digest_fasta(self, processes: int = 1):
        """Digest FASTA file to peptides and populate search space."""
        # Convert to string to avoid issues with Path objects
        self.fasta_file = str(self.fasta_file)
        n_proteins = _count_fasta_entries(self.fasta_file)
        if self.add_decoys:
            fasta_db = pyteomics.fasta.decoy_db(
                self.fasta_file,
                mode="reverse",
                decoy_only=False,
                keep_nterm=True,
            )
            n_proteins *= 2
        else:
            fasta_db = pyteomics.fasta.FASTA(self.fasta_file)

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
            self._peptidoform_spaces = list(chain.from_iterable(results))

    def _remove_redundancy(self):
        """Remove redundancy in peptides and combine protein lists."""
        peptide_dict = dict()
        for peptide in track(
            self._peptidoform_spaces,
            description="Removing peptide redundancy...",
            transient=True,
        ):
            if peptide.sequence in peptide_dict:
                peptide_dict[peptide.sequence].proteins.extend(peptide.proteins)
            else:
                peptide_dict[peptide.sequence] = peptide

        # Overwrite with non-redundant peptides
        self._peptidoform_spaces = list(peptide_dict.values())

    def _add_modifications(self, processes: int = 1):
        """Add modifications to peptides in search space."""
        modifications_by_target = _restructure_modifications_by_target(self.modifications)
        modification_options = []
        with _get_pool(processes) as pool:
            partial_get_modification_versions = partial(
                _get_peptidoform_modification_versions,
                modifications=self.modifications,
                modifications_by_target=modifications_by_target,
                max_variable_modifications=self.max_variable_modifications,
            )
            modification_options = pool.imap(
                partial_get_modification_versions, self._peptidoform_spaces
            )
            for pep, mod_opt in track(
                zip(self._peptidoform_spaces, modification_options),
                description="Adding modifications...",
                total=len(self._peptidoform_spaces),
                transient=True,
            ):
                pep.modification_options = mod_opt

    def _add_charges(self):
        """Add charge permutations to peptides in search space."""
        for peptide in track(
            self._peptidoform_spaces,
            description="Adding charge permutations...",
            transient=True,
        ):
            peptide.charge_options = self.charges


class _PeptidoformSearchSpace(BaseModel):
    """Search space for a given amino acid sequence."""

    sequence: str
    proteins: List[str]
    is_n_term: Optional[bool] = None
    is_c_term: Optional[bool] = None
    modification_options: List[Dict[int, ModificationConfig]] = []
    charge_options: List[int] = []

    def __init__(self, **data: Any):
        """
        Search space for a given amino acid sequence.

        Parameters
        ----------
        sequence
            Amino acid sequence of the peptidoform.
        proteins
            List of protein IDs containing the peptidoform.
        is_n_term
            Whether the peptidoform is an N-terminal peptide. Default is None.
        is_c_term
            Whether the peptidoform is a C-terminal peptide. Default is None.
        modification_options
            List of dictionaries with modification positions and configurations. Default is [].
        charge_options
            List of charge states to consider. Default is [].

        """
        super().__init__(**data)

    def __len__(self):
        return len(self.modification_options) * len(self.charge_options)

    def __iter__(self) -> Generator[str, None, None]:
        """Yield peptidoform strings with given charges and modifications."""
        if not self.charge_options:
            raise ValueError("Peptide charge options not defined.")
        if not self.modification_options:
            raise ValueError("Peptide modification options not defined.")

        for modifications, charge in product(self.modification_options, self.charge_options):
            yield self._construct_peptidoform_string(self.sequence, modifications, charge)

    @staticmethod
    def _construct_peptidoform_string(
        sequence: str, modifications: Dict[int, ModificationConfig], charge: int
    ) -> str:
        if not modifications:
            return f"{sequence}/{charge}"

        modded_sequence = list(sequence)
        for position, mod in modifications.items():
            if isinstance(position, int):
                aa = modded_sequence[position]
                if aa != mod.amino_acid:
                    raise ValueError(
                        f"Modification {mod.label} at position {position} does not match amino "
                        f"acid {aa}."
                    )
                modded_sequence[position] = f"{aa}[{mod.label}]"
            elif position == "N":
                modded_sequence.insert(0, f"[{mod.label}]-")
            elif position == "C":
                modded_sequence.append(f"-[{mod.label}]")
            else:
                raise ValueError(f"Invalid position {position} for modification {mod.label}.")

        return f"{''.join(modded_sequence)}/{charge}"


def _digest_single_protein(
    protein: pyteomics.fasta.Protein,
    min_length: int = 8,
    max_length: int = 30,
    cleavage_rule: str = "trypsin",
    missed_cleavages: int = 2,
    semi_specific: bool = False,
) -> List[_PeptidoformSearchSpace]:
    """Digest protein sequence and return a list of validated peptides."""

    def valid_residues(sequence: str) -> bool:
        return not any(aa in sequence for aa in ["B", "J", "O", "U", "X", "Z"])

    def parse_peptide(
        start_position: int,
        sequence: str,
        protein: pyteomics.fasta.Protein,
    ) -> _PeptidoformSearchSpace:
        """Parse result from parser.icleave into Peptide."""
        return _PeptidoformSearchSpace(
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
        "sidechain": defaultdict(lambda: []),
        "peptide_n_term": defaultdict(lambda: []),
        "peptide_c_term": defaultdict(lambda: []),
        "protein_n_term": defaultdict(lambda: []),
        "protein_c_term": defaultdict(lambda: []),
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


def _get_modification_possibilities_by_site(
    peptide: _PeptidoformSearchSpace,
    modifications_by_target: Dict[str, Dict[str, List[ModificationConfig]]],
    modifications: List[ModificationConfig],
) -> Dict[Union[str, int], List[ModificationConfig]]:
    """Get all possible modifications for each site in a peptide sequence."""
    possibilities_by_site = defaultdict(list)

    # Generate dictionary of positions per amino acid
    position_dict = defaultdict(list)
    for pos, aa in enumerate(peptide.sequence):
        position_dict[aa].append(pos)
    # Map modifications to positions
    for aa in set(position_dict).intersection(set(modifications_by_target["sidechain"])):
        possibilities_by_site.update(
            {pos: modifications_by_target["sidechain"][aa] for pos in position_dict[aa]}
        )

    # Assign possible modifications per terminus
    for terminus, position, site_name, specificity in [
        ("peptide_n_term", 0, "N", None),
        ("peptide_c_term", -1, "C", None),
        ("protein_n_term", 0, "N", "is_n_term"),
        ("protein_c_term", -1, "C", "is_c_term"),
    ]:
        if specificity is None or getattr(peptide, specificity):
            for site, mods in modifications_by_target[terminus].items():
                if site == "any" or peptide.sequence[position] == site:
                    possibilities_by_site[site_name].extend(mods)

    # Override with fixed modifications
    for mod in modifications:
        aa = mod.amino_acid
        # Skip variable modifications
        if not mod.fixed:
            continue
        # Assign if specific aa matches or if no aa is specified for each terminus
        for terminus, position, site_name, specificity in [
            ("peptide_n_term", 0, "N", None),
            ("peptide_c_term", -1, "C", None),
            ("protein_n_term", 0, "N", "is_n_term"),
            ("protein_c_term", -1, "C", "is_c_term"),
        ]:
            if getattr(mod, terminus):  # Mod has this terminus
                if specificity is None or getattr(peptide, specificity):  # Specificity matches
                    if not aa or (aa and peptide.sequence[position] == aa):  # AA matches
                        possibilities_by_site[site_name] = [mod]  # Override with fixed mod
                break  # Allow `else: if amino_acid` if no terminus matches
        # Assign if fixed modification is not terminal and specific aa matches
        else:
            if aa:
                for pos in position_dict[aa]:
                    possibilities_by_site[pos] = [mod]

    return possibilities_by_site


def _get_peptidoform_modification_versions(
    peptide: _PeptidoformSearchSpace,
    modifications: List[ModificationConfig],
    modifications_by_target: Dict[str, Dict[str, List[ModificationConfig]]],
    max_variable_modifications: int = 3,
) -> List[Dict[Union[str, int], List[ModificationConfig]]]:
    """
    Get all potential combinations of modifications for a peptide sequence.

    Examples
    --------
    >>> peptide = PeptidoformSpace(sequence="PEPTIDE", proteins=["PROTEIN"])
    >>> phospho = ModificationConfig(label="Phospho", amino_acid="T", fixed=False)
    >>> acetyl = ModificationConfig(label="Acetyl", peptide_n_term=True, fixed=False)
    >>> modifications = [phospho, acetyl]
    >>> modifications_by_target = {
    ...     "sidechain": {"S": [modifications[0]]},
    ...     "peptide_n_term": {"any": [modifications[1]]},
    ...     "peptide_c_term": {"any": []},
    ...     "protein_n_term": {"any": []},
    ...     "protein_c_term": {"any": []},
    ... }
    >>> _get_modification_versions(peptide, modifications, modifications_by_target)
    [{}, {3: phospho}, {0: acetyl}, {0: acetyl, 3: phospho}]

    """
    # Get all possible modifications for each site in the peptide sequence
    possibilities_by_site = _get_modification_possibilities_by_site(
        peptide, modifications_by_target, modifications
    )

    # Separate fixed and variable modification sites
    fixed_modifications = {}
    variable_sites = []
    for site, mods in possibilities_by_site.items():
        for mod in mods:
            if mod.fixed:
                fixed_modifications[site] = mod
            else:
                variable_sites.append((site, mod))

    # Generate all combinations of variable modifications up to the maximum allowed
    modification_versions = []
    for i in range(max_variable_modifications + 1):
        for comb in combinations(variable_sites, i):
            combination_dict = fixed_modifications.copy()
            for site, mod in comb:
                combination_dict[site] = mod
            modification_versions.append(combination_dict)

    return modification_versions


def _get_pool(processes: int) -> Union[multiprocessing.Pool, multiprocessing.dummy.Pool]:
    """Get a multiprocessing pool with the given number of processes."""
    # TODO: fix None default value for processes
    if processes > 1:
        return multiprocessing.Pool(processes)
    else:
        return multiprocessing.dummy.Pool(processes)
