"""Peptide and modification handling for MS2PIP."""
from __future__ import annotations

import os
import tempfile
import logging

import numpy as np
from psm_utils import PSM, Peptidoform, PSMList
from pyteomics import proforma

import ms2pip.exceptions as exceptions

logger = logging.getLogger(__name__)

AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

AMINO_ACID_MASSES = [
    71.037114,
    103.00919,
    115.026943,
    129.042593,
    147.068414,
    57.021464,
    137.058912,
    113.084064,
    128.094963,
    131.040485,
    114.042927,
    97.052764,
    128.058578,
    156.101111,
    87.032028,
    101.047679,
    99.068414,
    186.079313,
    163.063329,
]

AMINO_ACID_IDS = {a: i for i, a in enumerate(AMINO_ACIDS)}
AMINO_ACID_IDS["L"] = AMINO_ACID_IDS["I"]


class Encoder:
    def __init__(self) -> None:
        """
        Modification-aware encoding of peptidoforms.

        MS²PIP requires all modification mass shifts to be written to a file for use in C code
        before running. This class handles the encoding of peptides and peptidoforms for
        modifications that have been defined.

        """
        self.modifications = {}

        self.afile = None
        self.mod_file = None
        self.mod_file2 = None

        self._next_mod_id = 38  # Avoid clash with amino acids and mutations (ionbot compatibility)

    def __repr__(self) -> str:
        return "{}.{}({})".format(
            self.__class__.__module__,
            self.__class__.__qualname__,
            self.modifications,
        )

    def configure_modification(self, target: str, modification: proforma.TagBase):
        """
        Add single pyteomics.proforma modification to configuration.

        Parameters
        ----------
        target : str
            Target amino acid one-letter code or terminus (``n_term`` or ``c_term``).
        modification : pyteomics.proforma.TagBase
            Modification to add.

        """
        if target == "n_term":
            amino_acid_id = -1
        elif target == "c_term":
            amino_acid_id = -2
        elif target in AMINO_ACID_IDS:
            amino_acid_id = AMINO_ACID_IDS[target]
        else:
            logger.warning(f"Skipping modification for invalid amino acid: {target}")

        self.modifications[(target, modification.key)] = {
            "mod_id": self._next_mod_id,
            "mass_shift": modification.mass,
            "amino_acid": target,
            "amino_acid_id": amino_acid_id,
            "modification": modification,
        }
        self._next_mod_id += 1

    def configure_modifications_from_psm_list(self, psm_list: PSMList):
        """
        Get MS²PIP-style modification configuration from PSMList.

        Notes
        -----
        Fixed, labile, and unlocalized modifications are ignored. Fixed modifications
        should therefore already have been applied (see
        :py:meth:`psm_utils.PSMList.apply_fixed_modifications`).
        """
        # Get unique modifications from psm_list
        unique_modifications = set()
        for psm in psm_list:
            for aa, mods in psm.peptidoform.parsed_sequence:
                if mods:
                    unique_modifications.update([(aa, mod) for mod in mods])
            for term in ["n_term", "c_term"]:
                if psm.peptidoform.properties[term]:
                    unique_modifications.update(
                        [(term, mod) for mod in psm.peptidoform.properties[term]]
                    )

        # Add modification entries
        for target, mod in unique_modifications:
            self.configure_modification(target, mod)

    def write_encoding_configuration(self) -> str:
        """Write configured masses to temporary files for use in C code."""
        # AA file
        amino_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="\n")
        for m in AMINO_ACID_MASSES:
            amino_file.write("{}\n".format(m))
        amino_file.write("0\n")
        amino_file.close()
        self.afile = amino_file.name

        # PTM file
        mod_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="\n")
        mod_file.write("{}\n".format(len(self.modifications)))
        for (target, mod_key), mod in self.modifications.items():
            mod_file.write(
                "{},1,{},{}\n".format(mod["mass_shift"], mod["amino_acid_id"], mod["mod_id"])
            )
        mod_file.close()
        self.mod_file = mod_file.name

        # SPTM file (ionbot compatibility)
        mod_file2 = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="\n")
        mod_file2.write("0\n")
        mod_file2.close()
        self.mod_file2 = mod_file2.name

    def cleanup(self):
        """Remove temporary files."""
        if self.afile:
            os.remove(self.afile)
        if self.mod_file:
            os.remove(self.mod_file)
        if self.mod_file2:
            os.remove(self.mod_file2)

    def encode_peptide(self, peptidoform: Peptidoform) -> np.ndarray:
        """
        Encode a peptide (without modifications) for MS²PIP.
        """
        # Peptides longer then 101 lead to "Segmentation fault (core dumped)"
        if len(peptidoform.parsed_sequence) > 100:
            raise exceptions.InvalidPeptideError(
                "Peptide sequence cannot be longer than 100 amino acids."
            )
        elif len(peptidoform.parsed_sequence) < 4:
            raise exceptions.InvalidPeptideError(
                "Peptide sequence cannot be shorter than 4 amino acids."
            )

        try:
            encoded = [0] + [AMINO_ACID_IDS[aa] for aa, _ in peptidoform.parsed_sequence] + [0]
        except KeyError:
            raise exceptions.InvalidAminoAcidError(
                f"Unsupported amino acid found in peptide `{peptidoform.proforma}`"
            )
        return np.array(encoded, dtype=np.uint16)

    def encode_peptidoform(self, peptidoform: Peptidoform) -> np.ndarray:
        """
        Encode a peptidoform for MS²PIP.

        Notes
        -----
        - Multiple modifications per site is not supported. The first modification is used.
        - Fixed, labile, and unlocalized modifications are ignored. Fixed modifications
        should therefore already have been applied (see
        :py:meth:`psm_utils.PSMList.apply_fixed_modifications`).
        """

        def _generate_encoding(peptidoform):
            if peptidoform.properties["n_term"]:
                mod_key = peptidoform.properties["n_term"][0].key
                yield self.modifications["n_term", mod_key]["mod_id"]
            else:
                yield 0

            for aa, mods in peptidoform.parsed_sequence:
                if not mods:
                    yield AMINO_ACID_IDS[aa]
                else:
                    yield self.modifications[aa, mods[0].key]["mod_id"]

            if peptidoform.properties["c_term"]:
                mod_key = peptidoform.properties["c_term"][0].key
                yield self.modifications["c_term", mod_key]["mod_id"]
            else:
                yield 0

        return np.array(list(_generate_encoding(peptidoform)), dtype=np.uint16)
