"""Peptide and modification handling for MS2PIP."""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Generator

import numpy as np
from psm_utils import Peptidoform, PSMList
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
    """Modification-aware encoding of peptidoforms."""

    def __init__(self) -> None:
        """
        Modification-aware encoding of peptidoforms.

        MS²PIP requires all modification mass shifts to be written to a file for use in C code
        before running. This class handles the encoding of peptides and peptidoforms for
        modifications that have been defined.

        Encoder files are to be passed on to the ``ms2pip_pyx.ms2pip_init`` function. E.g.,
        ``ms2pip_pyx.ms2pip_init(*encoder.encoder_files)``.

        Notes
        -----
        - Either used as a context manager or manually call :py:meth:`write_encoder_files` before
          use and :py:meth:`remove_encoder_files` after use.
        - Fixed, labile, and unlocalized modifications are ignored. Fixed modifications
          should therefore already have been applied (see
          :py:meth:`psm_utils.psm_list.PSMList.apply_fixed_modifications`).

        """
        self.modifications = {}
        self.encoder_files = None

        self._next_mod_id = 38  # Avoid clash with amino acids and mutations (ionbot compatibility)

    def __enter__(self):
        if not self.encoder_files:
            self.write_encoder_files()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_encoder_files()

    def __repr__(self) -> str:
        return "{}.{}(modifications={})".format(
            self.__class__.__module__,
            self.__class__.__qualname__,
            self.modifications,
        )

    @classmethod
    def from_peptidoform(cls, peptidoform: Peptidoform) -> Encoder:
        """
        Create Encoder instance from peptidoform.

        Parameters
        ----------
        peptidoform : Peptidoform
            Peptidoform to use for modification configuration.

        Returns
        -------
        Encoder
            Encoder instance with modifications configured.

        """
        encoder = cls()
        encoder._configure_from_peptidoform(peptidoform)
        encoder.write_encoder_files()
        return encoder

    @classmethod
    def from_psm_list(cls, psm_list: PSMList) -> Encoder:
        """
        Create Encoder instance from PSMList.

        Parameters
        ----------
        psm_list : PSMList
            PSMList to use for modification configuration.

        Returns
        -------
        Encoder
            Encoder instance with modifications configured.

        """
        encoder = cls()
        encoder._configure_from_psm_list(psm_list)
        encoder.write_encoder_files()
        return encoder

    def _configure_modification(self, target: str, modification: proforma.TagBase):
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
            return None

        self.modifications[(target, str(modification))] = {
            "mod_id": self._next_mod_id,
            "mass_shift": modification.mass,
            "amino_acid": target,
            "amino_acid_id": amino_acid_id,
            "modification": modification,
        }
        self._next_mod_id += 1

    def _configure_from_peptidoform(self, peptidoform: Peptidoform):
        """Configure encoder with modifications from single Peptidoform."""
        # Get unique modifications from psm
        unique_modifications = dict()
        try:
            for aa, mods in peptidoform.parsed_sequence:
                if mods:
                    unique_modifications.update({(aa, str(mod)): mod for mod in mods})
            for term in ["n_term", "c_term"]:
                if peptidoform.properties[term]:
                    unique_modifications.update(
                        {(term, str(mod)): mod for mod in peptidoform.properties[term]}
                    )
        except KeyError as e:
            raise exceptions.UnresolvableModificationError(e.args[0]) from e

        # Add modification entries
        for (target, _), mod in unique_modifications.items():
            self._configure_modification(target, mod)

    def _configure_from_psm_list(self, psm_list: PSMList):
        """Configure encoder with modifications from PSMList."""
        # Get unique modifications from psm_list
        unique_modifications = dict()
        try:
            for psm in psm_list:
                for aa, mods in psm.peptidoform.parsed_sequence:
                    if mods:
                        unique_modifications.update({(aa, str(mod)): mod for mod in mods})
                for term in ["n_term", "c_term"]:
                    if psm.peptidoform.properties[term]:
                        unique_modifications.update(
                            {(term, str(mod)): mod for mod in psm.peptidoform.properties[term]}
                        )
        except KeyError as e:
            raise exceptions.UnresolvableModificationError(e.args[0]) from e

        # Add modification entries
        for (target, _), mod in unique_modifications.items():
            self._configure_modification(target, mod)

    def write_encoder_files(self) -> str:
        """Write configured masses to temporary files for use in C code."""
        # AA file
        amino_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="\n")
        for m in AMINO_ACID_MASSES:
            amino_file.write("{}\n".format(m))
        amino_file.write("0\n")
        amino_file.close()

        # PTM file
        mod_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="\n")
        mod_file.write("{}\n".format(len(self.modifications)))
        for mod in self.modifications.values():
            mod_file.write(
                "{},1,{},{}\n".format(mod["mass_shift"], mod["amino_acid_id"], mod["mod_id"])
            )
        mod_file.close()

        # SPTM file (ionbot compatibility)
        mod_file2 = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="\n")
        mod_file2.write("0\n")
        mod_file2.close()

        # Store temporary file names
        self.encoder_files = (amino_file.name, mod_file.name, mod_file2.name)

    def remove_encoder_files(self):
        """Remove temporary encoder files."""
        if self.encoder_files:
            for f in self.encoder_files:
                os.remove(f)
            self.encoder_files = None

    @staticmethod
    def validate_peptidoform(peptidoform: Peptidoform):
        """Validate whether a peptidoform can be encoded for MS²PIP."""
        # Charge required
        if peptidoform.precursor_charge is None:
            raise exceptions.InvalidPeptidoformError("Peptidoform charge is required.")

        # Peptides longer then 101 lead to "Segmentation fault (core dumped)"
        if len(peptidoform.parsed_sequence) > 100:
            raise exceptions.InvalidPeptidoformError(
                "Peptidoform sequence cannot be longer than 100 amino acids."
            )
        elif len(peptidoform.parsed_sequence) < 4:
            raise exceptions.InvalidPeptidoformError(
                "Peptidoform sequence cannot be shorter than 4 amino acids."
            )

    def encode_peptide(self, peptidoform: Peptidoform) -> np.ndarray:
        """Encode a peptide (without modifications) for MS²PIP."""
        self.validate_peptidoform(peptidoform)

        try:
            encoded = [0] + [AMINO_ACID_IDS[aa] for aa, _ in peptidoform.parsed_sequence] + [0]
        except KeyError as e:
            raise exceptions.InvalidAminoAcidError(
                f"Unsupported amino acid found in peptide `{peptidoform.proforma}`"
            ) from e
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

        def _generate_encoding(peptidoform) -> Generator[int, None, None]:
            if peptidoform.properties["n_term"]:
                mod_str = str(peptidoform.properties["n_term"][0])
                yield self.modifications["n_term", mod_str]["mod_id"]
            else:
                yield 0

            for aa, mods in peptidoform.parsed_sequence:
                try:
                    if not mods:
                        yield AMINO_ACID_IDS[aa]
                    else:
                        yield self.modifications[aa, str(mods[0])]["mod_id"]
                except KeyError as e:
                    raise exceptions.InvalidAminoAcidError(
                        f"Unsupported amino acid found in peptide `{peptidoform.proforma}`"
                    ) from e

            if peptidoform.properties["c_term"]:
                mod_str = str(peptidoform.properties["c_term"][0])
                yield self.modifications["c_term", mod_str]["mod_id"]
            else:
                yield 0

        self.validate_peptidoform(peptidoform)
        return np.array(list(_generate_encoding(peptidoform)), dtype=np.uint16)
