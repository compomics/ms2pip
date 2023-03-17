from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import psm_utils.io.peptide_record
from psm_utils import PSMList

from ms2pip import exceptions

logger = logging.getLogger(__name__)


def read_psms(psm_list: Union[str, Path, PSMList]) -> Tuple[pd.DataFrame, dict]:
    """Read PSMList or PSM file into old-style PEPREC DataFrame and modification configuration."""
    # Read PSMs
    if isinstance(psm_list, (str, Path)):
        psm_list = psm_utils.io.read_file(psm_list)
    elif isinstance(psm_list, PSMList):
        pass
    else:
        raise TypeError("Invalid type for psm_list. Should be str, Path, or PSMList.")

    # Convert to old-style PEPREC DataFrame
    psm_list.apply_fixed_modifications()
    peprec = psm_utils.io.peptide_record.to_dataframe(psm_list)

    # Add psm_id column
    if not "psm_id" in peprec.columns:
        logger.debug("Adding psm_id column to peptide file")
        peprec.reset_index(inplace=True)
        peprec["psm_id"] = peprec["index"].astype(str)
        peprec.rename({"index": "psm_id"}, axis=1, inplace=True)

    # Set modification config
    modification_config = _get_modification_config(psm_list)

    # Validate PeptideRecord
    if len(peprec) == 0:
        raise exceptions.NoValidPeptideSequencesError()
    peprec = peprec.fillna("-")
    if not "ce" in peprec.columns:
        peprec["ce"] = 30
    else:
        peprec["ce"] = peprec["ce"].astype(int)

    peprec["charge"] = peprec["charge"].astype(int)

    # Filter for unsupported peptides
    num_pep = len(peprec)
    peprec = peprec[
        ~(peprec["peptide"].str.contains("B|J|O|U|X|Z"))
        & ~(peprec["peptide"].str.len() < 3)
        & ~(peprec["peptide"].str.len() > 99)
    ].copy()
    num_pep_filtered = num_pep - len(peprec)
    if num_pep_filtered > 0:
        logger.warning(
            f"Removed {num_pep_filtered} unsupported peptide sequences (< 3, > 99 amino "
            f"acids, or containing B, J, O, U, X or Z). Retained {len(peprec)} entries."
        )

    if len(peprec) == 0:
        raise exceptions

    return peprec, modification_config


def _get_modification_config(psm_list: PSMList):
    """
    Get MS²PIP-style modification configuration from PSMList.

    Notes
    -----
    Fixed, labile, and unlocalized modifications are ignored. Fixed modifications
    should therefore already have been applied (see
    :py:meth:`psm_utils.PSMList.apply_fixed_modifications`).
    """
    unique_modifications = set()
    for psm in psm_list:
        for aa, mods in psm.peptidoform.parsed_sequence:
            if mods:
                unique_modifications.update([(aa, mod) for mod in mods])
        if psm.peptidoform.properties["n_term"]:
            unique_modifications.update(
                [("N-term", mod) for mod in psm.peptidoform.properties["n_term"]]
            )
        if psm.peptidoform.properties["c_term"]:
            unique_modifications.update(
                [("C-term", mod) for mod in psm.peptidoform.properties["c_term"]]
            )
    return [
        ",".join([mod.value, str(mod.mass), "opt", target]) for target, mod in unique_modifications
    ]
