from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import psm_utils.io.peptide_record
from psm_utils import PSMList

from ms2pip import exceptions
from ms2pip._utils.encoder import Encoder

logger = logging.getLogger(__name__)


def read_psms(psms: Union[str, Path, PSMList]) -> PSMList:
    """Read PSMList or PSM file into old-style PEPREC DataFrame and modification configuration."""
    # Read PSMs
    if isinstance(psms, (str, Path)):
        psm_list = psm_utils.io.read_file(psms)
    elif isinstance(psms, PSMList):
        pass
    else:
        raise TypeError("Invalid type for psms. Should be str, Path, or PSMList.")

    # Convert to old-style PEPREC DataFrame
    psm_list.apply_fixed_modifications()

    # Set encoder for modifications in PSMList
    encoder = Encoder()
    encoder.configure_modifications_from_psm_list(psm_list)

    # # Add psm_id column
    # if not "psm_id" in peprec.columns:
    #     logger.debug("Adding psm_id column to peptide file")
    #     peprec.reset_index(inplace=True)
    #     peprec["psm_id"] = peprec["index"].astype(str)
    #     peprec.rename({"index": "psm_id"}, axis=1, inplace=True)

    # # Set modification config
    # modification_config = _get_modification_config(psm_list)

    # # Validate PeptideRecord
    # if len(peprec) == 0:
    #     raise exceptions.NoValidPeptideSequencesError()
    # peprec = peprec.fillna("-")
    # if not "ce" in peprec.columns:
    #     peprec["ce"] = 30
    # else:
    #     peprec["ce"] = peprec["ce"].astype(int)

    # peprec["charge"] = peprec["charge"].astype(int)

    # # Filter for unsupported peptides
    # num_pep = len(peprec)
    # peprec = peprec[
    #     ~(peprec["peptide"].str.contains("B|J|O|U|X|Z"))
    #     & ~(peprec["peptide"].str.len() < 3)
    #     & ~(peprec["peptide"].str.len() > 99)
    # ].copy()
    # num_pep_filtered = num_pep - len(peprec)
    # if num_pep_filtered > 0:
    #     logger.warning(
    #         f"Removed {num_pep_filtered} unsupported peptide sequences (< 3, > 99 amino "
    #         f"acids, or containing B, J, O, U, X or Z). Retained {len(peprec)} entries."
    #     )

    # if len(peprec) == 0:
    #     raise exceptions.InvalidPEPRECError()

    return psm_list, encoder
