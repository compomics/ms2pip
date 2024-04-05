from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import psm_utils.io.peptide_record
from psm_utils import PSMList

from ms2pip import exceptions

logger = logging.getLogger(__name__)


def read_psms(psms: Union[str, Path, PSMList], filetype: Union[str, None]) -> PSMList:
    """Read PSMList or PSM file."""
    # Read PSMs
    if isinstance(psms, (str, Path)):
        logger.info("Reading PSMs...")
        psm_list = psm_utils.io.read_file(psms, filetype=filetype or "infer")
    elif isinstance(psms, PSMList):
        psm_list = psms
    else:
        raise TypeError("Invalid type for psms. Should be str, Path, or PSMList.")

    # Validate runs and collections
    if not len(psm_list.collections) == 1 or not len(psm_list.runs) == 1:
        raise exceptions.InvalidInputError("PSMs should be for a single run and collection.")

    # Apply fixed modifications if any
    psm_list.apply_fixed_modifications()

    logger.debug(f"Read {len(psm_list)} PSMs.")

    return psm_list
