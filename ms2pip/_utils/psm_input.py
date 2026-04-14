import logging
from pathlib import Path

import psm_utils.io.peptide_record
from psm_utils import PSM, PSMList

logger = logging.getLogger(__name__)


def read_psms(psms: str | Path | PSMList | list[PSM], filetype: str | None) -> PSMList:
    """Read PSMList or PSM file."""
    # Read PSMs
    if isinstance(psms, (str, Path)):
        logger.info("Reading PSMs...")
        psm_list = psm_utils.io.read_file(psms, filetype=filetype or "infer")
    elif isinstance(psms, list):
        psm_list = PSMList(psm_list=psms)
    elif isinstance(psms, PSMList):
        psm_list = psms
    else:
        raise TypeError("Invalid type for psms. Should be str, Path, PSMList, or list[PSM].")

    # Apply fixed modifications if any
    psm_list.apply_fixed_modifications()

    logger.debug(f"Read {len(psm_list)} PSMs.")

    return psm_list
