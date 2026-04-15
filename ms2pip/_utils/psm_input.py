import logging
from pathlib import Path

import psm_utils.io.peptide_record
from psm_utils import PSM, PSMList
from psm_utils import Peptidoform

logger = logging.getLogger(__name__)

_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
_MIN_PEPTIDE_LENGTH = 4
_MAX_PEPTIDE_LENGTH = 100


def validate_peptidoform(peptidoform: Peptidoform) -> str | None:
    """Return an error reason if the peptidoform is invalid, or None if valid."""
    try:
        parsed = peptidoform.parsed_sequence
    except Exception as e:
        return str(e)
    if len(parsed) < _MIN_PEPTIDE_LENGTH:
        return f"too short ({len(parsed)} residues, minimum {_MIN_PEPTIDE_LENGTH})"
    if len(parsed) > _MAX_PEPTIDE_LENGTH:
        return f"too long ({len(parsed)} residues, maximum {_MAX_PEPTIDE_LENGTH})"
    for aa, _ in parsed:
        if aa not in _VALID_AMINO_ACIDS:
            return f"unsupported amino acid '{aa}'"
    if not peptidoform.precursor_charge:
        return "missing charge state"
    return None


def filter_valid_psms(
    psm_list: PSMList,
) -> tuple[list[int], dict[str, int]]:
    """
    Validate all PSMs and return valid indices and skip reasons.

    Returns
    -------
    valid_indices
        Indices of PSMs that passed validation.
    skip_reasons
        Dict mapping reason string to count of skipped PSMs. Empty if all valid.

    """
    valid_indices: list[int] = []
    skip_reasons: dict[str, int] = {}

    for i, psm in enumerate(psm_list):
        reason = validate_peptidoform(psm.peptidoform)
        if reason is None:
            valid_indices.append(i)
        else:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    if skip_reasons:
        summary = ", ".join(f"{count} {reason}" for reason, count in sorted(skip_reasons.items()))
        logger.warning(
            "Skipped %d/%d PSMs with invalid peptidoforms: %s",
            sum(skip_reasons.values()), len(psm_list), summary,
        )

    return valid_indices, skip_reasons


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
