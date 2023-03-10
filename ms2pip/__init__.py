# isort: skip_file
"""MS2PIP: Accurate and versatile peptide fragmentation spectrum prediction."""

__version__ = "4.0.0-dev0"

from ms2pip.core import (
    predict_single,
    predict_batch,
    predict_library,
    correlate,
    get_training_data,
    match_spectra,
)
