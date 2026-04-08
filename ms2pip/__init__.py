# isort: skip_file
"""MS2PIP: Accurate and versatile peptide fragmentation spectrum prediction."""

__version__ = "4.2.0-alpha.2"
__all__ = [
    "predict_single",
    "predict_batch",
    "predict_library",
    "correlate",
    "correlate_preloaded",
    "get_training_data",
    "annotate_spectra",
    "download_models",
]

from warnings import filterwarnings

filterwarnings(
    "ignore", message="hdf5plugin is missing", category=UserWarning, module="psims.mzmlb"
)


from ms2pip.core import (  # noqa: E402
    predict_single,
    predict_batch,
    predict_library,
    correlate,
    correlate_preloaded,
    get_training_data,
    annotate_spectra,
    download_models,
)
