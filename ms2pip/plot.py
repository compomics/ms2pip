from pathlib import Path
from typing import Union

from ms2pip.spectrum import Spectrum

try:
    import matplotlib.pyplot as plt
    import spectrum_utils.plot as sup

    _can_plot = True
except ImportError:
    _can_plot = False


def spectrum_to_png(spectrum: Spectrum, filepath: Union[str, Path]):
    """Plot a single spectrum and write to a PNG file."""
    if not _can_plot:
        raise ImportError("Matplotlib and spectrum_utils are required to plot spectra.")
    ax = plt.gca()
    ax.set_title("MS²PIP prediction for " + str(spectrum.peptidoform))
    sup.spectrum(spectrum.to_spectrum_utils(), ax=ax)
    plt.savefig(Path(filepath).with_suffix(".png"))
    plt.close()
