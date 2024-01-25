"""Command line utilities for MS²PIP."""

from rich.table import Table
from rich.text import Text

from ms2pip import __version__


def build_credits():
    """Build credits."""
    text = Text()
    text.append("\n")
    text.append("MS²PIP", style="bold link https://github.com/compomics/ms2pip")
    text.append(f" (v{__version__})\n", style="bold")
    text.append("Developed at CompOmics, VIB / Ghent University, Belgium.\n")
    text.append("Please cite: ")
    text.append("Declercq et al. NAR (2023)", style="link https://doi.org/10.1093/nar/gkad335")
    text.append("\n")
    text.stylize("cyan")
    return text


def build_prediction_table(predicted_spectrum):
    """Print a table with the predicted spectrum values."""
    peptidoform = str(predicted_spectrum.peptidoform)
    table = Table(title=f"Predicted spectrum for [bold]{peptidoform}[/bold]")

    table.add_column("m/z", justify="right")
    table.add_column("intensity", justify="right")
    table.add_column("annotation", justify="left")

    for mz, intensity, annotation in zip(
        predicted_spectrum.mz,
        predicted_spectrum.intensity,
        predicted_spectrum.annotations,
    ):
        style = "blue" if "b" in annotation else "red" if "y" in annotation else "black"
        table.add_row(f"{mz:.4f}", f"{intensity:.4f}", annotation, style=style)

    return table
