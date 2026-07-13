"""Configuration file for the Sphinx documentation builder."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

from ms2pip import __version__

# Project information
project = "ms2pip"
author = "CompOmics"
github_project_url = "https://github.com/compomics/ms2pip/"
github_doc_root = "https://github.com/compomics/ms2pip/tree/main/docs/"

# Version
release = __version__

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_click.ext",
    "sphinx_rtd_theme",
]
source_suffix = [".rst", ".md"]
master_doc = "index"
exclude_patterns = ["_build"]

# Options for HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Autodoc options
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "init"
# autodoc_type_aliases = {
#     "Path": "pathlib.Path",
#     "DataFrame": "pandas.DataFrame",
#     "Series": "pandas.Series",
#     "PSMList": "psm_utils.psm_list.PSMList",
# }

# Intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "psm_utils": ("https://psm-utils.readthedocs.io/en/stable/", None),
}

# Napoleon options
# numpydoc_xref_aliases = autodoc_type_aliases


def setup(app):
    config = {
        # "auto_toc_tree_section": "Contents",
        "enable_eval_rst": True,
    }
