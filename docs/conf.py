import os
import sys

project = "rl-insight"
copyright = "2026 rl-insight contributors"
author = "rl-insight community"

master_doc = "index"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

