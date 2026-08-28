"""Sphinx configuration for the rayTEM documentation set."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys


# -- Path setup --------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# -- Project information -----------------------------------------------------
project = "rayTEM"
author = "rayTEM contributors"
copyright = f"{datetime.now().year}, {author}"
release = "0.0.1beta"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# this repository's docstring style uses a "Related" section; alias it to
# napoleon's See Also parsing so it renders instead of tripping field lists
napoleon_custom_sections = [("Related", "see also")]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

# Enable common MyST markdown niceties
myst_enable_extensions = ["colon_fence", "deflist"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
