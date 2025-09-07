import os
import sys

# Add the src directory to sys.path to allow autodoc to find modules
sys.path.insert(0, os.path.abspath("../src"))

project = "nircls"
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"
