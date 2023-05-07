# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
import pathlib
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
print(f"{sys.path=}")


# -- Project information -----------------------------------------------------

project = 'CrossPy'
copyright = '2021-2023, University of Texas at Austin'
author = 'Bozhi YOU'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.doctest',
  'sphinx.ext.intersphinx',
  # 'sphinx_autodoc_typehints',
  'sphinx.ext.todo',
  'sphinx.ext.mathjax',
  # 'sphinx.ext.viewcode'
  "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"

html_theme = "pydata_sphinx_theme"
# html_theme_options = {
#     #'show_prev_next': False,
#     'github_url': 'https://github.com/spatialaudio/nbsphinx',
#     'use_edit_page_button': True,
# }
# html_context = {
#     'github_user': 'spatialaudio',
#     'github_repo': 'nbsphinx',
#     'github_version': 'master',
#     'doc_path': 'doc',
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_show_sourcelink = False


# -- Jupyter Notebook support --

nb_execution_mode = "cache"
nb_execution_in_temp = True