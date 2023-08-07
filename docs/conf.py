# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))



# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Megatron-LLM'
copyright = '2023, Alejandro Hernández Cano, Matteo Pagliardini, Kyle Matoba, Amirkeivan Mohtashami, Olivia Simin Fan, Axel Marmet, Deniz Bayazit, Igor Krawczuk, Zeming Chen, Francesco Salvi, Antoine Bosselut, Martin Jaggi'
author = 'Alejandro Hernández Cano, Matteo Pagliardini, Kyle Matoba, Amirkeivan Mohtashami, Olivia Simin Fan, Axel Marmet, Deniz Bayazit, Igor Krawczuk, Zeming Chen, Francesco Salvi, Antoine Bosselut, Martin Jaggi'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.intersphinx',
        'sphinx.ext.autosummary',
        'sphinx.ext.napoleon',
        'sphinx.ext.mathjax',
        'myst_parser'
]

# autosummary
autosummary_generate = True

# napoleon
napoleon_google_docstring = True

# myst
myst_enable_extensions = ["colon_fence"]

# autodoc
autodoc_mock_imports = ['amp_C', 'torchvision', 'flash_attn', 'apex']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None)
}

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
