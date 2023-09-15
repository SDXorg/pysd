# -*- coding: utf-8 -*-
#
# PySD documentation build configuration file, created by
# sphinx-quickstart on Thu Jun 18 10:02:28 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys

import mock
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../'))

from docs.generate_tables import generate_tables


# Generate tables used for documentation
generate_tables()

MOCK_MODULES = [
    'numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.stats',
    'scipy.integrate', 'pandas', 'parsimonious', 'parsimonious.nodes',
    'xarray', 'autopep8', 'scipy.linalg', 'parsimonious.exceptions',
    'scipy.stats.distributions', 'progressbar', 'black', 'scipy.optimize'
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks'
]

extlinks = {
    "issue": ("https://github.com/SDXorg/pysd/issues/%s", "issue #%s"),
    "pull": ("https://github.com/SDXorg/pysd/pull/%s", "PR #%s"),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'PySD'
copyright = '2022, PySD contributors'
author = 'PySD contributors'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#

# The short X.Y version.
__version__ = "x.x.x"
exec(open('../pysd/_version.py').read())
version = '.'.join(__version__.split('.')[:-1])

# The full version, including alpha/beta/rc tags.
release = __version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'PySDdoc'
html_logo = "images/PySD_Logo_letters.png"
html_favicon = "images/PySD_Logo.ico"
html_theme_options = {
    'logo_only': True,
}


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
# author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, 'PySD.tex', 'PySD Documentation',
   'PySD contributors', 'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'pysd', 'PySD Documentation',
     [author], 1)
]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'PySD', 'PySD Documentation',
   author, 'PySD', 'One line description of project.',
   'Miscellaneous'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.8', None),
    'pysdcookbook': ('http://pysd-cookbook.readthedocs.org/en/latest/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pytest': ('https://docs.pytest.org/en/7.1.x/', None),
    'openpyxl': ('https://openpyxl.readthedocs.io/en/stable', None)
}

# -- Options for autodoc --------------------------------------------------
autodoc_member_order = 'bysource'
