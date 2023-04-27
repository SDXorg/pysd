# PySD

[![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/SDXorg/pysd/pulse)
[![Coverage Status](https://coveralls.io/repos/github/SDXorg/pysd/badge.svg?branch=master)](https://coveralls.io/github/SDXorg/pysd?branch=master)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pysd/badges/version.svg)](https://anaconda.org/conda-forge/pysd)
[![PyPI version](https://badge.fury.io/py/pysd.svg)](https://badge.fury.io/py/pysd)
[![PyPI status](https://img.shields.io/pypi/status/pysd.svg)](https://pypi.python.org/pypi/pysd/)
[![Py version](https://img.shields.io/pypi/pyversions/pysd.svg)](https://pypi.python.org/pypi/pysd/)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.04329/status.svg)](https://doi.org/10.21105/joss.04329)
[![Contributions](https://img.shields.io/badge/contributions-welcome-blue.svg)](https://pysd.readthedocs.io/en/latest/development/development_index.html)
[![Docs](https://readthedocs.org/projects/pysd/badge/?version=latest)](https://pysd.readthedocs.io/en/latest/?badge=latest)

![PySD Logo](https://raw.githubusercontent.com/SDXorg/pysd/5cc4fe5dc65e6b5140a00e87a1be9d261570ee8d/docs/images/PySD_Logo_letters.svg?style=centerme)

This project is a library for running [System Dynamics](http://en.wikipedia.org/wiki/System_dynamics) models in Python, with the purpose of improving integration of *Big Data* and *Machine Learning* into the SD workflow.

**The current version needs to run at least Python 3.7.**

## Resources

See the [project documentation](http://pysd.readthedocs.org/) for information about:

- [Installation](http://pysd.readthedocs.org/en/latest/installation.html)
- [Getting Started](http://pysd.readthedocs.org/en/latest/getting_started.html)
- [Release Notes](http://pysd.readthedocs.org/en/latest/whats_new.html)
- [Citing PySD](https://pysd.readthedocs.io/en/latest/#citing)

For standard methods for data analysis with SD models, see the  [PySD Cookbook](https://github.com/SDXorg/PySD-Cookbook), containing (for example):

- [Model Fitting](http://nbviewer.ipython.org/github/SDXorg/PySD-Cookbook/blob/master/source/analyses/fitting/Fitting_with_Optimization.ipynb)
- [Surrogating model components with machine learning regressions](http://nbviewer.ipython.org/github/SDXorg/PySD-Cookbook/blob/master/source/analyses/surrogating_functions/Surrogating_with_regression.ipynb)
- [Multi-Scale geographic comparison of model predictions](http://nbviewer.ipython.org/github/SDXorg/PySD-Cookbook/blob/master/source/analyses/geo/Exploring_models_across_geographic_scales.ipynb)

## Why create a new SD simulation engine?

There are a number of great SD programs out there ([Vensim](http://vensim.com/), [iThink](http://www.iseesystems.com/Softwares/Business/ithinkSoftware.aspx), [AnyLogic](http://www.anylogic.com/system-dynamics), [Insight Maker](http://insightmaker.com/), and [others](https://en.wikipedia.org/wiki/Comparison_of_system_dynamics_software)). In order not to waste our effort, or fall victim to the [Not-Invented-Here](http://en.wikipedia.org/wiki/Not_invented_here) fallacy, we should have a very good reason for starting a new project.

That reason is this: There is a whole world of computational tools being developed in the larger data science community. **System dynamicists should directly use the tools that other people are building, instead of replicating their functionality in SD specific software.** The best way to do this is to bring specific SD functionality to the domain where those other tools are being developed.

This approach allows SD modelers to take advantage of the most recent developments in data science, and focus our efforts on improving the part of the stack that is unique to System Dynamics modeling.

## Cloning this repository

If you'd like to work with this repository directly, you'll need to use a recursive git checkout in order to properly load the test suite (sorry..)

The command should be something like:

```shell
git clone --recursive https://github.com/SDXorg/pysd.git
```

## Extensions

You can use PySD in [R](https://www.r-project.org/) via the [PySD2R](https://github.com/JimDuggan/pysd2r) package, also available on [cran](https://CRAN.R-project.org/package=pysd2r).

## Contributing

PySD is currently a community-maintained project, any contribution is welcome.

Many people have contributed to developing this project - by [submitting code](https://github.com/SDXorg/pysd/graphs/contributors), bug reports, and advice. Main historic changes in PySD are described in the [About PySD section](https://pysd.readthedocs.io/en/latest/about.html). The [Developer Documentation](https://pysd.readthedocs.io/en/latest/development/development_index.html) could help new developers.

The code for this package is available at: https://github.com/SDXorg/pysd

Join our slack channel in [sd-tools-and-methodology-community](https://slofile.com/slack/sdtoolsandmet-slj3251).
