PySD
====

[![Coverage Status](https://coveralls.io/repos/github/JamesPHoughton/pysd/badge.svg?branch=master)](https://coveralls.io/github/JamesPHoughton/pysd?branch=master)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pysd/badges/version.svg)](https://anaconda.org/conda-forge/pysd)
[![PyPI version](https://badge.fury.io/py/pysd.svg)](https://badge.fury.io/py/pysd)
[![PyPI status](https://img.shields.io/pypi/status/pysd.svg)](https://pypi.python.org/pypi/pysd/)
[![Py version](https://img.shields.io/pypi/pyversions/pysd.svg)](https://pypi.python.org/pypi/pysd/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5654824.svg)](https://doi.org/10.5281/zenodo.5654824)
[![Docs](https://readthedocs.org/projects/pysd/badge/?version=latest)](https://pysd.readthedocs.io/en/latest/?badge=latest)

Simulating System Dynamics Models in Python

This project is a simple library for running [System Dynamics](http://en.wikipedia.org/wiki/System_dynamics) models in python, with the purpose of improving integration of *Big Data* and *Machine Learning* into the SD workflow.

**The current version needs to run at least Python 3.7.**

### Resources

See the [project documentation](http://pysd.readthedocs.org/) for information about:

- [Installation](http://pysd.readthedocs.org/en/latest/installation.html)
- [Getting Started](http://pysd.readthedocs.org/en/latest/getting_started.html)
- [Release Notes](http://pysd.readthedocs.org/en/latest/whats_new.html)
- [Citing PySD](https://pysd.readthedocs.io/en/latest/#citing)

For standard methods for data analysis with SD models, see the  [PySD Cookbook](https://github.com/JamesPHoughton/PySD-Cookbook), containing (for example):

- [Model Fitting](http://nbviewer.ipython.org/github/JamesPHoughton/PySD-Cookbook/blob/master/2_1_Fitting_with_Optimization.ipynb)
- [Surrogating model components with machine learning regressions](http://nbviewer.ipython.org/github/JamesPHoughton/PySD-Cookbook/blob/master/6_1_Surrogating_with_regression.ipynb)
- [Multi-Scale geographic comparison of model predictions](http://nbviewer.ipython.org/github/JamesPHoughton/PySD-Cookbook/blob/master/Exploring%20models%20across%20geographic%20scales.ipynb)

### Why create a new SD simulation engine?

There are a number of great SD programs out there ([Vensim](http://vensim.com/), [iThink](http://www.iseesystems.com/Softwares/Business/ithinkSoftware.aspx), [AnyLogic](http://www.anylogic.com/system-dynamics), [Insight Maker](http://insightmaker.com/), and [others](http://en.wikipedia.org/wiki/List_of_system_dynamics_software)). In order not to waste our effort, or fall victim to the [Not-Invented-Here](http://en.wikipedia.org/wiki/Not_invented_here) fallacy, we should have a very good reason for starting a new project.

That reason is this: There is a whole world of computational tools being developed in the larger data science community. **System dynamicists should directly use the tools that other people are building, instead of replicating their functionality in SD specific software.** The best way to do this is to bring specific SD functionality to the domain where those other tools are being developed.

This approach allows SD modelers to take advantage of the most recent developments in data science, and focus our efforts on improving the part of the stack that is unique to System Dynamics modeling.

### Cloning this repository

If you'd like to work with this repository directly, you'll need to use a recursive git checkout in order to properly load the test suite (sorry..)

The command should be something like:
```shell
git clone --recursive https://github.com/JamesPHoughton/pysd.git
```

### Extensions

You can use PySD in [R](https://www.r-project.org/) via the [PySD2R](https://github.com/JimDuggan/pysd2r) package, also available on [cran](https://CRAN.R-project.org/package=pysd2r).

### Contributors

Many people have contributed to developing this project - by
[submitting code](https://github.com/JamesPHoughton/pysd/graphs/contributors), bug reports, and advice.

Special thanks to the [sdCloud.io](http://sdcloud.io) development team, who have
made great contributions to XMILE support, and for integrating PySD into their cloud-based model simulation environment.

Extra special thanks to [@enekomartinmartinez](https://github.com/enekomartinmartinez) for dramatically pushing forward subscript capabilities (and many other attributes).
