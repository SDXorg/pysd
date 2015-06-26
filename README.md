PySD
====

System Dynamics Modeling in Python

This project is a simple library for running [System Dynamics](http://en.wikipedia.org/wiki/System_dynamics) models in python, with the purpose of improving integration of *Big Data* and *Machine Learning* into the SD workflow. 

## Resources
See the [project documentation](http://pysd.readthedocs.org/) for information about:

- [Installation](http://pysd.readthedocs.org/en/latest/installation.html)
- [Basic Usage](http://pysd.readthedocs.org/en/latest/basic_usage.html)
- [Function Reference](http://pysd.readthedocs.org/en/latest/functions.html)

For standard methods for data analysis with SD models, see the  [PySD Cookbook](https://github.com/JamesPHoughton/PySD-Cookbook), containing (for example):

- [Model Fitting](http://nbviewer.ipython.org/github/JamesPHoughton/PySD-Cookbook/blob/master/2_1_Fitting_with_Optimization.ipynb)
- [Surrogating model components with machine learning regressions](http://nbviewer.ipython.org/github/JamesPHoughton/PySD-Cookbook/blob/master/6_1_Surrogating_with_regression.ipynb)
- [Multi-Scale geographic comparison of model predictions](http://nbviewer.ipython.org/github/JamesPHoughton/PySD-Cookbook/blob/master/Exploring%20models%20across%20geographic%20scales.ipynb)

If you use PySD in any published work, consider citing the [PySD Introductory Paper](https://github.com/JamesPHoughton/pysd/blob/master/docs/PySD%20Intro%20Paper%20Preprint.pdf):

>Houghton, James; Siegel, Michael. "Advanced data analytics for system dynamics models using PySD." *Proceedings of the 33rd International Conference of the System Dynamics Society.* 2015.


## About the project

### Why create a new SD modeling engine?

There are a number of great SD programs out there ([Vensim](http://vensim.com/), [iThink](http://www.iseesystems.com/Softwares/Business/ithinkSoftware.aspx), [AnyLogic](http://www.anylogic.com/system-dynamics), [Insight Maker](http://insightmaker.com/), [Forio](http://forio.com/), and [others](http://en.wikipedia.org/wiki/List_of_system_dynamics_software)). Additionally, there are a number of existing python-based dynamic system modeling tools, such as [PyDSTool](http://www.ni.gsu.edu/~rclewley/PyDSTool/FrontPage.html) and [others](http://www.scipy.org/topical-software.html#dynamical-systems). In order not to waste our effort, or fall victim to the [Not-Invented-Here](http://en.wikipedia.org/wiki/Not_invented_here) fallacy, we should have a very good reason for starting a new project. 

That reason is this: There is a whole world of computational tools being developed in the larger technical community. **We should directly use the tools that other people are building, instead of replicating their functionality in SD specific software.** The best way to do this is to bring specific SD functionality to the domain where those other tools are being developed. 

Several successful projects have built harnesses to allow control of SD programs from python. This can be an unwieldy solution, and in order to scale the analysis to clusters of computers running millions of iterations, with speed and precise control of execution, an integrated solution is preferred.

This approach allows SD modelers to take advantage of the most recent developments in data science, and focus our efforts on improving the part of the stack that is unique to System Dynamics modeling.

#### What computational tools are we talking about?

System dynamics and other programming and analysis techniques require some common components:

Data components:

1. Data gathering - possibly in real time, from web sources
2. Data storage - potentially in distributed, cloud oriented databases
4. Data formatting, verification, transformation, 
5. Timeseries data interpolation and alignment
5. Geographic data manipulation

Computational components:

1. High speed numerical integration - gpu techniques, etc
2. Multiprocessing - possibly distributed, cloud based parallel computing
7. Statistical analysis / inference
8. Machine Learning algorithms / Neural Networks / AI
9. Optimization techniques

Display components:

1. Real-time data visualization 
2. Interactive data visualization - possibly web based, at scale
3. Theory/documentation/code/result colocation and presentation

Collaboration components:

1. Real-time collaboration
2. Version control
4. Open sourcing/archiving
3. Analysis replicability - improving research replicating

The Future...

#### What components are specific to System Dynamics?

1. System Dynamics model equations
3. System Dynamics model diagrams - stock and flow, CLD
3. Advanced SD concepts - coflows, aging chains, etc.
4. Dynamic model analysis tools
4. Future Developments in SD modeling

These are the components that should be included in PySD.

### Complementary Projects

The most valuable component for better integrating models with *basically anything else* is a standard language for communicating the structure of those models. That language is [XMILE](http://www.iseesystems.com/community/support/XMILE.aspx). The draft specifications for this have been finalized and the standard should be approved in the next few months.

A python library for analyzing system dynamics models called the [Exploratory Modeling and Analysis (EMA) Workbench](http://simulation.tbm.tudelft.nl/ema-workbench/contents.html) is being developed by [Erik Pruyt](http://www.tbm.tudelft.nl/en/about-faculty/departments/multi-actor-systems/policy-analysis/people/erik-pruyt/) and [Jan Kwakkel](https://github.com/quaquel) at TU Delft. This package implements a variety of analysis methods that are unique to dynamic models, and could work very tightly with PySD. 

An excellent javascript library called [sd.js](https://github.com/bpowers/sd.js/tree/master) created by Bobby Powers at [SDlabs](http://sdlabs.io/) exists as a standalone SD engine, and provides a beautiful front end. This front end could be rendered as an iPython widget to facilitate display of SD models.

The [Behavior Analysis and Testing Software(BATS)](http://www.ie.boun.edu.tr/labs/sesdyn/projects/bats/index.html) delveloped by [Gönenç Yücel](http://www.ie.boun.edu.tr/people/pages/yucel.html) includes a really neat method for categorizing behavior modes and exploring parameter space to determine the boundaries between them.


