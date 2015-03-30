PySD
====

System Dynamics Modeling in Python

### Status
Version 0.2.1 complete

### Installation
To install via python package index:
```
    pip install pysd
```

## Usage
To import a model from Vensim:
```
    import pysd
    model = pysd.read_vensim('vensim_model_file.mdl')
```
To run the model:
```
    model.run()
```
Model results are by default given as pandas dataframes, so to plot output:
```
    stocks = model.run()
    stocks.plot()
```

#### Making Changes to the model
To run the model with new constant values for parameters:
```
    model.run(params={'parameter_name':value})
```
To run the model with timeseries values for parameters:
```
    parameter_tseries = pd.Series(index=range(30), data=range(20,80,2)) #increasing over 30 timeperiods from 20 to 80
    model.run(params={'parameter_name':parameter_tseries})
```

#### Specifying simulation elements to return
To return model elements other than the default (stocks):
```
    model.run(return_columns=['stock_name', 'flow_name', 'aux_name'])
```
To return simulation values at timestamps other than the default (specified in the model file):
```
    model.run(return_timestamps=[0,1,3,7,9.5,13.178,21,25,30])
```


### Examples
For a simple 'Hello World' type demonstration, see:
- hello world: [Teacup Model](http://nbviewer.ipython.org/github/JamesPHoughton/pysd/blob/master/example_models/Teacup/Teacup.ipynb)

For advanced usage, see some of these examples:

- parameter space exploration: [SIR Model](http://nbviewer.ipython.org/github/JamesPHoughton/pysd/blob/master/example_models/SIR/SIR%20Exploration.ipynb)
- running models interactively in real-time: [Delay Game](http://nbviewer.ipython.org/github/JamesPHoughton/pysd/blob/master/example_models/Delay%20Game/Nth%20Order%20Delay%20Demo.ipynb)
- using ipython's [parallelization capability](http://ipython.org/ipython-doc/dev/parallel/parallel_intro.html) to speed up Monte Carlo Analysis: [Bank Balance Example](http://nbviewer.ipython.org/github/JamesPHoughton/pysd/blob/master/example_models/Bank%20Balance/Parallel%20Monte%20Carlo%20Simulation.ipynb)
- performing Markov Chain Monte Carlo using [pymc](http://pymc-devs.github.io/pymc/README.html) to estimate the number of pennies in circulation from data collected in a penny jar: [Penny Jar Inference](http://nbviewer.ipython.org/github/JamesPHoughton/pysd/blob/master/example_models/Penny%20Jar/Penny%20Jar.ipynb)
- several optimization methods using [scipy.optimize](http://docs.scipy.org/doc/scipy-0.14.0/reference/optimize.html): [Predator-Prey Model](http://nbviewer.ipython.org/github/JamesPHoughton/pysd/blob/master/example_models/Predator-Prey/Predator%20-%20Prey.ipynb)

## About the project
This project is a simple library for running [System Dynamics](http://en.wikipedia.org/wiki/System_dynamics) models in python, with the purpose of improving integration of *Big Data* and *Machine Learning* into the SD workflow. 

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

### Notional Capabilities Development Pathway

The initial use case would be to import a fully developed SD model (created in Vensim, or Stella/iThink, etc.) into PySD via the XMILE format; and then use third party tools to perform statistical analysis and inference, and to interface with external data.

####Current Features

1. Basic XMILE and Vensim parser, limited to a subset of functions for which the Markov Property holds
4. Established library structure and data formats
4. Simulation using existing python integration tools
5. Basic demonstrations of integration with python Data Science functionality
6. Run-at-a-time parameter modification

####Planned Features

1. Step-at-a-time parameter modification / time-variant exogenous inputs
2. Extended backends for storing parameters and output values
3. More extensive XMILE and Vensim model parsers
4. Demonstration of integration with Machine Learning/Monte Carlo/Statistical Methods
5. Integration with EMA Workbench

####Possible features

- Embed SD.js front end in iPython widget
- XMILE display component parser
- Customizations of external Data Science tools for dynamic systems
- Incorporation of analysis tools specific to dynamic systems
- Python methods for programmatically manipulating SD model structure
- Additional SD tools: checks for model units, value limits, etc.
- Python methods for saving XMILE models
- Manage varying the time step of integration/adaptive integration time step
- Infer logical ways to automatically lay out stock and flow diagrams
- Add hooks to vensim, other SD programs, to allow running models with other engines
- Turn off and on 'traces' or records of the values of variables
- Show different 'submodels' in the model diagram
- Infer units when possible from other areas
- Hover over stock/flow elements to get things like units, descriptions, values, etc.
- Output DataFrame including tags for units


### PySD Design Philosophy

- Do as little as possible. 
 - Anything that is not endemic to System Dynamics (such as plotting, integration, fitting, etc) should either be implemented using external tools, or omitted. 
 - Stick to SD. Let other disciplines (ABM, Discrete Event Simulation, etc) create their own tools.
 - Use external model creation tools
- Use the language of system dynamics.
- Be simple to use. Let SD practitioners who haven't used python before understand the basics.
- Take advantage of general python constructions and best practices.
- Be simple to maintain.    

