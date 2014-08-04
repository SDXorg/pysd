PySD
====

System Dynamics Modeling in Python

This project will create simple library for running [System Dynamics](http://en.wikipedia.org/wiki/System_dynamics) models in python, with the purpose of improving integration of *Big Data* and *Machine Learning* into the SD workflow. 

## Why create a new SD modeling engine?

There are a number of great SD programs out there ([Vensim](http://vensim.com/), [iThink](http://www.iseesystems.com/Softwares/Business/ithinkSoftware.aspx), [AnyLogic](http://www.anylogic.com/system-dynamics), [Insight Maker](http://insightmaker.com/), [Forio](http://forio.com/), and [others](http://en.wikipedia.org/wiki/List_of_system_dynamics_software)). Additionally, there are a number of existing python-based dynamic system modeling tools, such as [PyDSTool](http://www.ni.gsu.edu/~rclewley/PyDSTool/FrontPage.html) and [others](http://www.scipy.org/topical-software.html#dynamical-systems). In order not to waste our effort, or fall victim to the [Not-Invented-Here](http://en.wikipedia.org/wiki/Not_invented_here) fallacy, we should have a very good reason for starting a new project. 

That reason is this: There is a whole world of computational tools being developed in the larger technical community. **We should directly use the tools that other people are building, instead of replicating their functionality in SD specific software.** The best way to do this is to bring specific SD functionality to the domain where those other tools are being developed. 

Efforts have been made to build harnesses for SD programs that allow them to be operated from python, and have been shown to be useful and effective. However, in order to scale the analysis to clusters of computers running millions of iterations, with speed and precise control of execution, an integrated solution is preferred.

This approach allows SD modelers to take advantage of the most recent developments in data science, and focus our efforts on improving the part of the stack that is unique to System Dynamics modeling.

#### What computational tools are we talking about?

System dynamics and other programming and analysis techniques require some common components:

Data components:

1. Data Gathering - possibly in real time, from web sources
2. Data Storage - potentially in distributed, cloud oriented databases
4. Data formatting, verification, transformation, 
5. Timeseries Data interpolation and alignment
5. Geographic Data Manipulation

Computational components:

1. High speed numerical integration - gpu techniques, etc
2. Multiprocessing - possibly distributed, cloud based parallel computing
7. Statistical analysis / inference
8. Machine Learning algorithms / Neural Networks / AI
9. Optimization Techniques

Display components:

1. Real-time data visualization 
2. Interactive data visualization - possibly web based, at scale
3. Theory/Documentation/Code/Result colocation

Collaboration components:

1. Real-time collaboration
2. Version Control
4. Open Sourcing/Archiving
3. Analysis Replicability - improving research replicating

The Future...

#### What components are specific to System Dynamics?

1. System Dynamics Model Equations
3. System Dynamics Model Diagrams - Stock and Flow, CLD
3. Advanced SD concepts - Coflows, Aging Chains, etc.
4. Dynamic Model Analysis tools
4. Future Developments in SD modeling

These are the components that should be included in PySD.

## Complementary Projects

The most valuable component for better integrating models with *basically anything else* is a standard language for communicating the structure of those models. That language is [XMILE](http://www.iseesystems.com/community/support/XMILE.aspx). The draft specifications for this have been finalized and the standard should be adopted in the next few months.

A python library for analysing system dynamics models called the [Exploratory Modeling and Analysis (EMA) Workbench](http://simulation.tbm.tudelft.nl/ema-workbench/contents.html) is being developed by [Erik Pruyt](http://www.tbm.tudelft.nl/en/about-faculty/departments/multi-actor-systems/policy-analysis/people/erik-pruyt/) and [Jan Kwakkel](https://github.com/quaquel) at TU Delft. This package implements a variety of analysis methods that are unique to dynamic models, and could work very tightly with PySD. 

An excellent javascript library called [sd.js](https://github.com/bpowers/sd.js/tree/master) created by Bobby Powers at [SDlabs](http://sdlabs.io/) exists as a standalone SD engine, and provides a beautiful front end. This front end could be rendered as an iPython widget to facilitate display of SD models.

The [Behavior Analysis and Testing Software(BATS)](http://www.ie.boun.edu.tr/labs/sesdyn/projects/bats/index.html) delveloped by [Gönenç Yücel](http://www.ie.boun.edu.tr/people/pages/yucel.html) includes a really neat method for categorizing behavior modes and exploring parameter space to determine the boundaries between them.

## Notional Capabilities Development Pathway

Initially (at least) the use case would be to bring a fully developed SD model (created in Vensim, or Stella/iThink, etc.) into PySD via the XMILE format, and use third party libraries to apply the heavy hammers of statistical analysis to models.

####Version 0: Proof of Concept

1. Basic XMILE parser (possibly limited to subset of XMILE functions for which the Markov Property holds)
2. Provision of derivatives function based upon stock values
3. Stock initialization from XMILE spec
4. Establish library structure and data formats
4. Simulation using existing python integration tools
5. Basic demonstrations of integration with python Data Science functionality

####Version 1: Actually Useful

1. Parameter modification (run-at-a-time, step-at-a-time)
2. Extended backends for storing parameters and run output values
3. Full XMILE Model component parser
4. Demonstration of integration with Machine Learning/Monte Carlo/Statistical Methods
5. Integration with EMA Workbench

####Version 2: Maturity

1. Embed SD.js front end in iPython widget
2. Python methods for saving XMILE changes
4. Customizations of external Data Science tools for dynamic systems
3. Incorporation of analysis tools specific to dynamic systems

####Someday/Maybe

- Python methods for programmatically manipulating SD model structure
- Additional SD tools: checks for model units, value limits, etc.
- Manage varying the time step of integration/adaptive integration time step
- Infer logical ways to automatically lay out stock and flow diagrams
- Add hooks to vensim, other SD programs, to allow running models with other engines
- Turn off and on 'traces' or records of the values of variables
- Show different 'submodels' in the model diagram
- Infer units when possible from other areas
- Hover over stock/flow elements to get things like units, descriptions, values, etc.
- Output DataFrame including tags for units


## PySD Design Philosophy

- Do as little as possible. 
 - Anything that is not endemic to System Dynamics (such as plotting, integration, fitting, etc) should either be implemented using external tools, or omitted. 
 - Stick to SD. Let other disciplines (ABM, Discrete Event Simulation, etc) create their own tools.
 - Use external model creation tools
- Use the language of system dynamics.
- Be simple to use. Let SD practitioners who haven't used python before understand the basics.
- Take advantage of general python constructions and best practices.
- Be simple to maintain.    

