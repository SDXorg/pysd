Developer Documentation
=======================

Motivation: The (coming of) age of Big Data
-------------------------------------------

The last few years have witnessed a massive growth in the collection of social and business data, and a corresponding boom of interest in learning about the behavior of social and business systems using this data. The field of ‘data science’ is developing a host of new techniques for dealing with and analyzing data, responding to an increase in demand for insights and the increased power of computing resources.

So far, however, these new techniques are largely confined to variants of statistical summary, categorization, and inference; and if causal models are used, they are generally static in nature, ignoring the dynamic complexity and feedback structures of the systems in question. As the field of data science matures, there will be increasing demand for insights beyond those available through analysis unstructured by causal understanding. At that point data scientists may seek to add dynamic models of system structure to their toolbox.

The field of system dynamics has always been interested in learning about social systems, and specializes in understanding dynamic complexity. There is likewise a long tradition of incorporating various forms of data into system dynamics models.3 While system dynamics practice has much to gain from the emergence of new volumes of social data, the community has yet to benefit fully from the data science revolution.

There are a variety of reasons for this, the largest likely being that the two communities have yet to commingle to a great extent. A further, and ultimately more tractable reason is that the tools of system dynamics and the tools of data analytics are not tightly integrated, making joint method analysis unwieldy. There is a rich problem space that depends upon the ability of these fields to support one another, and so there is a need for tools that help the two methodologies work together. PySD is designed to meet this need.

General approaches for integrating system dynamic models and data analytics
---------------------------------------------------------------------------
Before considering how system dynamics techniques can be used in data science applications, we should consider the variety of ways in which the system dynamics community has traditionally dealt with integration of data and models. 

The first paradigm for using numerical data in support of modeling efforts is to import data into system dynamics modeling software. Algorithms for comparing models with data are built into the tool itself, and are usable through a graphical front-end interface as with model fitting in Vensim, or through a programming environment unique to the tool. When new techniques such as Markov chain Monte Carlo analysis become relevant to the system dynamics community, they are often brought into the SD tool.
 
This approach appropriately caters to system dynamics modelers who want to take advantage of well-established data science techniques without needing to learn a programming language, and extends the functionality of system dynamics to the basics of integrated model analysis.

A second category of tools uses a standard system dynamics tool as a computation engine for analysis performed in a coding environment. This is the approach taken by the Exploratory Modeling Analysis (EMA) Workbench6, or the Behavior Analysis and Testing Software (BATS)7. This first step towards bringing system dynamics to a more inclusive analysis environment enables many new types of model understanding, but imposes limits on the depth of interaction with models and the ability to scale simulation to support large analysis.

A third category of tools imports the models created by traditional tools to perform analyses independently of the original modeling tool. An example of this is SDM-Doc8, a model documentation tool, or Abdel-Gawad et. al.’s eigenvector analysis tool9. It is this third category to which PySD belongs. 
 
The central paradigm of PySD is that it is more efficient to bring the mature capabilities of system dynamics into an environment in use for active development in data science, than to attempt to bring each new development in inference and machine learning into the system dynamics enclave.

PySD reads a model file – the product of a modeling program such as Vensim10 or Stella/iThink11 – and cross compiles it into Python, providing a simulation engine that can run these models natively in the Python environment. It is not a substitute for these tools, and cannot be used to replace a visual model construction environment.

Structure of the PySD module
----------------------------
PySD provides a set of translators that interpret a Vensim or XMILE12 format model into a Python native class. The model components object represents the state of the system, and contains methods that compute auxiliary and flow variables based upon the current state. 

The components object is wrapped within a Python class that provides methods for modifying and executing the model. These three pieces constitute the core functionality of the PySD module, and allow it to interact with the Python data analytics stack.

Builder
^^^^^^^



Translation
^^^^^^^^^^^
The PySD module is capable of importing models from a Vensim model file (*.mdl) or an XMILE format xml file. Translation makes use of a Parsing Expression Grammar parser, using the third party Python library Parsimonious13 to construct an abstract syntax tree based upon the full model file (in the case of Vensim) or individual expressions (in the case of XMILE). 

The translators then crawl the tree, using a dictionary to translate Vensim or Xmile syntax into its appropriate Python equivalent. The use of a translation dictionary for all syntactic and programmatic components prevents execution of arbitrary code from unverified model files, and ensures that we only translate commands that PySD is equipped to handle. Any unsupported model functionality should therefore be discovered at import, instead of at runtime.

The use of a one-to-one dictionary in translation means that the breadth of functionality is inherently limited. In the case where no direct Python equivalent is available, PySD provides a library of functions such as pulse, step, etc. that are specific to dynamic model behavior.

In addition to translating individual commands between Vensim/XMILE and Python, PySD reworks component identifiers to be Python-safe by replacing spaces with underscores. The translator allows source identifiers to make use of alphanumeric characters, spaces, or the $ symbol.

The model class
^^^^^^^^^^^^^^^
The translator constructs a Python class that represents the system dynamics model. The class maintains a dictionary representing the current values of each of the system stocks, and the current simulation time, making it a ‘statefull’ model in much the same way that the system itself has a specific state at any point in time.

The model class also contains a function for each of the model components, representing the essential model equations. The docstring for each function contains the model documentation and units as translated from the original model file. A query to any of the model functions will calculate and return its value according to the stored state of the system.

The model class maintains only a single state of the system in memory, meaning that all functions must obey the Markov property  - that the future state of the system can be calculated entirely based upon its current state. In addition to simplifying integration, this requirement enables analyses that interact with the model at a step-by-step level. The downside to this design choice is that several components of Vensim or XMILE functionality – the most significant being the infinite order delay – are intentionally not supported. In many cases similar behavior can be approximated through other constructs.

Lastly, the model class provides a set of methods that are used to facilitate simulation. PySD uses the standard ordinary differential equations solver provided in the well-established Python library Scipy, which expects the state and its derivative to be represented as an ordered list. The model class provides the function .d_dt() that takes a state vector from the integrator and uses it to update the model state, and then calculates the derivative of each stock, returning them in a corresponding vector. A complementary function .state_vector() creates an ordered vector of states for use in initializing the integrator.
The PySD class
The PySD class provides the machinery to get the model moving, supply it with data, or modify its parameters. In addition, this class is the primary way that users interact with the PySD module.

The basic function for executing a model is appropriately named.run(). This function passes the model into scipy’s odeint() ordinary differential equations solver. The scipy integrator is itself utilizing the lsoda integrator from the Fortran library odepack14, and so integration takes advantage of highly optimized low-level routines to improve speed. We use the model’s timestep to set the maximum step size for the integrator’s adaptive solver to ensure that the integrator properly accounts for discontinuities.

The .run() function returns to the user a Pandas dataframe representing the output of their simulation run. A variety of options allow the user to specify which components of the model they would like returned, and the timestamps at which they would like those measurements. Additional parameters make parameter changes to the model, modify its starting conditions, or specify how simulation results should be logged.
