About the Project
=================

PySD was originally developed by `James P Houghton <https://github.com/JamesPHoughton>`_ in 2014 as a software to translate Vensim to Python. The original goal for translating SD models into Python was to be able to take advantage of all the tools available in Python and thus be able to solve a lot of problems that Vensim itself does not allow.

Since the creation of the library many people have contributed to the project by reporting and fixing bugs and adding new features. These contributions are listed in the `contributions section of the GitHub repository <https://github.com/SDXorg/pysd/graphs/contributors>`_.

Some of the big changes that have allowed PySD to get to where you are now are the development of an XMILE to Python code translator in 2017 by `Alex Prey <https://github.com/alexprey>`_ and the restructuring of the translation and model building through an Abstract Syntax by `Eneko Martin-Martinez <https://github.com/enekomartinmartinez>`_ in 2022.

Some other contributions  until release 3.0.0 are:

- `Julien Malard-Adam <https://github.com/julienmalard>`_ has added unicode support for the Vensim parser.
- `sdCloud.io <http://sdcloud.io>`_ development team have made great contributions to XMILE support and have integrated PySD into their cloud-based model simulation environment.
- `Eneko Martin-Martinez <https://github.com/enekomartinmartinez>`_ has pushed forward the subscripts capabilities for both Vensim and XMILE and has included support for several Vensim functions and improved the performance.
- `Roger Samsó <https://github.com/rogersamso>`_ has included a sketch parser for Vensim model and has added the option to split a Vensim model per view based on the sketch information.

The changes made since release 3.0.0 are tracked in the :doc:`whats_new` section.


Motivation: The (coming of) age of Big Data
-------------------------------------------
The last few years have witnessed a massive growth in the collection of social and business data, and a corresponding boom of interest in learning about the behavior of social and business systems using this data. The field of ‘data science’ is developing a host of new techniques for dealing with and analyzing data, responding to an increase in demand for insights and the increased power of computing resources.

So far, however, these new techniques are largely confined to variants of statistical summary, categorization, and inference; and if causal models are used, they are generally static in nature, ignoring the dynamic complexity and feedback structures of the systems in question. As the field of data science matures, there will be increasing demand for insights beyond those available through analysis unstructured by causal understanding. At that point data scientists may seek to add dynamic models of system structure to their toolbox.

The field of system dynamics has always been interested in learning about social systems, and specializes in understanding dynamic complexity. There is likewise a long tradition of incorporating various forms of data into system dynamics models. While system dynamics practice has much to gain from the emergence of new volumes of social data, the community has yet to benefit fully from the data science revolution.

There are a variety of reasons for this, the largest likely being that the two communities have yet to commingle to a great extent. A further, and ultimately more tractable reason is that the tools of system dynamics and the tools of data analytics are not tightly integrated, making joint method analysis unwieldy. There is a rich problem space that depends upon the ability of these fields to support one another, and so there is a need for tools that help the two methodologies work together. PySD is designed to meet this need.


General approaches for integrating system dynamic models and data analytics
---------------------------------------------------------------------------
Before considering how system dynamics techniques can be used in data science applications, we should consider the variety of ways in which the system dynamics community has traditionally dealt with integration of data and models.

The first paradigm for using numerical data in support of modeling efforts is to import data into system dynamics modeling software. Algorithms for comparing models with data are built into the tool itself, and are usable through a graphical front-end interface as with model fitting in Vensim, or through a programming environment unique to the tool. When new techniques such as Markov chain Monte Carlo analysis become relevant to the system dynamics community, they are often brought into the SD tool.

This approach appropriately caters to system dynamics modelers who want to take advantage of well-established data science techniques without needing to learn a programming language, and extends the functionality of system dynamics to the basics of integrated model analysis.

A second category of tools uses a standard system dynamics tool as a computation engine for analysis performed in a coding environment. This is the approach taken by the Exploratory Modeling Analysis (EMA) Workbench, or the Behavior Analysis and Testing Software (BATS). This first step towards bringing system dynamics to a more inclusive analysis environment enables many new types of model understanding, but imposes limits on the depth of interaction with models and the ability to scale simulation to support large analysis.

A third category of tools imports the models created by traditional tools to perform analyses independently of the original modeling tool. An example of this is SDM-Doc, a model documentation tool, or Abdel-Gawad et. al.’s eigenvector analysis tool. It is this third category to which PySD belongs.

The central paradigm of PySD is that it is more efficient to bring the mature capabilities of system dynamics into an environment in use for active development in data science, than to attempt to bring each new development in inference and machine learning into the system dynamics enclave.

PySD reads a model file – the product of a modeling program such as Vensim or Stella/iThink – and cross compiles it into Python, providing a simulation engine that can run these models natively in the Python environment. It is not a substitute for these tools, and cannot be used to replace a visual model construction environment.
