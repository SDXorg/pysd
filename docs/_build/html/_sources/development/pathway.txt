PySD Development Pathway
========================

The basic use case of PySD is for a user to import a fully developed SD model (created in Vensim, or Stella/iThink, etc.) into python; and then use third party tools to perform statistical analysis and inference, and to interface with external data.

Current Features
----------------

* Basic XMILE and Vensim parser,
* Established library structure and data formats
* Simulation using existing python integration tools
* Basic demonstrations of integration with python Data Science functionality
* Run-at-a-time parameter modification
* Step-at-a-time parameter modification / time-variant exogenous inputs
* Extended backends for storing parameters and output values
* Demonstration of integration with Machine Learning/Monte Carlo/Statistical Methods
* Python methods for programmatically manipulating SD model structure
* Turn off and on 'traces' or records of the values of variables

Possible Future features
------------------------

* Complete parsers (including subscripting, etc)
* Embed SD.js front end in iPython widget
* XMILE display component parser
* Customizations of external Data Science tools for dynamic systems
* Incorporation of analysis tools specific to dynamic systems
* Additional SD tools: checks for model units, value limits, etc.
* Python methods for saving XMILE models
* Hover over stock/flow elements to get things like units, descriptions, values, etc.
* Output DataFrame including tags for units


