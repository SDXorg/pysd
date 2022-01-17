---
title: 'PySD: System Dynamics modelling in Python'
tags:
  - Python
  - System Dynamics
  - Vensim
  - Stella
authors:
  - name: Eneko Martín^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-9213-7818
    affiliation: 1
  - name: Roger Samsó^[co-first author]^[corresponding author]
    orcid: 0000-0003-0348-3047
    affiliation: 1
  - name: James Houghton
    orcid: #TODO
    affiliation: 2
  - name: Jordi Solé
    orcid: 0000-0002-2371-1652
    affiliation: "1, 3"
affiliations:
 - name: Centre for Ecological Research and Forestry Applications (CREAF)
   index: 1
 - name: Massachusetts Institute of Technology (MIT)
   index: 2
 - name: University of Barcelona
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

## Summary
**Begin your paper with a summary of the high-level functionality of your software for a non-specialist reader. Avoid jargon in this section.

Include in the paper some sentences that explain the software functionality and domain of use to a non-specialist reader.**

PySD is a CLI-based Python library that allows to transpile (to Python), load, parametrize, modify and execute System Dynamics (SD) models built in Stella&#174; or Vensim&#174;. This tool allows the integration of these models in data science and analytics workflows and to benefit from the plethora of librearies available in the Python ecosystem for data visualization, sensitivity analysis, graph theory and machine learning, among others.

System Dynamics is a mathematical modeling technique used to describe and simulate the dynamics of complex systems over time. The foundations of the methodology were laid in the 1950s by Professor Jay W. Forrester of the Massachusetts Institute of Technology (MIT) [@Forrester:1971]. The building blocks of SD models are stocks, flows, feedback-loops and lookup tables. Stocks represent cumulative quantities which take a certain value at each moment in time (integral); flows are the rates at which those quantities change per unit of time (derivative); feedback-loops are output variables that are routed back as inputs to the model; and lookup tables are single parameter functions that return one value or another from a predefined set of values, depending on the input value. The relationships between these model components are generally represented using causal loop diagrams.

Since its inception, the SD methodology has been applied in different knowledge fields, including energy, climate, population and ecological and economic systems **(#TODO cite)**. In the 1990s, the popularity of the methodology gave rise to the development of several visual programming systems for building SD models. These modeling platforms were widely adopted by the SD community due to their convenient graphical user interfaces (GUIs). Stella&#174; and Vensim&#174; are two of the main exponents of such modeling tools, though many others exist. The models created using these two platforms can be exported into text format (in .xmile and .mdl formats for Stella&#174; and Vensim&#174;, respectively), though most of the functions included in these files are proprietary software.

PySD, was first released in XXXX. The first release included 1) two parsing expression grammars (PEG) to parse .xmile and .mdl files, respectively; 2) Python implementations of a limited number of the Stella&#174; and Vensim&#174; proprietary internal functions; 3) a builder, to transpile the parsed code into Python; and 4) a solver to run the models.

Since then most efforts have been put to:
- make the library more accessible, by adding a CLI and easing the installation process (Anaconda forge)
- attracting developers, by increasing code coverage
- reengineering additional internal Stella&#174; and Vensim&#174; functions
- allowing to import external data (from excel files)
- adding new functionalities not available in Stella&#174; and Vensim&#174; (modularize, run parts of the model indivi)
- fixing bugs


which included 


## Statement of need
**Clearly illustrate the research purpose of the software.**

**Explain the research applications of the software.**

**Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.**

Sister projects include PySD SDEverywere, which is similar to PySD, though it is more focused on publishing the models as web applications, using assembly code and JavaScript.




These are proprietary and closed-source pieces of software.
None of them include tools to make modern visualization and data analitics that are available in other programming environments

The PySD was originally developed by James Houghton, to address all these issues. The translated models are built in pure Python, and the PySD library is used to load them and execute them.

Some additional functionalities make 

This piece of software was used to translate the models developed in the H2020 project MEDEAS, which were put together using Vensim.

Future development: 

## Acknowledgements
**Acknowledgement of any financial support.**
**Acknowledge contributors**

## Features
Import and load external data from Excel files.
Divide the model into modules and submodules.
Isolate and run certain parts of the model individually.
Store intermediate simulation results and resume the simulation from there.

## Citations