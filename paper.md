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
date: 20 January 2022
bibliography: paper.bib

---

## Summary

System Dynamics is a mathematical modeling technique used to describe and simulate the dynamics of complex systems over time. The foundations of the methodology were laid in the 1950s by Professor Jay W. Forrester of the Massachusetts Institute of Technology (MIT) [@Forrester:1971]. The building blocks of SD models are stocks, flows, feedback-loops and lookup tables. Stocks represent cumulative quantities which take a certain value at each moment in time (integral); flows are the rates at which those quantities change per unit of time (derivative); feedback-loops are output variables that are routed back as inputs to the model; and lookup tables are single argument functions that return one value or another from a predefined set, depending on the argument value. The relationships between these model components are frequently represented using causal loop diagrams.

Since its inception, the SD methodology has been applied in different knowledge fields, including energy, climate, population and ecological and economic systems **(#TODO cite)**. In the 1990s, the popularity of the methodology gave rise to the development of several visual programming systems for building SD models. These modeling platforms were widely adopted by the SD community due to their convenient graphical user interfaces (GUIs). Stella&#174; and Vensim&#174; are two of the main commercial exponents of such modeling paradigm, though many others exist **(#TODO cite?)**. 

PySD is a Python library that allows to transpile (to Python), load, parametrize and execute System Dynamics (SD) models built using Stella&#174; or Vensim&#174;. It was first released in September 2014 by James Houghton, and since then there have been 26 releases, being *v2.2.0* the most recent (December 2021) at the time of writing. 

In a nutshell, the main functional elements of the library are 1) two parsing expression grammars (PEG) to parse *.xmile* and *.mdl* files, respectively; 2) reverse engineered implementations of the most frequently used Stella&#174; and Vensim&#174; built-in functions; 3) a builder, to transpile the parsed model code into Python; 4) a fordward Euler solver to run the models; and 5) a command-line interface to manage user interactions with the library.

In addition to the aforementioned core functionality, PySD also allows to: build and run models using PySD alone (without Stella&#174; or Vensim&#174;), though this was not a design premise; import model inputs from external sources (from spreadsheet files); modify model variables at runtime; split the model in any number of modules and submodules (corresponding to Vensim&#174; views); isolate parts of the model and run them individually; and store intermediate simulation results and resume the simulation from that particular state.

## Statement of need
**Clearly illustrate the research purpose of the software.**

**Explain the research applications of the software.**

**Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.**

Visual programming systems are excellent tools for model conceptualization and design using casual loop diagrams. Further than that, the Python ecosystem offers comparable (even better in some cases) alternatives for data visualization, sensitivity analysis, graph theory, machine learning and other data analitics tools. Some other inconveniences of these platforms at current time include the fact that they are not available in all of the most widely used OS platforms and they lack good support for Version Control Systems (VCS). Most importantly, though models created using the most widely used SD platforms can be exported into text format (in *.xmile* and *.mdl* formats, for Stella&#174; and Vensim&#174;, respectively), the implementation of the built-in functions included in these files is not disclosed.

Though the actual use-cases in which PySD is applied by the community goes beyond the knowledge of the authors, the latest developments of the library have taken place in the context of the European H2020 projects MEDEAS **(#CITE paper pymedeas)** and LOCOMOTION **(#CITE locomotion website?)**. The MEDEAS project ended in late 2019, and aimed at developing an Integrated Assessment Model (IAM) to evaluate energy transition scenarios, taking into account societal dimensions (energy, economy, environment) and biophysical constraints (resource availability, climate). The LOCOMOTION project, which is still ongoing (2019-2023), aims to continue the development of the model as it was at the end of the MEDEAS project.


Sister projects include PySD SDEverywere, which is similar to PySD, though it is more focused on publishing the models as web applications, using assembly code and JavaScript.

## Future development
Future development can be classified in three axes: bug fixing, reverse engineering more Stella&#174; and Vensim&#174; built-in functions, adding new features.


## Acknowledgements

**Acknowledgement of any financial support.**
**Acknowledge contributors**


## Citations