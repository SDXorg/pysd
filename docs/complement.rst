Complementary Projects
======================

The most valuable component for better integrating models with *basically anything else* is a standard language for communicating the structure of those models. That language is `XMILE <http://www.iseesystems.com/community/support/XMILE.aspx>`_. The draft specifications for this have been finalized and the standard should be approved in the next few months.

A Python library for analyzing system dynamics models called the `Exploratory Modeling and Analysis (EMA) Workbench <http://simulation.tbm.tudelft.nl/ema-workbench/contents.html>`_ developed by Erik Pruyt and Jan Kwakkel at TU Delft. This package implements a variety of analysis methods that are unique to dynamic models, and could work very tightly with PySD.

An web-based app called `Simlin <https://simlin.com/>`_ created by Bobby Powers exists as a standalone SD engine. Allows building models and exporting them to XMILE output.

The `Behavior Analysis and Testing Software (BATS) <https://proceedings.systemdynamics.org/2014/proceed/papers/P1211.pdf>`_ developed by Gönenç Yücel includes a really neat method for categorizing behavior modes and exploring parameter space to determine the boundaries between them.

The `SDQC library <https://sdqc.readthedocs.io>`_ developed by Eneko Martin Martinez may be used to check the quality of the data imported by Vensim models from spreadsheet files.

The `excels2vensim library <https://excels2vensim.readthedocs.io>`_, also developed by Eneko Martin Martinez, aims to simplify the incorporation of equations from external data into Vensim.