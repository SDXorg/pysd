PySD
====
*Simulating System Dynamics Models in Python*

|made-with-sphinx-doc|
|JOSS|
|Maintained|
|PyPI license|
|conda package|
|PyPI package|
|PyPI status|
|PyPI pyversions|
|Contributions|

.. |made-with-sphinx-doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
   :target: https://www.sphinx-doc.org/

.. |Maintained| image:: https://img.shields.io/badge/Maintained-Yes-brightgreen.svg
   :target: https://github.com/SDXorg/pysd/pulse

.. |docs| image:: https://readthedocs.org/projects/pysd/badge/?version=latest
   :target: https://pysd.readthedocs.io/en/latest/?badge=latest

.. |PyPI license| image:: https://img.shields.io/pypi/l/sdqc.svg
   :target: https://github.com/SDXorg/pysd/blob/master/LICENSE

.. |PyPI package| image:: https://badge.fury.io/py/pysd.svg
    :target: https://badge.fury.io/py/pysd

.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/pysd.svg
   :target: https://pypi.python.org/pypi/pysd/

.. |PyPI status| image:: https://img.shields.io/pypi/status/pysd.svg
   :target: https://pypi.python.org/pypi/pysd/

.. |conda package| image:: https://anaconda.org/conda-forge/pysd/badges/version.svg
   :target: https://anaconda.org/conda-forge/pysd

.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.04329/status.svg
   :target: https://doi.org/10.21105/joss.04329

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5654824.svg
   :target: https://doi.org/10.5281/zenodo.5654824

.. |Contributions| image:: https://img.shields.io/badge/contributions-welcome-blue.svg
   :target: https://pysd.readthedocs.io/en/latest/development/development_index.html

This project is a simple library for running System Dynamics models in Python, with the purpose of improving integration of Big Data and Machine Learning into the SD workflow.

PySD translates :doc:`Vensim <structure/vensim_translation>` or
:doc:`XMILE <structure/xmile_translation>` model files into Python modules,
and provides methods to modify, simulate, and observe those translated models. The translation is done through an intermediate :doc:`Abstract Syntax Tree representation <structure/structure_index>`,
which makes it possible to add builders in other languages in a simpler way

Why create a new SD simulation engine?
--------------------------------------

There are a number of great SD programs out there (`Vensim <http://vensim.com/>`_, `iThink <http://www.iseesystems.com/Softwares/Business/ithinkSoftware.aspx>`_, `AnyLogic <http://www.anylogic.com/system-dynamics>`_, `Insight Maker <http://insightmaker.com/>`_, and `others <http://en.wikipedia.org/wiki/List_of_system_dynamics_software>`_). In order not to waste our effort, or fall victim to the `Not-Invented-Here <http://en.wikipedia.org/wiki/Not_invented_here>`_ fallacy, we should have a very good reason for starting a new project.

That reason is this: There is a whole world of computational tools being developed in the larger data science community. **System dynamicists should directly use the tools that other people are building, instead of replicating their functionality in SD specific software.** The best way to do this is to bring specific SD functionality to the domain where those other tools are being developed.

This approach allows SD modelers to take advantage of the most recent developments in data science, and focus our efforts on improving the part of the stack that is unique to System Dynamics modeling.


Limitations
-----------

Currently PySD does not implement all the functions and features of Vensim and XMILE. This may mean that some models cannot be run with the Python translated version, or can only be partially run. In most cases, functions that are not implemented will be translated as :py:func:`pysd.py_backend.functions.not_implemented_function`. However, the most used functions and features are implemented in PySD and most of the models will run properly.

For more information, see the sections on  :ref:`supported Vensim functions <Vensim supported functions>`, :ref:`supported Xmile functions <Xmile supported functions>`, and :ref:`supported Python builder functions <Python supported functions>`. In case you want to add any new functions, please follow the tips in the :doc:`development section <../development/development_index>`. The examples of :doc:`adding functions section <../development/adding_functions>` may help you.

Additional Resources
--------------------

PySD Cookbook
^^^^^^^^^^^^^
A cookbook of simple recipes for advanced data analytics using PySD is available at:
http://pysd-cookbook.readthedocs.org/

The cookbook includes models, sample data, and code in the form of iPython notebooks that demonstrate a variety of data integration and analysis tasks. These models can be executed on your local machine, and modified to suit your particular analysis requirements.


Contributing
^^^^^^^^^^^^
|Contributions|

PySD is currently a community-maintained project, any contribution is welcome.

The code for this package is available at: https://github.com/SDXorg/pysd

If you find any bug, or are interested in a particular feature, see :doc:`reporting bugs <../reporting_bugs>`.

If you are interested in contributing to the development of PySD, see the :doc:`developer documentation <../development/development_index>` listed above.

Join our slack channel in `sd-tools-and-methodology-community <https://slofile.com/slack/sdtoolsandmet-slj3251>`_.

Citing
^^^^^^
If you use PySD in any published work, consider citing the `PySD Paper (2022) <https://doi.org/10.21105/joss.04329>`_:

|JOSS|

.. code-block:: latex

   Martin-Martinez et al., (2022). PySD: System Dynamics Modeling in Python. Journal of Open Source Software, 7(78), 4329, https://doi.org/10.21105/joss.04329

.. code-block:: latex

   @article{Martin-Martinez2022,
      doi = {10.21105/joss.04329},
      url = {https://doi.org/10.21105/joss.04329},
      year = {2022},
      publisher = {The Open Journal},
      volume = {7},
      number = {78},
      pages = {4329},
      author = {Eneko Martin-Martinez and Roger Samsó and James Houghton and Jordi Solé},
      title = {PySD: System Dynamics Modeling in Python},
      journal = {Journal of Open Source Software}
   }

Please, also add the `PySD Introductory Paper (2015) <https://github.com/SDXorg/pysd/blob/master/docs/PySD-Intro-Paper-Preprint.pdf>`_:

.. code-block:: latex

   Houghton, J. P., & Siegel, M. (2015). Advanced data analytics for system dynamics models using PySD. Proceedings of the 33rd International Conference of the System Dynamics Society, 2, 1436–1462. ISBN: 9781510815056

.. code-block:: latex

   @inproceedings{Houghton_PySD_2015,
      author = {Houghton, James P and Siegel, Michael},
      booktitle = {{Proceedings of the 33rd International Conference of the System Dynamics Society}},
      publisher = {{System Dynamics Society}},
      title = {{Advanced data analytics for system dynamics models using PySD}},
      url = {https://www.proceedings.com/28517.html},
      isbn = {9781510815056},
      volume = {2},
      pages = {1436-1462},
      eventdate = {2015-07-19/2015-07-23},
      location = {Cambridge, Massachusetts, USA},
      year = {2015},
      month = {7},
      keywords = {System Dynamics, Vensim, Python}
   }

You can also cite the library using the `DOI provided by Zenodo <https://doi.org/10.5281/zenodo.5654824>`_. It is recommendable to specify the used PySD version and its correspondent DOI. If you want to cite all versions you can use the generic DOI for PySD instead:

|DOI|

Support
^^^^^^^
For additional help or consulting, join our slack channel in `sd-tools-and-methodology-community <https://slofile.com/slack/sdtoolsandmet-slj3251>`_.


.. toctree::
   :hidden:

   installation
   getting_started
   advanced_usage
   command_line_usage
   python_api/python_api_index
   tools
   structure/structure_index
   development/development_index
   reporting_bugs
   whats_new
   about
   complement
