PySD
====
*Simulating System Dynamics Models in Python*

|made-with-sphinx-doc|
|DOI|
|PyPI license|
|conda package|
|PyPI package|
|PyPI status|
|PyPI pyversions|

.. |made-with-sphinx-doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
   :target: https://www.sphinx-doc.org/

.. |docs| image:: https://readthedocs.org/projects/pysd/badge/?version=latest
   :target: https://pysd.readthedocs.io/en/latest/?badge=latest

.. |PyPI license| image:: https://img.shields.io/pypi/l/sdqc.svg
   :target: https://github.com/JamesPHoughton/pysd/blob/master/LICENSE

.. |PyPI package| image:: https://badge.fury.io/py/pysd.svg
    :target: https://badge.fury.io/py/pysd

.. |PyPI pyversions| image:: https://img.shields.io/pypi/pyversions/pysd.svg
   :target: https://pypi.python.org/pypi/pysd/

.. |PyPI status| image:: https://img.shields.io/pypi/status/pysd.svg
   :target: https://pypi.python.org/pypi/pysd/

.. |conda package| image:: https://anaconda.org/conda-forge/pysd/badges/version.svg
   :target: https://anaconda.org/conda-forge/pysd

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5654824.svg
   :target: https://doi.org/10.5281/zenodo.5654824

This project is a simple library for running System Dynamics models in Python, with the purpose of improving integration of Big Data and Machine Learning into the SD workflow.

PySD translates :doc:`Vensim <structure/vensim_translation>` or
:doc:`XMILE <structure/xmile_translation>` model files into Python modules,
and provides methods to modify, simulate, and observe those translated models. The translation is done throught an intermediate :doc:`Abstract Synatax Tree representation <structure/structure_index>`,
which makes it possible to add builders in other languages in a simpler way

Why create a new SD simulation engine?
--------------------------------------

There are a number of great SD programs out there (`Vensim <http://vensim.com/>`_, `iThink <http://www.iseesystems.com/Softwares/Business/ithinkSoftware.aspx>`_, `AnyLogic <http://www.anylogic.com/system-dynamics>`_, `Insight Maker <http://insightmaker.com/>`_, and `others <http://en.wikipedia.org/wiki/List_of_system_dynamics_software>`_). In order not to waste our effort, or fall victim to the `Not-Invented-Here <http://en.wikipedia.org/wiki/Not_invented_here>`_ fallacy, we should have a very good reason for starting a new project.

That reason is this: There is a whole world of computational tools being developed in the larger data science community. **System dynamicists should directly use the tools that other people are building, instead of replicating their functionality in SD specific software.** The best way to do this is to bring specific SD functionality to the domain where those other tools are being developed.

This approach allows SD modelers to take advantage of the most recent developments in data science, and focus our efforts on improving the part of the stack that is unique to System Dynamics modeling.

Additional Resources
--------------------

PySD Cookbook
^^^^^^^^^^^^^
A cookbook of simple recipes for advanced data analytics using PySD is available at:
http://pysd-cookbook.readthedocs.org/

The cookbook includes models, sample data, and code in the form of iPython notebooks that demonstrate a variety of data integration and analysis tasks. These models can be executed on your local machine, and modified to suit your particular analysis requirements.


Contributing
^^^^^^^^^^^^
The code for this package is available at: https://github.com/JamesPHoughton/pysd

If you find a bug, or are interested in a particular feature, see :doc:`reporting bugs <../reporting_bugs>`.

If you are interested in contributing to the development of PySD, see the :doc:`developer documentation <../development/development_index>`
listed above.

Citing
^^^^^^
If you use PySD in any published work, consider citing the `PySD Paper (2022) <https://github.com/JamesPHoughton/pysd>`_:

.. code-block:: latex

   Martin-Martinez, E., Samsó, R., Houghton, J. P., & Solé, J. (2022). PySD: System Dynamics Modeling in Python. Journal of Open Source Software. To do

.. code-block:: latex

   @article{Martin-Martinez_PySD_System_Dynamics_2022,
      author = {Martin-Martinez, Eneko and Samsó, Roger and Houghton, James P and Solé, Jordi},
      doi = {TODO},
      journal = {{Journal of Open Source Software}},
      title = {{PySD: System Dynamics Modeling in Python}},
      url = {https://github.com/JamesPHoughton/pysd},
      year = {2022},
      month = {6},
      keywords = {System Dynamics, Vensim, XMILE, Python}
   }

Please, also add the `PySD Introductory Paper (2015) <https://github.com/JamesPHoughton/pysd/blob/master/docs/PySD-Intro-Paper-Preprint.pdf>`_:

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

You can also cite the library using the `DOI provided by Zenodo <https://doi.org/10.5281/zenodo.5654824>`_. It is recomendable to specify the used PySD version and its correspondent DOI. If you want to cite all versions you can use the generic DOI for PySD instead:

|DOI|


Support
^^^^^^^
For additional help or consulting, contact james.p.houghton@gmail.com or eneko.martin.martinez@gmail.com.


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
