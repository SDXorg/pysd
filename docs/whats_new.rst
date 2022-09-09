What's New
==========
v3.7.0 (to be released)
-------------------

New Features
~~~~~~~~~~~~
- Simulation results can now be stored as netCDF4 files (:issue:`355`). (`@rogersamso <https://github.com/rogersamso>`_)
- The CLI also accepts netCDF4 file paths after the -o argument. (`@rogersamso <https://github.com/rogersamso>`_)

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix bug when a WITH LOOKUPS argument has subscripts. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)

Documentation
~~~~~~~~~~~~~
- Adds Storing simulation results on a file section in the getting started page. (`@rogersamso <https://github.com/rogersamso>`_)

Performance
~~~~~~~~~~~
- Exporting outputs as netCDF4 is much faster than exporting a pandas DataFrame, especially for large models. (`@rogersamso <https://github.com/rogersamso>`_)

Internal Changes
~~~~~~~~~~~~~~~~
- Make PySD work with :py:mod:`parsimonius` 0.10.0. (`@enekomartinmartinez <https://github.com/enekomartinmartinez>`_)
- Add netCDF4 and hdf5 dependencies. (`@rogersamso <https://github.com/rogersamso>`_)


v3.6.1 (2022/09/05)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Set :py:mod:`parsimonius` requirement to 0.9.0 to avoid a breaking-change in the newest version. Pending to update PySD to run it with :py:mod:`parsimonious` 0.10.0.

v3.6.0 (2022/08/31)
-------------------

New Features
~~~~~~~~~~~~
- Include warning messages when a variable is defined in more than one view, when a control variable appears in a view or when a variable doesn't appear in any view as a `workbench variable` (:issue:`357`).
- Force variables in a module to be saved alphabetically for being able to compare differences between versions (only for the models that are split by views).

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Classify control variables in the main file always (:issue:`357`).

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Include :py:class:`pysd.translators.structures.abstract_model.AbstractControlElement` child of :py:class:`pysd.translators.structures.abstract_model.AbstractElement` to differentiate the control variables.


v3.5.2 (2022/08/15)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Make sketch's `font_size` optional.

Documentation
~~~~~~~~~~~~~
- Correct typos.

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

v3.5.1 (2022/08/11)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix bug generated when :EXCEPT: keyword is used with subscript subranges (:issue:`352`).
- Fix bug of precision error for :py:func:`pysd.py_backend.allocation.allocate_by_priority` (:issue:`353`).
- Fix bug of constant cache assignment.

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~
- Improve the performance of reading :py:class:`pysd.py_backend.external.External` data with cellrange names by loading the data in memory with :py:mod:`pandas`. As recommended by :py:mod:`openpyxl` developers, this is a possible way of improving performance to avoid parsing all rows up each time for getting the data (`issue 1867 in openpyxl <https://foss.heptapod.net/openpyxl/openpyxl/-/issues/1867>`_).

Internal Changes
~~~~~~~~~~~~~~~~

v3.5.0 (2022/07/25)
-------------------

New Features
~~~~~~~~~~~~
- Add support for subscripted arguments in :py:func:`pysd.py_backend.functions.ramp` and :py:func:`pysd.py_backend.functions.step` (:issue:`344`).

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix bug related to the order of elements in 1D GET expressions (:issue:`343`).
- Fix bug in request 0 values in allocate by priority (:issue:`345`).
- Fix a numerical error in starting time of step and ramp.

Documentation
~~~~~~~~~~~~~
- Include new PySD logo.

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Ignore 'distutils Version classes are deprecated. Use packaging.version instead' error in tests as it is an internal error of `xarray`.
- Add a warning message when a subscript range is duplicated in a variable reference.


v3.4.0 (2022/06/29)
-------------------

New Features
~~~~~~~~~~~~
- Add support for Vensim's `ALLOCATE AVAILABLE <https://www.vensim.com/documentation/fn_allocate_available.html>`_ (:py:func:`pysd.py_backend.allocation.allocate_available`) function (:issue:`339`). Integer allocation cases have not been implemented neither the fixed quantity and constant elasticity curve priority functions.

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~
- Improve the documentation of the :py:mod:`pysd.py_backend.allocation` module.

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Add a class to manage priority profiles so it can be also used by the `many-to-many allocation <https://www.vensim.com/documentation/24340.html>`_.


v3.3.0 (2022/06/22)
-------------------

New Features
~~~~~~~~~~~~
- Add support for Vensim's `ALLOCATE BY PRIORITY <https://www.vensim.com/documentation/fn_allocate_by_priority.html>`_ (:py:func:`pysd.py_backend.allocation.allocate_by_priority`) function (:issue:`263`).

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~
- Fix bug of using subranges to define a bigger range (:issue:`335`).

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Improve error messages for :class:`pysd.py_backend.External` objects.

v3.2.0 (2022/06/10)
-------------------

New Features
~~~~~~~~~~~~
- Add support for Vensim's `GET TIME VALUE <https://www.vensim.com/documentation/fn_get_time_value.html>`_ (:py:func:`pysd.py_backend.functions.get_time_value`) function (:issue:`332`). Not all cases have been implemented.
- Add support for Vensim's `VECTOR SELECT <http://vensim.com/documentation/fn_vector_select.html>`_ (:py:func:`pysd.py_backend.functions.vector_select`) function (:issue:`266`).

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~



v3.1.0 (2022/06/02)
-------------------

New Features
~~~~~~~~~~~~
- Add support for Vensim's `VECTOR SORT ORDER <https://www.vensim.com/documentation/fn_vector_sort_order.html>`_ (:py:func:`pysd.py_backend.functions.vector_sort_order`) function (:issue:`326`).
- Add support for Vensim's `VECTOR RANK <https://www.vensim.com/documentation/fn_vector_rank.html>`_ (:py:func:`pysd.py_backend.functions.vector_rank`) function (:issue:`326`).
- Add support for Vensim's `VECTOR REORDER <https://www.vensim.com/documentation/fn_vector_reorder.html>`_ (:py:func:`pysd.py_backend.functions.vector_reorder`) function (:issue:`326`).

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~
- Add the section :doc:`/development/adding_functions` with examples for developers.

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

- Include a template for PR.


v3.0.1 (2022/05/26)
-------------------

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

- Simplify subscripts dictionaries for :py:class:`pysd.py_backend.data.TabData` objects.

Documentation
~~~~~~~~~~~~~
- Improve tests/README.md.
- Minor improvements in the documentation.

Performance
~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
- Add Python 3.10 to CI pipeline and include it in the supported versions list.
- Correct LICENSE file extension in the `setup.py`.
- Move from `importlib`'s :py:func:`load_module` to :py:func:`exec_module`.
- Remove warnings related to :py:data:`set` usage.
- Move all the missing test to :py:mod:`pytest`.
- Remove warning messages from test and make test fail if there is any warning.


v3.0.0 (2022/05/23)
-------------------

New Features
~~~~~~~~~~~~

- The new :doc:`Abstract Model Representation <structure/structure_index>` translation and building workflow will allow to add new output languages in the future.
- Added new properties to the :py:class:`pysd.py_backend.model.Macro` to make more accessible some information: :py:attr:`.namespace`, :py:attr:`.subscripts`, :py:attr:`.dependencies`, :py:attr:`.modules`, :py:attr:`.doc`.
- Cleaner Python models:
    - :py:data:`_namespace` and :py:data:`_dependencies` dictionaries have been removed from the file.
    - Variables original names, dependencies metadata now are given through :py:meth:`pysd.py_backend.components.Component.add` decorator, instead of having them in the docstring.
    - Merging of variable equations is now done using the coordinates to a pre-allocated array, instead of using the `magic` function :py:data:`pysd.py_backend.utils.xrmerge()`.
    - Arranging and subseting arrays are now done inplace instead of using the magic function :py:data:`pysd.py_backend.utils.rearrange()`.

Breaking changes
~~~~~~~~~~~~~~~~

- Set the argument :py:data:`flatten_output` from :py:meth:`.run` to :py:data:`True` by default. Previously it was set to :py:data:`False` by default.
- Move the docstring of the model to a property, :py:attr:`.doc`. Thus, it is not callable anymore.
- Allow the function :py:func:`pysd.py_backend.functions.pulse` to also perform the operations performed by :py:data:`pysd.py_backend.functions.pulse_train()` and :py:data:`pysd.py_backend.functions.pulse_magnitude()`.
- Change first argument of :py:func:`pysd.py_backend.functions.active_initial`, now it is the `stage of the model` and not the `time`.
- Simplify the function :py:data:`pysd.py_backend.utils.rearrange()` orienting it to perform simple rearrange cases for user interaction.
- Move :py:data:`pysd.py_backend.statefuls.Model` and  :py:data:`pysd.py_backend.statefuls.Macro` to  :py:class:`pysd.py_backend.model.Model` and :py:class:`pysd.py_backend.model.Macro`, respectively.
- Manage all kinds of lookups with the :py:class:`pysd.py_backend.lookups.Lookups` class.
- Include a second optional argument to lookups functions to set the final coordinates when a subscripted variable is passed as an argument.

Deprecations
~~~~~~~~~~~~

- Remove :py:data:`pysd.py_backend.utils.xrmerge()`, :py:data:`pysd.py_backend.functions.pulse_train()`, :py:data:`pysd.py_backend.functions.pulse_magnitude()`, :py:data:`pysd.py_backend.functions.lookup()`, :py:data:`pysd.py_backend.functions.lookup_discrete()`, :py:data:`pysd.py_backend.functions.lookup_extrapolation()`, :py:data:`pysd.py_backend.functions.logical_and()`, :py:data:`pysd.py_backend.functions.logical_or()`, :py:data:`pysd.py_backend.functions.bounded_normal()`, :py:data:`pysd.py_backend.functions.log()`.
- Remove old translation and building files (:py:data:`pysd.translation`).


Bug fixes
~~~~~~~~~

- Generate the documentation of the model when loading it to avoid lossing information when replacing a variable value (:issue:`310`, :pull:`312`).
- Make random functions return arrays of the same shape as the variable, to avoid repeating values over a dimension (:issue:`309`, :pull:`312`).
- Fix bug when Vensim's :MACRO: definition is not at the top of the model file (:issue:`306`, :pull:`312`).
- Make builder identify the subscripts using a main range and subrange to allow using subscripts as numeric values as Vensim does (:issue:`296`, :issue:`301`, :pull:`312`).
- Fix bug of missmatching of functions and lookups names (:issue:`116`, :pull:`312`).
- Parse Xmile models case insensitively and ignoring the new lines characters (:issue:`203`, :issue:`253`, :pull:`312`).
- Add support for Vensim's `\:EXCEPT\: keyword <https://www.vensim.com/documentation/exceptionequations.html>`_ (:issue:`168`, :issue:`253`, :pull:`312`).
- Add spport for Xmile's FORCST and SAFEDIV functions (:issue:`154`, :pull:`312`).
- Add subscripts support for Xmile (:issue:`289`, :pull:`312`).
- Fix numeric error bug when using :py:data:`return_timestamps` and time step with non-integer values.

Documentation
~~~~~~~~~~~~~

- Review the whole documentation, refract it, and describe the new features.

Performance
~~~~~~~~~~~

- The variables defined in several equations are now assigned to a pre-allocated array instead of using :py:data:`pysd.py_backend.utils.xrmerge()`.
- The arranging and subseting of arrays is now done inplace instead of using the magic function :py:data:`pysd.py_backend.utils.rearrange()`.
- The grammars for Parsimonious are only compiled once per translation.

Internal Changes
~~~~~~~~~~~~~~~~
- The translation and the building of models has been totally modified to use the :doc:`Abstract Model Representation <structure/structure_index>`.
