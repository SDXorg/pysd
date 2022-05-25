
What's New
==========
v3.0.1 (unreleased)
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
