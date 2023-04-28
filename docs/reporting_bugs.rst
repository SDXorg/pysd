Reporting bugs
==============

Before reporting any bug, please make sure that you are using the latest version of PySD. You can get the version number by running `python -m pysd -v` on the command line.

All bugs must be reported in the project's `issue tracker on github <https://github.com/SDXorg/pysd/issues>`_.

.. note::
  Not all the features and functions are implemented. If you are in trouble while translating or running a Vensim or Xmile model check the :ref:`Vensim supported functions <Vensim supported functions>` or :ref:`Xmile supported functions <Xmile supported functions>` and consider that when opening a new issue.

Bugs during translation
-----------------------
1. Check the line where it happened and try to identify if it is due to a missing function or feature or for any other reason.
2. See if there is any open issue with the same or a similar bug. If there is, you can add your specific problem there.
3. If not previously reported, open a new issue. Try to use a descriptive title such us `Missing subscripts support for Xmile models`, avoid titles like `Error while parsing Xmile model`. Provide the given error information and, if possible, a small model reproducing the same error.

Bugs during runtime
-------------------
1. Check if a similar bug has been reported on the issue tracker. If that is not the case, open a new issue with a descriptive title.
2. Provide the error information and all the relevant lines you used to execute the model.
3. If possible, provide a small model reproducing the bug.



