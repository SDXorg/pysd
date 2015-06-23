Translation
===========

PySD currently supports basic vensim functionality including:


Types
-----

=============      ============   =======
Vensim             Supported       Notes
=============      ============   =======
Auxiliary          y
Constant           n
Data               n
Initial            n
Level              y
Lookup             y
Reality Check      n
String             n
Subscript          n
Time Base          n
=============      ============   =======

PySD only supports the basic functional types, although many of these can be replaced in a vensim model with functions operating on one of the supported types without loss of effect.
For instance, the type initial can be replaced with an auxiliary type and the function 'INITIAL'


Functions
---------
=============      ============   =======
Vensim             Supported       Notes
=============      ============   =======

Not supported:
tagging variables as 'supplementary'
