"""
Module view_3
Translated using PySD version 1.4.1

"""


@cache.step
def variablex():
    """
    Real Name: "variable-x"
    Original Eqn: 6*another var
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return 6 * another_var()
