"""
Module view2
Translated using PySD version 1.4.1

"""


@cache.step
def another_var():
    """
    Real Name: another var
    Original Eqn: 3*Stock
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return 3 * stock()
