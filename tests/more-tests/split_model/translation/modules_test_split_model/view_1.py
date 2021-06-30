"""
Module view_1
Translated using PySD version 1.4.1

"""


@cache.step
def rate1():
    """
    Real Name: "rate-1"
    Original Eqn: "var-n"
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return varn()


@cache.run
def varn():
    """
    Real Name: "var-n"
    Original Eqn: 5
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 5


@cache.step
def stock():
    """
    Real Name: Stock
    Original Eqn: INTEG ( "rate-1", 1)
    Units:
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_stock()


_integ_stock = Integ(lambda: rate1(), lambda: 1, "_integ_stock")
