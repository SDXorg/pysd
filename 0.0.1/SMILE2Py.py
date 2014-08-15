'''
Created August 14 2014
James Houghton <james.p.houghton@gmail.com>
    
This module converts a string of SMILE syntax into Python
    
'''


import parsimonious
import numpy as np


_grammar = """
    Condition = Term Conditional*
    Conditional = ("<=" / "<" / ">=" / ">" / "=") Term

    Term     = Factor Additive*
    Additive = ("+"/"-") Factor

    Factor   = ExpBase Multiplicative*
    Multiplicative = ("*" / "/") ExpBase

    ExpBase  = Primary Exponentive*
    Exponentive = "^" Primary

    Primary  = Call / Parens / Neg / Number / Identifier
    Parens   = "(" Condition ")"
    Call     = Keyword "(" Condition ("," Condition)* ")"
    Neg      = "-" Primary
    Number   = ((~"[0-9]"+ "."? ~"[0-9]"*) / ("." ~"[0-9]"+)) (("e"/"E") ("-"/"+") ~"[0-9]"+)?
    Identifier = ~"[a-z]" ~"[a-z0-9_\$]"*

    Keyword = "exprnd" / "exp" / "sin" / "cos" / "abs" / "int" / "inf" / "log10" / "pi" /
              "sqrt" / "tan" / "lognormal" / "normal" / "poisson" / "ln" / "min" / "max" /
              "random" / "arccos" / "arcsin" / "arctan" / "if_then_else"
"""
_g = parsimonious.Grammar(_grammar)

# Here we define which python function each XMILE keyword corresponds to
_dictionary = {"abs":"abs", "int":"int", "exp":"np.exp", "inf":"np.inf", "log10":"np.log10",
              "pi":"np.pi", "sin":"np.sin", "cos":"np.cos", "sqrt":"np.sqrt", "tan":"np.tan",
              "lognormal":"np.random.lognormal", "normal":"np.random.normal", 
              "poisson":"np.random.poisson", "ln":"np.ln", "exprnd":"np.random.exponential",
              "random":"np.random.rand", "min":"min", "max":"max", "arccos":"np.arccos",
              "arcsin":"np.arcsin", "arctan":"np.arctan", "if_then_else":"if_then_else"}

def _translate_tree(node):
    if node.expr_name == 'Exponentive': # special case syntax change...
        return '**' + ''.join([_translate_tree(child) for child in node.children[1:]])
    elif node.expr_name == "Keyword":
        return _dictionary[node.text]
    else:
        if node.children:
            return ''.join([_translate_tree(child) for child in node])
        else:
            return node.text

def _get_identifiers(node):
    identifiers = []
    for child in node:
        for item in _get_identifiers(child): #merge all into one list
            identifiers.append(item)
    if node.expr_name in ['Identifier']:
        identifiers.append(node.text)
    return identifiers



def clean(string=''):
    string = string.lower().replace(' ','').replace('\\n','_')
    return string

def translate(s=''):
    # format the string, take out whitespace, etc.
    s = clean(s)
    tree = _g.parse(s)
    return _translate_tree(tree), _get_identifiers(tree)



