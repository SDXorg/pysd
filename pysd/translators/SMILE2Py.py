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
              "arcsin":"np.arcsin", "arctan":"np.arctan", "if_then_else":"functions.if_then_else",
              "step":"functions.step", "pulse":"functions.pulse",
              "=":"==", "<=":"<=", "<":"<", ">=":">=", ">":">", "^":"**"}

def _translate_tree(node):
    """
    Here we navigate the Abstract syntax tree and swap out any commands which are 
    not direct python commands
    
    We could probably combine the terminal and non-terminal lookups, as the terminals would have 
    nothing to loop through, but the lookup for the dictionary would be more complex
    """
    if node.expr_name in ['Exponentive', 'Conditional']: #non-terminal lookup
        return _dictionary[node.children[0].text] + ''.join([_translate_tree(child) for child in node.children[1:]])
    elif node.expr_name == "Keyword": #terminal lookup
        return _dictionary[node.text]
    else:
        if node.children:
            return ''.join([_translate_tree(child) for child in node])
        else:
            return node.text

def _parse(string=''):
    """
    We don't really need this to be a separate function, because its really 
    only one line of code - but it makes it easier to test just the parsing
    part of the script
    """
    return _g.parse(string)

def _get_identifiers(node):
    """
    Its helpful to know what the identifiers are in a string of XMILE syntax, 
    as we can then build up a dependancy graph based upon references to other
    elements. 
    
    In this function we crawl through the Abstract Syntax Tree recursively
    and when we find a node that is an identifier, we add it to our list.
    """
    identifiers = []
    for child in node:
        identifiers += _get_identifiers(child) #merge all into one list
    identifiers += [node.text] if node.expr_name in ['Identifier'] else []
    return identifiers


def clean(string=''):
    """
    format the string, take out whitespace, etc.
    sometime in the future, it would be nice if our parsing 
    grammar could handle the more complicated cases, and 
    we wouldnt have to use regex to tidy up before hand.
    """
    string = string.lower().replace(' ','').replace('\\n','_').replace('\n','_')
    return string


def translate(s=''):
    """
    given a string, clean it, parse it, translate it into python,
    and return a line of executable python code, and a list of
    variables which the code depends upon.
    """
    s = clean(s)
    tree = _g.parse(s)
    return _translate_tree(tree), _get_identifiers(tree)



