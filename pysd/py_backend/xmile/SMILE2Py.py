"""
Created August 14 2014
James Houghton <james.p.houghton@gmail.com>

Changed May 03 2017
Alexey Prey Mulyukin <alexprey@yandex.ru>
    Changes:
    
    [May 03 2017] Alexey Prey Mulyukin: Integrate support to
        logical operators like 'AND', 'OR' and 'NOT'.
        Fix support the whitespaces in expressions between 
        operators and operands.
        Add support to modulo operator - 'MOD'.
        Fix support for case insensitive in function names.
    
This module converts a string of SMILE syntax into Python
    
"""
import parsimonious
from parsimonious.nodes import NodeVisitor
import pkg_resources
import re

# Here we define which python function each XMILE keyword corresponds to
functions = {
    # ===
    # 3.5.1 Mathematical Functions
    # http://docs.oasis-open.org/xmile/xmile/v1.0/csprd01/xmile-v1.0-csprd01.html#_Toc398039980
    # ===
    
    "abs": "abs", 
    "arccos": "np.arccos",
    "arcsin": "np.arcsin", 
    "arctan": "np.arctan",
    "cos": "np.cos", 
    "sin": "np.sin", 
    "tan": "np.tan",
    "exp": "np.exp", 
    "inf": "np.inf", 
    "int": "int",
    "ln": "np.log", 
    "log10": "np.log10",
    "max": "max", 
    "min": "min",
    "pi": "np.pi",
    "sqrt": "np.sqrt", 
    
    # ===
    # 3.5.2 Statistical Functions
    # http://docs.oasis-open.org/xmile/xmile/v1.0/csprd01/xmile-v1.0-csprd01.html#_Toc398039981
    # ===
    
    "exprnd": "np.random.exponential",
    "lognormal": "np.random.lognormal", 
    "normal": "np.random.normal",
    "poisson": "np.random.poisson", 
    "random": "np.random.rand", 
    
    # ====
    # 3.5.3 Delay Functions
    # http://docs.oasis-open.org/xmile/xmile/v1.0/csprd01/xmile-v1.0-csprd01.html#_Toc398039982
    # ====
    
    # "delay" !TODO!
    # "delay1" !TODO!
    # "delay2" !TODO!
    # "delay3" !TODO!
    # "delayn" !TODO!
    # "forcst" !TODO!
    # "smth1" !TODO!
    # "smth3" !TODO!
    # "smthn" !TODO!
    # "trend" !TODO!
    
    # ===
    # 3.5.4 Test Input Functions
    # http://docs.oasis-open.org/xmile/xmile/v1.0/csprd01/xmile-v1.0-csprd01.html#_Toc398039983
    # ===

    "pulse": "functions.pulse_magnitude",
    "step": "functions.step",
    "ramp" "functions.ramp",
    
    # ===
    # 3.5.5 Time Functions
    # http://docs.oasis-open.org/xmile/xmile/v1.0/csprd01/xmile-v1.0-csprd01.html#_Toc398039984
    # ===
    # Should we include as function list or it provided by another way?
    
    # "dt" !TODO!
    # "starttime" !TODO!
    # "stoptime" !TODO!
    # "time" !TODO!
    
    # ===
    # 3.5.6 Miscellaneous Functions
    # http://docs.oasis-open.org/xmile/xmile/v1.0/csprd01/xmile-v1.0-csprd01.html#_Toc398039985
    # ===
    "if_then_else": "functions.if_then_else",
    # "init" !TODO!
    # "previous" !TODO!
    # "self" !TODO!
}

prefix_operators = {
    "not": " not ",
    "-": "-", 
    "+": " ",
}

infix_operators = {
    "and": " and ", 
    "or": " or ",
    "=": "==", 
    "<=": "<=", 
    "<": "<", 
    ">=": ">=", 
    ">": ">", 
    "<>": "!=",
    "^": "**", 
    "+": "+", 
    "-": "-",
    "*": "*", 
    "/": "/", 
    "mod": "%",
}

builders = {}


def format_word_list(word_list):
    return '|'.join(
        [re.escape(k) for k in reversed(sorted(word_list, key=len))])


class SMILEParser(NodeVisitor):
    def __init__(self, model_namespace):
        self.model_namespace = model_namespace

        self.extended_model_namespace = {
            key.replace(' ', '_'): value for key, value in self.model_namespace.items()}
        self.extended_model_namespace.update(self.model_namespace)
        self.extended_model_namespace.update({'dt': 'time_step'})

        grammar = pkg_resources.resource_string("pysd", "py_backend/xmile/smile.grammar")
        grammar = grammar.decode('ascii').format(
            funcs=format_word_list(functions.keys()),
            in_ops=format_word_list(infix_operators.keys()),
            pre_ops=format_word_list(prefix_operators.keys()),
            identifiers=format_word_list(self.extended_model_namespace.keys()),
            build_keywords=format_word_list(builders.keys())
        )

        self.grammar = parsimonious.Grammar(grammar)

    def parse(self, text, context='eqn'):
        """
           context : <string> 'eqn', 'defn'
                If context is set to equation, lone identifiers will be parsed as calls to elements
                If context is set to definition, lone identifiers will be cleaned and returned.
        """
        # !TODO! Should remove the inline comments from `text` before parsing the grammar
        # http://docs.oasis-open.org/xmile/xmile/v1.0/csprd01/xmile-v1.0-csprd01.html#_Toc398039973
        self.ast = self.grammar.parse(text)
        self.context = context
        return self.visit(self.ast)

    def visit_identifier(self, n, vc):
        return self.extended_model_namespace[n.text] + '()'

    def visit_func(self, n, vc):
        return functions[n.text.lower()]

    def visit_pre_oper(self, n, vc):
        return prefix_operators[n.text.lower()]

    def visit_in_oper(self, n, vc):
        return infix_operators[n.text.lower()]

    def generic_visit(self, n, vc):
        """
        Replace childbearing nodes with a list of their children;
        for leaves, return the node text;
        for empty nodes, return an empty string.

        Handles:
        - call
        - parens
        -
        """
        return ''.join(filter(None, vc)) or n.text or ''
