'''
Created August 14 2014
James Houghton <james.p.houghton@gmail.com>
    
This module converts a string of SMILE syntax into Python
    
'''
import parsimonious
from parsimonious.nodes import NodeVisitor

# Here we define which python function each XMILE keyword corresponds to
dictionary = {"abs":"abs", "int":"int", "exp":"np.exp", "inf":"np.inf", "log10":"np.log10",
              "pi":"np.pi", "sin":"np.sin", "cos":"np.cos", "sqrt":"np.sqrt", "tan":"np.tan",
              "lognormal":"np.random.lognormal", "normal":"np.random.normal", 
              "poisson":"np.random.poisson", "ln":"np.ln", "exprnd":"np.random.exponential",
              "random":"np.random.rand", "min":"min", "max":"max", "arccos":"np.arccos",
              "arcsin":"np.arcsin", "arctan":"np.arctan",
              "if_then_else":"self.functions.if_then_else",
              "step":"self.functions.step", "pulse":"self.functions.pulse",
              "=":"==", "<=":"<=", "<":"<", ">=":">=", ">":">", "^":"**"}

keywords = ' / '.join(['"%s"'%key for key in reversed(sorted(dictionary.keys(), key=len))])

grammar = (
    'Condition   = Term Conditional*                                                            \n'+
    'Conditional = ("<=" / "<" / ">=" / ">" / "=") Term                                         \n'+

    'Term        = Factor Additive*                                                             \n'+
    'Additive    = ("+"/"-") Factor                                                             \n'+

    'Factor      = ExpBase Multiplicative*                                                      \n'+
    'Multiplicative = ("*" / "/") ExpBase                                                       \n'+

    'ExpBase  = Primary Exponentive*                                                            \n'+
    'Exponentive = "^" Primary                                                                  \n'+

    'Primary  = Call / Parens / Signed / Number / Reference                                     \n'+
    'Parens   = "(" Condition ")"                                                               \n'+
    'Call     = Keyword "(" Condition ("," Condition)* ")"                                      \n'+
    'Signed   = ("-"/"+") Primary                                                               \n'+
    'Number   = ((~"[0-9]"+ "."? ~"[0-9]"*) / ("." ~"[0-9]"+)) (("e"/"E") ("-"/"+") ~"[0-9]"+)? \n'+
    'Reference = Identifier _                                                                   \n'+
    'Identifier = ~"[a-zA-Z]" ~"[a-zA-Z0-9_\$\s]"*                                              \n'+

    '_ = spacechar*                                                                             \n'+
    'spacechar = " "* ~"\t"*                                                                    \n'+

    'Keyword = %s  \n'%keywords
           )


class SMILEParser(NodeVisitor):
    def __init__(self):
        self.grammar = parsimonious.Grammar(grammar)

    def parse(self, text, context='eqn'):
        """
           context : <string> 'eqn', 'defn'
                If context is set to equation, lone identifiers will be parsed as calls to elements
                If context is set to definition, lone identifiers will be cleaned and returned.
        """
        self.ast = self.grammar.parse(text)
        self.context = context
        return self.visit(self.ast)

    def visit_Keyword(self, n, vc):
        return dictionary[n.text]
    
    def visit_Reference(self, n, (Identifier, _)):
        if self.context == 'eqn':
            return 'self.'+Identifier+'()'
        elif self.context == 'defn':
            return Identifier

    def visit_Identifier(self, n, vc):
        #todo: should check here that identifiers are not python keywords...
        string = n.text
        string = string.lower()
        string = string.strip()
        string = string.replace(' ', '_')
        return string

    def visit_Call(self, n, (Translated_Keyword, _1, lparen, _2, args, _3, rparen)):
        return Translated_Keyword+'('+', '.join(args)+')'

    def visit_Conditional(self, n, (condition, _, term)):
        return dictionary[condition] + term

    visit_Exponentive = visit_Conditional
    
    def generic_visit(self, n, vc):
        """
        Replace childbearing nodes with a list of their children;
        for leaves, return the node text;
        for empty nodes, return an empty string.
        """
        return ''.join(filter(None, vc)) or n.text or ''


