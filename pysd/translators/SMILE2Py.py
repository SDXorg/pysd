'''
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
              "step":"self.functions.step", "pulse":"self.functions.pulse"
             }

operators = {
              "and": "and", "or": "or", "not": "not",
              "=":"==", "<=":"<=", "<":"<", ">=":">=", ">":">", 
              "^":"**", "+": "+", "-": "-", 
              "*": "*", "/": "/", "mod": "%"       
            }

keywords = ' / '.join(['"%s"'%key for key in reversed(sorted(dictionary.keys(), key=len))])

grammar = (
    'PrimaryLogic = InvertedLogic / Logic                                                       \n'+
    'InvertedLogic = "not" _ Logic                                                              \n'+
    'Logic       = Condition _ Logical*                                                         \n'+
    'Logical     = ("and" / "or") _ Condition                                                   \n'+

    'Condition   = Term _ Conditional*                                                          \n'+
    'Conditional = ("<=" / "<" / ">=" / ">" / "=") _ Term                                       \n'+

    'Term        = Factor _ Additive*                                                           \n'+
    'Additive    = ("+"/"-") _ Factor                                                           \n'+

    'Factor      = ExpBase _ Multiplicative*                                                    \n'+
    'Multiplicative = ("*" / "/" / "mod") _ ExpBase                                             \n'+

    'ExpBase  = Primary _ Exponentive*                                                          \n'+
    'Exponentive = "^" _ Primary                                                                \n'+

    'Primary  = Call / Parens / Signed / Number / Reference                                     \n'+
    'Parens   = "(" _ PrimaryLogic _ ")"                                                        \n'+
    'Call     = Keyword _ "(" _ ArgList _ ")"                                                   \n'+
    'ArgList  = AddArg+                                                                         \n'+
    'AddArg   = ","* _ PrimaryLogic                                                             \n'+
    'Signed   = ("-"/"+") Primary                                                               \n'+
    'Number   = ((~"[0-9]"+ "."? ~"[0-9]"*) / ("." ~"[0-9]"+)) (("e"/"E") ("-"/"+") ~"[0-9]"+)? \n'+
    'Reference = Identifier _                                                                   \n'+
    'Identifier = ~"[a-zA-Z]" ~"[a-zA-Z0-9_\$]"*                                              \n'+

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
        convertedIdentifier = Identifier
        if convertedIdentifier == 'dt':
            convertedIdentifier = 'time_step'
        
        if self.context == 'eqn':
            return 'self.'+convertedIdentifier+'()'
        elif self.context == 'defn':
            return convertedIdentifier

    def visit_Identifier(self, n, vc):
        #todo: should check here that identifiers are not python keywords...
        string = n.text
        string = string.lower()
        string = string.strip()
        string = string.replace(' ', '_')
        return string

    def visit_Call(self, n, (Translated_Keyword, _1, lparen, _2, args, _3, rparen)):
        return Translated_Keyword+'('+', '.join(args)+')'

    def visit_ArgList(self, n, args):
        return args

    def visit_AddArg(self, n, (comma, _1, argument)):
        return argument

    def operationVisitor(self, n, (operator, _, operand)):
        return operators[operator] + ' ' + operand + ' '

    visit_Conditional = operationVisitor
    visit_Exponentive = operationVisitor
    visit_Logical = operationVisitor
    visit_Additive = operationVisitor
    visit_InvertedLogic = operationVisitor
    
    def generic_visit(self, n, vc):
        """
        Replace childbearing nodes with a list of their children;
        for leaves, return the node text;
        for empty nodes, return an empty string.
        """
        return ''.join(filter(None, vc)) or n.text or ''


