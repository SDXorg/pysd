"""vensim2py
    created: August 13, 2014
    last update: June 6 2015
    James Houghton <james.p.houghton@gmail.com>
"""
import parsimonious
import string
from parsimonious.nodes import NodeVisitor

from pysd import builder


###############################################################
#Updates:
#
# - Feb 16, 2015: brought parsing of full sections of the model into the PEG parser.
# - March 28, 2015: incorporated parsimonious's node walker to take care of class construction.
# - June 6, 2015: factored out component class construction code.
###############################################################


###############################################################
#Todo:
#
# - check that the identifiers discovered are not the same as python keywords
# - parse excessive spaces out of docstrings

# - It would be good to include the original vensim/xmile non-python-safed identifiers for
#   model components in the docstring, along with the cleaned version
# - also, include the  original equation, and the python clean version
# - maybe we add elements to the functions which represent the pre and post translation components?
# - model documentation shouldn't include the 'self' parts of a descriptor

# should build in a mechanism to fail gracefully
# maybe want to make the node visitor intelligently add spaces around things
# should check that sin/cos are using the same units in vensim and numpy
# the 'pi' keyword is broken

# are we parsing model control components properly?
###############################################################


######################################################
#Dictionary:
#
# This is the vensim to python translation dictionary. Many expressions (+-/*, etc)
# translate directly, but some need to have their names changed, etc. This is
# how we decide that. If it isn't in the dictionary, it probably won't work
#
######################################################

dictionary = {"ABS":"abs", "INTEGER":"int", "EXP":"np.exp",
    "PI":"np.pi", "SIN":"np.sin", "COS":"np.cos", "SQRT":"np.sqrt", "TAN":"np.tan",
    "LOGNORMAL":"np.random.lognormal", "RANDOM NORMAL":"self.functions.bounded_normal",
    "POISSON":"np.random.poisson", "LN":"np.log", "EXPRND":"np.random.exponential",
    "RANDOM UNIFORM":"np.random.rand", "MIN":"min", "MAX":"max", "ARCCOS":"np.arccos",
    "ARCSIN":"np.arcsin", "ARCTAN":"np.arctan", "IF THEN ELSE":"self.functions.if_then_else",
    "STEP":"self.functions.step", "MODULO":"np.mod", "PULSE":"self.functions.pulse",
    "PULSE TRAIN":"self.functions.pulse_train", "RAMP":"self.functions.ramp",
    "=":"==", "<=":"<=", "<":"<", ">=":">=", ">":">", "^":"**",
    ":AND:": "and", ":OR:":"or", ":NOT:":"not"}

construction_functions = ['DELAY1', 'DELAY3', 'DELAY3I', 'DELAY N', 'DELAY1I',
                          'SMOOTH3I', 'SMOOTH3', 'SMOOTH N', 'SMOOTH', 'SMOOTHI',
                          'INITIAL'] #order is important for peg parser

###############################################################
# General Notes on PEG parser:
#
#  - backslashes in the .mdl file get removed by python (usually?) before they go into parsimonious,
#     as python interprets them as a line continuation character. This is probably ok,
#     as that is how vensim is using them as well, but if the user includes backslashes
#     anywhere, will give them trouble.
#
# - # If you put a *+? quantifier on a node listing (ie: Docstring?) then it creates an anonymous
#     node, which makes it hard to match up later in the tree crawler
#
# - we could think about putting the dictionary and the grammars within the class,
#     because the class uses them exclusively
#
################################################################


# We have to sort keywords in decreasing order of length so that the peg parser doesnt
#    quit early when finding a partial keyword
caps_keywords = dictionary.keys()
multicaps_keywords = list(set(caps_keywords +
                      [string.capwords(word) for word in caps_keywords] +
                      [string.lower(word) for word in caps_keywords]))
keywords = ' / '.join(['"%s"'%word for word in reversed(sorted(multicaps_keywords, key=len))])

multicaps_con_keywords = list(set(construction_functions +
                              [string.capwords(word) for word in construction_functions] +
                              [string.lower(word) for word in construction_functions]))

con_keywords = ' / '.join(['"%s"'%key for key in reversed(sorted(multicaps_con_keywords, key=len))])



file_grammar = (
    # In the 'Model' definiton, we use arbitrary characters (.) to represent the backslashes,
    #    because escaping in here is a nightmare
    'Model = SNL "{UTF-8}" Content ~"...---///" Rubbish*                                        \n'+
    'Content = Entry+                                                                           \n'+
    'Rubbish = ~"."* NL*                                                                        \n'+

    'Entry = MetaEntry / ModelEntry                                                             \n'+

    # meta entry recognizes model documentation, or the partition separating model control params.
    'MetaEntry  = SNL starline SNL Docstring SNL starline _ "~" SNL Docstring SNL "|" SNL*      \n'+
    'ModelEntry = SNL Component SNL "~" _ Unit SNL "~" _ Docstring SNL "|" SNL*                 \n'+

    'Unit      = ~"[^~|]"*                                                                      \n'+
    # The docstring parser also consumes trailing newlines. Not sure if we want this?
    # The second half of the docstring parser is a lookahead to make sure we don't consume starlines
    'Docstring = (~"[^~|]" !~"(\*{3,})")*                                                       \n'+
    'Component = Stock / Flaux / Lookup                                                         \n'+

    'Lookup     = Identifier _ "(" _ NL _ Range _ CopairList _ ")"                              \n'+
    'Range      = "[" _ Copair _ "-" _ Copair _ "]"                                             \n'+
    'CopairList = AddCopair*                                                                    \n'+
    'AddCopair  = "," _ Copair                                                                  \n'+
    # 'Copair' represents a coordinate pair
    'Copair     = "(" _ Primary _ "," _ Primary _ ")"                                           \n'+

    'Stock     = Identifier _ "=" _ "INTEG" _ "(" _ NL _ Condition _ "," _ NL _ Condition _ ")" \n'+
    # 'Flaux' represents either a flow or an auxiliary, as the syntax is the same
    'Flaux     = Identifier _ "=" SNL Condition                                                 \n'+

    # SNL means 'spaced newline'
    'SNL = _ NL* _                                                                              \n'+
    'NL = _ ~"[\\r\\n]" _                                                                       \n'+
    'starline = ~"\*{3,}"                                                                       \n'+

    'Condition   = Term _ Conditional*                                                          \n'+
    'Conditional = ("<=" / "<" / ">=" / ">" / "=") _ Term                                       \n'+

    'Term     = Factor _ Additive*                                                              \n'+
    'Additive = ("+"/"-") _ Factor                                                              \n'+

    # Factor may be consuming newlines? don't want it to...
    'Factor   = ExpBase _ Multiplicative*                                                       \n'+
    'Multiplicative = ("*" / "/") _ ExpBase                                                     \n'+

    'ExpBase  = Primary _ Exponentive*                                                          \n'+
    'Exponentive = "^" _ Primary                                                                \n'+

    'Primary  = Call / ConCall / LUCall / Parens / Signed / Number / Reference                  \n'+
    'Parens   = "(" _ Condition _ ")"                                                           \n'+

    'Call     = Keyword _ "(" _ ArgList _ ")"                                                   \n'+
    'ArgList  = AddArg+                                                                         \n'+
    'AddArg   = ","* _ Condition                                                                \n'+
    # Calls to lookup functions don't use keywords, so we have to use identifiers.
    #    They take only one parameter. This could cause problems.
    'LUCall   = Identifier _ "(" _ Condition _ ")"                                              \n'+
    'Signed   = ("-"/"+") Primary                                                               \n'+
    # We have to break 'reference' out from identifier, because in reference situations
    #    we'll want to add self.<name>(), but we also need to be able to clean and access
    #    the identifier independently. To force parsimonious to handle the reference as
    #    a seperate object from the Identifier, we add a trailing optional space character
    'Reference = Identifier _                                                                   \n'+

    'ConCall  = ConKeyword _ "(" _ ArgList _ ")"                                                \n'+

    'Number   = ((~"[0-9]"+ "."? ~"[0-9]"*) / ("." ~"[0-9]"+)) (("e"/"E") ("-"/"+") ~"[0-9]"+)? \n'+
    'Identifier = Basic_Id / Special_Id                                                         \n'+
    'Basic_Id = Letter (Letter / Digit / ~"[_\s]")*                                             \n'+
    'Special_Id = "\\""  ~"[^\\"]"*  "\\""                                                      \n'+
    'Letter   = ~"[a-zA-Z]"                                                                     \n'+
    'Digit    = ~"[0-9]"                                                                        \n'+

    # We separated out single space characters from the _ entry, so that it can have any number or
    #   combination of space characters.
    '_ = spacechar*                                                                             \n'+
    # Vensim uses a 'backslash, newline' syntax as a way to split an equation onto multiple lines.
    #   We address this by including it as  an allowed space character.
    'spacechar = " "* ~"\t"* (~r"\\\\" NL)*                                                     \n'+

    'Keyword = %s  \n'%keywords +
    'ConKeyword = %s  \n'%con_keywords
    )


#################################################################
# Notes on node visitor
#
# - each function in the parser takes an argument which is the result of having visited
#   each of the children
#
# - we separate the parse function from the class initialization to facilitate unit testing,
#   although in practice, they will most certainly always be called sequentially
#################################################################


class TextParser(NodeVisitor):
    def __init__(self, grammar, infilename):
        self.filename = infilename[:-4]+'.py'
        builder.new_model(self.filename)
        self.grammar = parsimonious.Grammar(grammar)
        self.parse(infilename)

    def parse(self, filename):
        with open(filename, 'rU') as file:
            text = file.read().decode('utf-8')

        self.ast = self.grammar.parse(text)
        return self.visit(self.ast)

    ############# 'entry' level translators ############################
    visit_Entry = visit_Component = NodeVisitor.lift_child

    def visit_ModelEntry(self, n, (_1, Identifier, _2, tld1, _3, Unit, _4,
                                   tld2, _5, Docstring, _6, pipe, _7)):
        """All we have to do here is add to the docstring of the relevant 
        function the things that arent available at lower levels. 
        The stock/flaux/lookup visitors will take care of adding methods to the class
        """
        #string = 'Units: %s \n'%Unit + Docstring
        #builder.add_to_element_docstring(self.component_class, Identifier, string)
        pass


    def visit_Stock(self, n, (Identifier, _1, eq, _2, integ, _3,
                              lparen, _4, NL1, _5, expression, _6,
                              comma, _7, NL2, _8, initial_condition, _9, rparen)):
        builder.add_stock(self.filename, Identifier, expression, initial_condition)
        return Identifier


    def visit_Flaux(self, n, (Identifier, _1, eq, SNL, expression)):
        builder.add_flaux(self.filename, Identifier, expression)
        return Identifier


    def visit_Lookup(self, n, (Identifier, _1, lparen, _2, NL1, _3, Range,
                               _4, CopairList, _5, rparen)):
        builder.add_lookup(self.filename, Identifier, Range, CopairList)
        return Identifier

    def visit_Unit(self, n, vc):
        return n.text.strip()

    visit_Docstring = visit_Unit

    ######### 'expression' level visitors ###############################

    def visit_ConCall(self, n, (ConKeyword, _1, lparen, _2, args, _4, rparen)):
        pass
        if ConKeyword == 'DELAY1': #DELAY3(Inflow, Delay)
            return builder.add_n_delay(self.filename, args[0], args[1], str(0), 1)
        elif ConKeyword == 'DELAY1I':
            pass
        elif ConKeyword == 'DELAY3':
            return builder.add_n_delay(self.filename, args[0], args[1], str(0), 3)
        elif ConKeyword == 'DELAY N':#DELAY N(Inflow, Delay, init, order)
            return builder.add_n_delay(self.filename, args[0], args[1], args[2], args[3])
        elif ConKeyword == 'SMOOTH':
            pass
        elif ConKeyword == 'SMOOTH3':#SMOOTH3(Input,Adjustment Time)
            return builder.add_n_delay(self.filename, args[0], args[1], str(0), 3)
        elif ConKeyword == 'SMOOTH3I': #SMOOTH3I( _in_ , _stime_ , _inival_ )
            return builder.add_n_smooth(self.filename, args[0], args[1], args[2], 3)
        elif ConKeyword == 'SMOOTHI':
            pass
        elif ConKeyword == 'SMOOTH N':
            pass

        elif ConKeyword == 'INITIAL':
            return builder.add_initial(self.filename, args[0])

        #need to return whatever you call to get the final stage in the construction


    def visit_Keyword(self, n, vc):
        return dictionary[n.text.upper()]

    def visit_Reference(self, n, (Identifier, _)):
        return 'self.'+Identifier+'()'

    def visit_Identifier(self, n, vc):
        string = n.text
        return builder.make_python_identifier(string)

    def visit_Call(self, n, (Translated_Keyword, _1, lparen, _2, args, _3, rparen)):
        return Translated_Keyword+'('+', '.join(args)+')'

    def visit_LUCall(self, n, (Identifier, _1, lparen, _2, Condition, _3,  rparen)):
        return 'self.'+Identifier+'('+Condition+')'

    def visit_Copair(self, n, (lparen, _1, xcoord, _2, comma, _3, ycoord, _, rparen)):
        return (float(xcoord), float(ycoord))

    def visit_AddCopair(self, n, (comma, _1, Copair)):
        return Copair

    def visit_ArgList(self, n, args):
        return args

    def visit_AddArg(self, n, (comma, _1, argument)):
        return argument

    def visit_Range(self, n, copairs):
        pass

    def visit_CopairList(self, n, copairs):
        return copairs

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


def doc_supported_vensim_functions():
    """prints a list of all of the vensim functions that are supported
    by the translator.
    """
    rowline     = '+------------------------------+------------------------------+\n'
    headers     = '|           Vensim             |       Python Translation     |\n'
    underline   = '+==============================+==============================+\n'
    string = rowline + headers + underline
    for key, value in dictionary.iteritems():
        string += '|'   + key.center(30) +  '|'  + value.center(30) + '|\n'
        string += rowline
    for item in construction_functions:
        string += '|'   + item.center(30) + '|      Model Construction      |\n'
        string += rowline

    string += '\n `np` corresponds to the numpy package'

    return string


def translate_vensim(mdl_file):
    """
    Translate a vensim model file into a python class.

    Supported functionality:\n\n"""
    parser = TextParser(file_grammar, mdl_file)
    #module = imp.load_source('modulename', parser.filename)

    return parser.filename #module.Components

translate_vensim.__doc__ += doc_supported_vensim_functions()
