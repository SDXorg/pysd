"""
vensim2py.py

This file contains the machinery for parsing a vensim file and formatting
components for the model builder.

Everything that touches vensim or is vensim specific should live here.

Updates
-------
- August 13, 2014: Created
- Feb 16, 2015: brought parsing of full sections of the model into the PEG parser.
- March 28, 2015: incorporated parsimonious's node walker to take care of class construction.
- June 6, 2015: factored out component class construction code.
- Jan 2016: reworked to include subscript components

Contributors
------------
- @jamesphoughton
- @mhy05


General Notes
-------------
- backslashes in the .mdl file get removed by python (usually?) before they go into parsimonious,
as python interprets them as a line continuation character. This is probably ok,
as that is how vensim is using them as well, but if the user includes backslashes
anywhere, will give them trouble.

- If you put a *+? quantifier on a node listing (ie: Docstring?) then it creates an anonymous
node, which makes it hard to match up later in the tree crawler

"""
# Todo: Include original vensim/xmile non-python-safe identifiers in element docstrings
# Todo: Consider using original vensim/xmile ids for dataframe column headers
# Todo: Improve error messaging for failed imports

# Todo: Redo the grammar
# Todo: Deal with lists properly - e.g. argument lists, ists that encode arrays, etc.
#  Poor list handling currently causes failure of some function calls (delayNI)
# Todo: Add proper preprocessor to pull out annoyances (\n,\t,\\,\\\sketch///,etc). Keep spaces.
# Todo: Put the grammar in its own file (begun at `vensim.grammar`) and import it. i.e.
#   from string import Template
#   import pkgutil
#   grammar_template = Template(pkgutil.get_data('pysd', 'translators/vensim.grammar'))
#   file_grammar = grammar_template.substitute(keywords=keywords, con_keywords=con_keywords)

# Todo: Process each pipe-delimited (|) section on its own
# Todo: Process each tilde-delimited (~) section on its own
# Todo: construct python model sections from multiple vensim model elements
# Todo: make 'stock' a construction keyword like 'delay' - they are equivalent
# Todo: when a parse error is detected, only print the first 5 lines of the parse tree

import parsimonious
from parsimonious.nodes import NodeVisitor
import string
import re
from pysd import builder


# This is the vensim to python translation dictionary. Many expressions (+-/*, etc)
# translate directly, but some need to have their names changed, etc. This is
# how we decide that. If it isn't in the dictionary, it probably won't work

dictionary = {"ABS":"abs", "INTEGER":"int", "EXP":"np.exp",
    "PI":"np.pi", "SIN":"np.sin", "COS":"np.cos", "SQRT":"np.sqrt", "TAN":"np.tan",
    "LOGNORMAL":"np.random.lognormal", "RANDOM NORMAL":"functions.bounded_normal",
    "POISSON":"np.random.poisson", "LN":"np.log", "EXPRND":"np.random.exponential",
    "RANDOM UNIFORM":"np.random.rand", "MIN":"np.minimum", "MAX":"np.maximum",
    "SUM":"np.sum", "ARCCOS":"np.arccos",
    "ARCSIN":"np.arcsin", "ARCTAN":"np.arctan", "IF THEN ELSE":"functions.if_then_else",
    "STEP":"functions.step", "MODULO":"np.mod", "PULSE":"functions.pulse",
    "PULSE TRAIN":"functions.pulse_train", "RAMP":"functions.ramp",
    "=":"==", "<=":"<=", "<>":"!=", "<":"<", ">=":">=", ">":">", "^":"**",
    "POS":"functions.pos", "TUNER":"functions.tuner", "TUNE1":"functions.tuner"}

construction_functions = ['DELAY1', 'DELAY3', 'DELAY3I', 'DELAY N', 'DELAY1I',
                          'SMOOTH3I', 'SMOOTH3', 'SMOOTH N', 'SMOOTH', 'SMOOTHI',
                          'INITIAL'] #order is important for peg parser


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
    # In the 'Model' definition, we use arbitrary characters (.) to represent the backslashes,
    #    because escaping in here is a nightmare
    #Subscripting definition trial
    'Model = SNL "{UTF-8}" Content ~"...---///" Rubbish*                                        \n'+
    'Content = Entry+                                                                           \n'+
    'Rubbish = ~"."* NL*                                                                        \n'+
    'Entry = MacroEntry / MetaEntry / ModelEntry / GroupEntry                                   \n'+
    'MacroEntry = SNL ":MACRO:" SNL MacroInside SNL ":END OF MACRO:" SNL                        \n'+
    'MacroInside = ~"[^:]"* SNL                                                                 \n'+
    # meta entry recognizes model documentation, or the partition separating model control params.
    'MetaEntry  = SNL starline SNL Docstring SNL starline SNL "~" SNL Docstring SNL "|" SNL*    \n'+
    'ModelEntry = SNL Component SNL "~" SNL Unit SNL "~" SNL Docstring SNL "|" SNL*             \n'+
    'GroupEntry = SNL "{" SNL ~"[^}]"* SNL "}" SNL \n'+
    'Unit      = ~"[^~|]"*                                                                      \n'+
    # The docstring parser also consumes trailing newlines. Not sure if we want this?
    # The second half of the docstring parser is a lookahead to make sure we don't consume starlines
    'Docstring = (~"[^~|]" !~"(\*{3,})")*                                                       \n'+
    'Component = Stock / Flows / Lookup / Subscript                       \n'+
    # ##################################Subscript Element##########################
    'Subscript = Identifier SNL ":" SNL SubElem                                           \n'+

    #We need to weed out subscripts from non-subscripts

    'Lookup     = Identifier SNL "(" SNL Range SNL CopairList SNL ")"                           \n'+
    #'Lookup     = Identifier _ "(" SNL Range _ CopairList _ ")"            \n'+

    # Subscript Element#############################################
    'Subtext = SNL "[" SNL SubElem SNL "]" SNL                                               \n'+

    'Range      = "[" _ Copair _ "-" _ Copair _ "]"                                             \n'+
    'CopairList = AddCopair*                                                                    \n'+
    'AddCopair  = SNL "," SNL Copair                                                            \n'+
    # 'Copair' represents a coordinate pair
    'Copair     = "(" SNL UnderSub SNL ")"                                           \n'+
    #'Copair     = "(" _ Primary _ "," _ Primary _ ")"                                \n'+

    'Stock = Identifier _ Subtext* _ "=" _ "INTEG" _ "(" SNL Condition _ "," SNL Condition _ ")"\n'+
    # 'Flaux' represents either a flow or an auxiliary, as the syntax is the same
    'Flows = Flowint Flaux \n' + # this is for subscripted equations, or situations where each element of the subscript is defined by itself.
    'Flowint = Flow*'
    'Flow = (Flaux SNL "~~|" SNL)                                                               \n'+

    'Flaux = Identifier SNL Subtext* SNL "=" SNL Condition                                      \n'+

    # SNL means 'spaced newline'
    'SNL = _ NL* _                                                                              \n'+
    'NL = _ ~"[\\r\\n]" _                                                                       \n'+
    'starline = ~"\*{3,}"                                                                       \n'+

    'Condition   = SNL Term SNL Conditional*                                                    \n'+
    'Conditional = ("<=" / "<>" / "<" / ">=" / ">" / "=") SNL Term                              \n'+

    'SubElem = Identifier (SNL "," SNL Identifier SNL)*                                         \n'+

    'Term     = Factor SNL Additive*                                                            \n'+
    'Additive = ("+"/"-") SNL Factor                                                            \n'+

    # Factor may be consuming newlines? don't want it to...
    'Factor   = ExpBase SNL Multiplicative*                                                     \n'+
    'Multiplicative = ("*" / "/") SNL ExpBase                                                   \n'+

    'ExpBase  = Primary SNL Exponentive*                                                        \n'+
    'Exponentive = "^" SNL Primary                                                              \n'+

    'Primary  = Call / ConCall / LUCall / Parens / Signed / Subs / UnderSub / Number / Reference\n'+
    'Parens   = "(" SNL Condition SNL ")"                                                       \n'+

    'Call     = Keyword SNL "(" SNL ArgList SNL ")"                                             \n'+

    'ArgList  = AddArg+                                                                         \n'+
    'AddArg   = ","* SNL Condition                                                              \n'+
    # Calls to lookup functions don't use keywords, so we have to use identifiers.
    #    They take only one parameter. This could cause problems.

    ###################################Subscript Element###########################################
    'Subs     = (SNL UnderSub SNL ";")+ SNL                                                     \n'+ #this is for parsing 2d array
    'UnderSub = ("-"/"+")? Number SNL ("," SNL ("-"/"+")? Number SNL)+                          \n'+ # for parsing 1d array - better to be 1d list
    #undersub is creating errors parsing function arguments that are also numbers...
    'LUCall   = Identifier SNL "(" SNL Condition SNL ")"                                        \n'+
    'Signed   = ("-"/"+") Primary                                                               \n'+
    # We have to break 'reference' out from identifier, because in reference situations
    #    we'll want to add self.<name>(), but we also need to be able to clean and access
    #    the identifier independently. To force parsimonious to handle the reference as
    #    a seperate object from the Identifier, we add a trailing optional space character
    'Reference = Identifier SNL Subtext* SNL                                                \n'+

    'ConCall  = ConKeyword SNL "(" SNL ArgList SNL ")"                                          \n'+

    'Number   = ((~"[0-9]"+ "."? ~"[0-9]"*) / ("." ~"[0-9]"+)) (("e"/"E") ("-"/"+") ~"[0-9]"+)? \n'+
    'Identifier = Basic_Id / Special_Id                                                         \n'+
    'Basic_Id = Letter (Letter / Digit / ~"[_\s]")*                                             \n'+
    'Special_Id = "\\""  ~"[^\\"]"*  "\\""                                                      \n'+ #won't handle where vensim has escaped double quotes...
    'Letter   = ~"[a-zA-Z]"                                                                     \n'+
    'Digit    = ~"[0-9]"                                                                        \n'+

    # We separated out single space characters from the _ entry, so that it can have any number or
    #   combination of space characters.
    '_ = spacechar*                                                                             \n'+
    # Vensim uses a 'backslash, newline' syntax as a way to split an equation onto multiple lines.
    #   We address this by including it as  an allowed space character.
    # Todo: do we really want the exclamation to be a space character? throw these away?
    'spacechar = exclamation* " "* ~"\t"* (~r"\\\\" NL)*                                       \n'+
    'exclamation = "!"+                                         \n' +
    'Keyword = %s  \n'%keywords +
    'ConKeyword = %s  \n'%con_keywords
    )


class TextParser(NodeVisitor):
    """
    Each function in the parser takes an argument which is the result of having visited
    each of the children.

    """
    def __init__(self, grammar, filename, text, dictofsubs):
        self.filename = filename
        self.builder = builder.Builder(self.filename, dictofsubs)
        self.grammar = parsimonious.Grammar(grammar)
        self.dictofsubs = dictofsubs
        self.parse(text)


    def parse(self, text):

        self.ast = self.grammar.parse(text)
        return self.visit(self.ast)

    ############# 'entry' level translators ############################
    visit_Entry = visit_Component = NodeVisitor.lift_child

    def skip(self, n, vc):
        # Todo: Take more advantage of this...
        return ''

    def visit_ModelEntry(self, n, (_1, Identifier, _2, tld1, _3, Unit, _4,
                                   tld2, _5, Docstring, _6, pipe, _7)):
        # Todo: Add docstring handling
        pass

    def visit_Stock(self, n, (Identifier, _1, Sub,_10, eq, _2, integ, _3,
                              lparen, NL1, expression, _6,
                              comma, NL2, initial_condition, _9, rparen)):
        self.builder.add_stock(Identifier, Sub, expression, initial_condition)
        return Identifier

    def visit_Subtext(self,n,(_1,lparen,_2,element,_3,rparen,_4)):
        return re.sub(r'[\n\t\\ ]','',element)

    def visit_Flows(self,n,(flows,flaux)):
        flowname=flaux[0]
        if flows:
            flows.append(flaux)
        else:
            flows=[]
            flows.append(flaux)
        flowsub=[]
        flowelem=[]
        for i in range(len(flows)):
            flowsub.append(flows[i][1])
            flowelem.append(flows[i][2])
        self.builder.add_flaux(flowname,flowsub,flowelem)
        return flowname

    def visit_Flowint(self,n,(flows)):
        return flows

    def visit_Flow(self, n, (Flaux,_1,tilde,SNL)):
        return Flaux

    def visit_Flaux(self, n, (Identifier, _1, Sub, _2, eq, SNL, expression)):
        return [Identifier,Sub,expression]

    def visit_Unit(self, n, vc):
        return n.text.strip()

    visit_Docstring = visit_Unit

    def visit_Subscript(self,n,(Identifier,_1,col,NL,Subelem)):
        #self.builder.add_Subscript(self.filename, Identifier, Subelem)
        return Identifier

    ######### 'expression' level visitors ###############################

    def visit_ConCall(self, n, (ConKeyword, _1, lparen, _2, args, _4, rparen)):
        #todo: build out omitted cases
        if ConKeyword == 'DELAY1': #DELAY3(Inflow, Delay)
            return self.builder.add_n_delay(delay_input=args[0],
                                            delay_time=args[1],
                                            inital_value=str(0),
                                            order=1,
                                            sub=[''])
        elif ConKeyword == 'DELAY1I':
            pass
        elif ConKeyword == 'DELAY3':
            return self.builder.add_n_delay(delay_input=args[0],
                                            delay_time=args[1],
                                            initial_value=str(0),
                                            order=3,
                                            sub=[''])
        elif ConKeyword == 'DELAY N':  # DELAY N(Inflow, Delay, init, order)
            return self.builder.add_n_delay(delay_input=args[0],
                                            delay_time=args[1],
                                            initial_valye=args[2],
                                            order=args[3],
                                            sub=[''])
        elif ConKeyword == 'SMOOTH':
            pass
        elif ConKeyword == 'SMOOTH3':  # SMOOTH3(Input,Adjustment Time)
            return self.builder.add_n_smooth(args[0], args[1], str(0), 3)
        elif ConKeyword == 'SMOOTH3I':  # SMOOTH3I( _in_ , _stime_ , _inival_ )
            return self.builder.add_n_smooth(args[0], args[1], args[2], 3)
        elif ConKeyword == 'SMOOTHI':
            pass
        elif ConKeyword == 'SMOOTH N':
            pass

        elif ConKeyword == 'INITIAL':
            return self.builder.add_initial(args[0])
        # Todo: return whatever you call to get the final stage in the construction

    def visit_Keyword(self, n, vc):
        return dictionary[n.text.upper()]

    def visit_Reference(self, n, (Identifier, _1, Subs, _2)):
        # Todo: seriously think about how this works with subscripts.
        #  This could be a fundamentally paradigm shattering thing. We need to know the
        #  position to reference in an array that isn't guaranteed to be created yet...
        #  for now, use old version. this is going to fail, though...
        if Subs:
            InterSub=Subs.split(",")
            subscript='[%s]'%','.join(map(str, getelempos(Subs.replace("!", ""), self.dictofsubs)))
            subscript=re.sub(r'\[(:,*)+]*\]','',subscript)
            getelemstr=Identifier+'()'+subscript
            if len(InterSub)==1:
                return getelemstr
            else:
                if re.search("!",Subs) and Subs.count("!")==1:
                    sumacross=0
                    for i in range(len(InterSub)):
                        if re.search("!",InterSub[i]):
                            sumacross=i
                    getelemstr+=",%i"%sumacross
                return getelemstr
        else:
            return Identifier+'()'

    def visit_Identifier(self, n, vc):
        string = n.text
        return builder.make_python_identifier(string)

    def visit_Call(self, n, (Translated_Keyword, _1, lparen, _2, args, _3, rparen)):
        return Translated_Keyword+'('+', '.join(args)+')'

    def visit_LUCall(self, n, (Identifier, _1, lparen, _2, Condition, _3,  rparen)):
        return Identifier+'('+Condition+')'

    def visit_Copair(self, n, (lparen, _1, Subs, _2, rparen)):
        Subs=Subs.replace("\\","")
        xcoord=Subs.split(',')[0]
        ycoord=Subs.split(',')[1]
        return (float(xcoord), float(ycoord))

    def visit_Lookup(self, n, (Identifier, _1, lparen, _2, Range,
                     _3, CopairList, _4, rparen)):
        self.builder.add_lookup(Identifier, Range, CopairList)
        return Identifier

    def visit_AddCopair(self, n, (_1, comma, _2, Copair)):
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

    visit_SNL = skip


def getelempos(element, dictofsubs):
    """
    Helps for accessing elements of an array: given the subscript element names,
    returns the numerical ranges that correspond
    Parameters
    ----------
    element
    dictofsubs
    Returns
    -------
    """
    # Todo: Make this accessible to the end user
    #  The end user will get an unnamed array, and will want to have access to
    #  members by name.

    position=[]
    elements=element.replace('!','').replace(' ', '').split(',')
    for element in elements:
        if element in dictofsubs.keys():
            position.append(':')
        else:
            for d in dictofsubs.itervalues():
                try:
                    position.append(d[element])
                    break
                except: pass

    return tuple(position)


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
    # Todo: Test translation independent of load
    #  We can make smaller unit tests that way...

    # Step 1 parser gets the subscript dictionary
    with open(mdl_file, 'rU') as file:
            text = file.read()

    # Todo: consider making this easier to understand, maybe with a second parsimonious grammar
    f2=re.findall(r'([a-zA-Z][^:~\n.\[\]\+\-\!/\(\)\\\&\=]+:+[^:~.\[\]\+\-\!/\(\),\\\&\=]+,+[^:~.\[\]\+\-\!/\(\)\\\&\=]+~)+',text)
    for i in range(len(f2)):
        f2[i]=re.sub(r'[\n\t~]','',f2[i])

    dictofsubs = {}
    for i in f2:
        Family = builder.make_python_identifier(i.split(":")[0])
        Elements = i.split(":")[1].split(",")
        for i in range(len(Elements)):
            Elements[i] = builder.make_python_identifier(Elements[i].strip())
        dictofsubs[Family] = dict(zip(Elements, range(len(Elements))))

    # Step 2 parser writes the file
    outfile_name = mdl_file[:-4]+'.py'
    parser = TextParser(file_grammar, outfile_name, text, dictofsubs)
    parser.builder.write()

    return parser.filename

translate_vensim.__doc__ += doc_supported_vensim_functions()
