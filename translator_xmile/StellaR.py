#-------------------------------------------------------------------------------
# Name:        StellaR
# Version:     1.3
# Purpose:     Translate dynamic simulation models from Stella into R
#
# Author:      Babak Naimi
# Email:       naimi@r-gis.net
#              naimi.b@gmail.com

# Copyright:   (c) naimi, March 2012
#-------------------------------------------------------------------------------

import os
import re
import sys
def main():
    if len(sys.argv) <  3:
        print ("Usage: StellaR <input_stella.txt> <outname>\n")
        print ("\t -- input_stella.txt is a text file exported from STELLA model\n")
        print ("\t -- outname is the name of R project where the main R script (outname.R) as well as data files and other functions are storded\n")
        sys.exit()

    fname1 = sys.argv[1]
    outputName = sys.argv[2]
    outPath=outputName
    outName_split = re.split('(\\\|/)',outputName)
    if len(outName_split) > 1:
        outputName=outName_split[-1]

    #############################
    ## Classes-------------------
    ###-----------------------------

    class model:
        models=0
        def __init__(self,name):
            self.modeltxtlist=name
            self.setup()
            model.models+=1
        def setup(self):
            self.inflows=[]
            self.outflows=[]
            self.rank = 0
            self.init = 0


    class flow:
        flows=0
        def __init__(self,name):
            self.name=name
            self.input_txt=""
            self.setup()
            flow.flows+=1
        def setup(self):
            self.in_convertors=[]
            self.in_stocks=[]
            self.in_flows=[]
            self.hasFunction=False
            self.isIf=False
            self.Eq=None



    class convertor:
        convertors=0
        def __init__(self,name):
            self.name=name
            self.input_txt = ""
            self.setup()
            convertor.convertors+=1
        def setup(self):
            self.in_convertors=[]
            self.value = None
            self.hasFunction = False
            self.isIf = False
            self.Eq = None
            self.isData = False
            self.Data=None


    class if_class:
        if_classes=0
        def __init__(self,name):
            self.name=name
            self.if_clause=""
            self.then_clause=""
            self.else_clause=""
            if_class.if_classes+=1
    class multi_if_class:
        if_classes=0
        def __init__(self,name):
            self.name=name
            self.if_clause=""
            self.then_clause=""
            self.else_clause=""
            multi_if_class.if_classes+=1
    class inputData:
        inputData=0
        def __init__(self,name):
            self.name=name
            self.xname=""
            self.x=[]
            self.y=[]



    #############################
    ## Functions-------------------
    ###-----------------------------
    def flowExplore(lines,flowname):
        i=0
        flowEq=""
        while i< len(lines):
            l=lines[i].strip().split()
            #extracting parameter names
            if l[0]==flowname:
                tempc=i+1
                multilineflow=True
                flowlineNumbers=1
                while (multilineflow):
                    if (tempc < len(lines)):
                        if ('=' in lines[tempc].strip().split()):
                            multilineflow=False
                        else:
                            flowlineNumbers+=1
                            tempc+=1
                    else:
                        multilineflow=False

                l=l[2:]
                flowEq=flowEq+single_string(l)
                tempc=1
                while (flowlineNumbers > 1):
                    ll=lines[tempc+i].strip().split()
                    flowEq=flowEq+single_string(ll)
                    tempc+=1
                    flowlineNumbers-=1
                i=len(lines)
            i+=1
        flowEq=re.sub('{.*}','',flowEq).strip()
        return (flowEq)

    def convExplore(lines,convname):
        i=0
        convEq=""
        while i< len(lines):
            l=lines[i].strip().split()
            #extracting parameter names
            if l[0]==convname:
                tempc=i+1
                multilineconv=True
                convlineNumbers=1
                while (multilineconv):
                    if (tempc < len(lines)):
                        if ('=' in lines[tempc].strip().split()):
                            multilineconv=False
                        else:
                            convlineNumbers+=1
                            tempc+=1
                    else:
                        multilineconv=False

                l=l[2:]
                convEq=convEq+single_string(l)
                tempc=1
                while (convlineNumbers > 1):
                    ll=lines[tempc+i].strip().split()
                    convEq=convEq+single_string(ll)
                    tempc+=1
                    convlineNumbers-=1
                i=len(lines)
            i+=1
        convEq=re.sub('{.*}','',convEq).strip()
        return (convEq)


    def single_string(splited_line):
        sing=""
        for i_count in range(0,len(splited_line)):
            sing=sing+" "+splited_line[i_count]
        return sing

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def if_extract(txtline):
        ifsplit = re.split('(if\s+|IF\s+|If\s+|if|IF|If)',txtline,1)
        if ('' in ifsplit): ifsplit.remove('')
        if (' ' in ifsplit): ifsplit.remove(' ')
        ifsplit2 = re.split('(then\s+|THEN\s+|Then\s+|then|THEN|Then)',ifsplit[1],1)
        if_clause = ifsplit2[0].strip()
        ifsplit3 = re.split('(else\s+|ELSE\s+|Else\s+|else|ELSE|Else)',ifsplit2[2],1)
        then_clause = ifsplit3[0]
        else_clause = None
        if len(ifsplit3) > 2: else_clause = ifsplit3[2].strip()
        return([if_clause,then_clause,else_clause])

    def HasFunction(txtline):
        hf=False
        conv_temp = re.split('(=|\*|>|<|,|\+|\^|\-|[)]|[(]|[{]|[}]|/|\s+)',txtline)
        for i in range(0,len(conv_temp)):
            conv_temp[i]=conv_temp[i].strip()
            if (conv_temp[i].upper() in supported_func_withoutIF): hf=True
        return(hf)

    def FunctionTranslator(x):
        return {
            'EXP': x.lower(),
            'MIN':x.lower(),
            'MAX':x.lower(),
            'MEAN':x.lower(),
            'SUM':x.lower(),
            'ABS':x.lower(),
            'SIN':x.lower(),
            'COS':x.lower(),
            'TAN':x.lower(),
            'LOG10':x.lower(),
            'SQRT':x.lower(),
            'ROUND':x.lower(),
            'LOGN':'log',
            'ARCTAN':'atan',
            'TIME':'t',
            'PI':x.lower(),
            'INT':'floor',
            'DT':'DT'
            }[x]


    def TranslateFunctions(txtline):
        lst = re.split('(=|\*|>|<|,|\^|\+|\-|[)]|[(]|[{]|[}]|/|\s+)',txtline)
        for i in range(0,len(lst)): lst[i]=lst[i].strip()
        while ('' in lst): lst.remove('')
        newtxt=''
        i=0
        while (i < len(lst)):
            if (lst[i].upper() in supported_func_S):
                lst[i]=FunctionTranslator(lst[i].upper())
                newtxt=newtxt+lst[i]+' '
                i+=1
            elif (lst[i].upper() in supported_func_L):
                newtxt=newtxt+lst[i].upper()+' '
                i+=1
            else:
                newtxt=newtxt+lst[i]+' '
                i+=1
        return(newtxt)

    def convWrite(conv):
        if convertors[conv].isIf:
            iftemp = if_extract(convertors[conv].Eq)
            if ('=' in iftemp[0]): iftemp[0].replace('=','==')
            if ('OR' in iftemp[0]): iftemp[0].replace('OR','|')
            if ('AND' in iftemp[0]): iftemp[0].replace('AND','&')
            ff.writelines(["\t",conv," <- ifelse(",iftemp[0],",\n"])
            if HasFunction(iftemp[1]): iftemp[1]=TranslateFunctions(iftemp[1])
            ff.writelines(["\t\t\t",iftemp[1],",\n"])
            if HasFunction(iftemp[2]): iftemp[2]=TranslateFunctions(iftemp[2])
            if (len(re.findall('(if\s+|IF\s+|If\s+|if|IF|If)',iftemp[2])) > 0):
                iftemp2 = if_extract(iftemp[2])
                if ('=' in iftemp2[0]): iftemp2[0].replace('=','==')
                if ('OR' in iftemp2[0]): iftemp2[0].replace('OR','|')
                if ('AND' in iftemp2[0]): iftemp2[0].replace('AND','&')
                ff.writelines(["\t\t\t\t ifelse(",iftemp2[0],",",iftemp2[1],",",iftemp2[1],") )\n"])
            else:
                ff.writelines(["\t\t\t",iftemp[2],")\n"])
            convT.append(conv)
        elif HasFunction(convertors[conv].Eq):
            convertors[conv].Eq=TranslateFunctions(convertors[conv].Eq)
            ff.writelines(["\t",conv," <- ",str(convertors[conv].Eq),"\n"])
            convT.append(conv)
        else:
            ff.writelines(["\t",conv," <- ",str(convertors[conv].Eq),"\n"])
            convT.append(conv)
        return(convT)

    #---------------------------------

    f1=open(fname1)
    WN =('INFLOWS','OUTFLOWS','INFLOWS:','OUTFLOWS:')
    models={}
    flows={}
    convertors={}
    ifclasses={}
    Mifclasses={}
    lines=f1.readlines()
    f1.close()
    init_pattern=r"INIT\s+(?P<initname>[a-zA-Z0-9_]+)\s+=\s+(?P<initvalue>.*)"
    init_expr=re.compile(init_pattern)
    i=0
    modelrank=0
    ln=0

    while ln < len(lines):
        ltxt=lines[ln].strip().split()
        while('' in ltxt): ltxt.remove('')
        if len(ltxt) > 0:
            if (ltxt[0] ==  'DOCUMENT:' or ltxt[0] in WN):
                lines.remove(lines[ln])
            else:
                lines[ln]=lines[ln].replace('\\','_')
                ln+=1
        else: lines.remove(lines[ln])

    ln=0
    while ln < len(lines):
        if "INIT" in lines[ln]:
            init_position = lines[ln].find("INIT")
            if  init_position > 0:
                init_line = lines[ln][init_position:]
                lines.insert(ln+1,init_line)
                lines[ln] = lines[ln][0:init_position]
        ln+=1


    for line in lines:
        init_match=re.search(init_expr,line)
        if init_match:
            modelrank+=1
            ltxt=lines[i-1].strip().split()[6:-2]
            models[init_match.group('initname')]=model(init_match.group('initname'))
            models[init_match.group('initname')].rank=modelrank
            models[init_match.group('initname')].init=init_match.group('initvalue')
            nextflow=True
            for txt in ltxt:
                txt=txt.replace('(','')
                txt=txt.replace(')','')
                if (txt == "-"):
                    nextflow=False
                if (txt is not "+" and txt is not "-"):
                    if (nextflow):
                        models[init_match.group('initname')].inflows.append(txt)
                        if (txt not in flows):
                            flows[txt]=flow(txt)
                            flows[txt].input_txt=flowExplore(lines,txt)
                    else:
                        models[init_match.group('initname')].outflows.append(txt)
                        if (txt not in flows):
                            flows[txt]=flow(txt)
                            flows[txt].input_txt=flowExplore(lines,txt)
                        nextflow=True
        i+=1

    ######## Analyzing text of each flow equations
    supported_func=('RANDOM','EXP','MAX','MEAN','MIN','MOD','INT','IF','THEN','ELSE','DT','SUM','PI','ABS','SINWAVE','SIN','COS','TAN','ARCTAN','LOG10','LOGN','LOGNORMAL','DELAY','NOT','AND','OR','TIME','TREND','SQRT','ROUND')
    supported_func_withoutIF=('RANDOM','EXP','MAX','MEAN','MIN','MOD','INT','DT','SUM','PI','ABS','SINWAVE','SIN','COS','TAN','ARCTAN','LOG10','LOGN','LOGNORMAL','DELAY','TIME','TREND','SQRT','ROUND')

    supported_func_S=('EXP','MAX','MEAN','MIN','SUM','INT','PI','ABS','SIN','COS','TAN','ARCTAN','LOG10','TIME','SQRT','LOGN','ROUND','DT')
    supported_func_L=('RANDOM','MOD','SINWAVE','LOGNORMAL','DELAY','TREND','NORMAL','POISON','EXPRAND','COUNTER')

    optlist=('*','/','+','-','(',')','=','<','>','<=','>=',',',""," ",'{','}','^','\\')
    for fl in flows.keys():
        txtline = flows[fl].input_txt
        txtline=re.split('(=|\*|>|<|,|\^|\+|\-|[)]|[(]|[{]|[}]|/|\s+)',txtline)
        while('' in txtline): txtline.remove('')
        while(' ' in txtline): txtline.remove(' ')
        txtline = single_string(txtline)
        flows[fl].input_txt=txtline
        # check for IF statement
        if_num=len(re.findall('(if\s+|IF\s+|If\s+|if|IF|If)',txtline))
        if (if_num > 0): flows[fl].isIf = True
        flows[fl].Eq=txtline
        # extract convertors
        conv_temp = re.split('(=|\*|>|<|,|\^|\+|\-|[)]|[(]|[{]|[}]|/|\s+)',txtline)
        for i in range(0,len(conv_temp)):
            conv_temp[i]=conv_temp[i].strip()
            if (conv_temp[i].upper() in supported_func_withoutIF): flows[fl].hasFunction=True
            if ((conv_temp[i].upper() not in supported_func) and (conv_temp[i] not in optlist) and (not is_number(conv_temp[i])) and conv_temp[i] not in models.keys() and conv_temp[i] not in flows.keys()):
                if (conv_temp[i] not in convertors.keys()):
                    convertors[conv_temp[i]]=convertor(conv_temp[i])
                #if (conv_temp[i] not in flows[fl].in_convertors):
                #    flows[fl].in_convertors.append(conv_temp[i])
                elif (conv_temp[i] in models.keys() and conv_temp[i] not in flows[fl].in_stocks):
                    flows[fl].in_stocks.append(conv_temp[i])
                if (conv_temp[i] in convertors.keys() and (conv_temp[i] not in flows[fl].in_convertors)):
                    flows[fl].in_convertors.append(conv_temp[i])
                if (conv_temp[i] in flows.keys()):
                    flows[fl].in_flows.append(conv_temp[i])

    ##################
    ## Analyzing text for each convertor ---------
    ##################
    newconv={}
    for conv in convertors.keys():
        txtline=convExplore(lines,conv)
        convertors[conv].input_txt=txtline
        if ('GRAPH' in txtline):
            convertors[conv].isData=True
        elif (is_number(txtline)):
            convertors[conv].value=float(txtline)
        else:
            if (len(re.findall('(if\s+|IF\s+|If\s+|if|IF|If)',txtline)) > 0):
                convertors[conv].isIf = True
            convertors[conv].Eq = txtline
            conv_temp = re.split('(=|\*|>|<|,|\+|\^|\-|[)]|[(]|[{]|[}]|/|\s+)',txtline)
            for i in range(0,len(conv_temp)):
                conv_temp[i]=conv_temp[i].strip()
                if (conv_temp[i].upper() in supported_func_withoutIF): convertors[conv].hasFunction=True
                if ((conv_temp[i].upper() not in supported_func) and (conv_temp[i] not in optlist) and (not is_number(conv_temp[i])) and conv_temp[i] not in models.keys() and conv_temp[i] not in flows.keys()):
                    if (conv_temp[i] not in convertors.keys()):
                        newconv[conv_temp[i]]=convertor(conv_temp[i])
                    if (conv_temp[i] not in convertors[conv].in_convertors):
                        convertors[conv].in_convertors.append(conv_temp[i])

    rr=True
    while (rr):
        newconv2={}
        for conv in newconv.keys():
            convertors[conv]=newconv[conv]
            txtline=convExplore(lines,conv)
            convertors[conv].input_txt=txtline
            if ('GRAPH' in txtline):
                convertors[conv].isData=True
            elif (is_number(txtline)):
                convertors[conv].value=float(txtline)
            else:
                if (len(re.findall('(if\s+|IF\s+|If\s+|if|IF|If)',txtline)) > 0):
                    iftemp = if_extract(txtline)
                    convertors[conv].isIf = True
                convertors[conv].Eq = txtline
                conv_temp = re.split('(=|\*|>|<|,|\+|\^|\-|[)]|[(]|[{]|[}]|/|\s+)',txtline)
                for i in range(0,len(conv_temp)):
                    conv_temp[i]=conv_temp[i].strip()
                    if (conv_temp[i].upper() in supported_func_withoutIF): convertors[conv].hasFunction=True
                    if ((conv_temp[i].upper() not in supported_func) and (conv_temp[i] not in optlist) and (not is_number(conv_temp[i])) and conv_temp[i] not in models.keys() and conv_temp[i] not in flows.keys()):
                        if (conv_temp[i] not in convertors.keys()):
                            newconv2[conv_temp[i]]=convertor(conv_temp[i])

                        if (conv_temp[i] not in convertors[conv].in_convertors):
                            convertors[conv].in_convertors.append(conv_temp[i])
        if (len(newconv2.keys())>0):
            newconv=newconv2
        else:
            rr=False


    additional_lines={}
    ln=0
    while ln < len(lines):
        l=lines[ln].strip().split()
        l=re.split('([)]|[(]|[{]|[}]|\s+)',l[0])[0]

        if (l !='INIT' and l not in flows.keys() and l not in convertors.keys() and l not in models.keys() and l != ''):

            txtline=convExplore(lines,l)

            if ('GRAPH' in txtline):
                convertors[l]=convertor(l)
                convertors[l].input_txt=txtline
                convertors[l].isData=True
            elif (is_number(txtline)):
                convertors[l]=convertor(l)
                convertors[l].input_txt=txtline
                convertors[l].value=float(txtline)
            else:
                additional_lines[l]=lines[ln]
        ln+=1

    ###############
    ##inputs={}
    for conv in convertors.keys():
        if (convertors[conv].isData):
            txt=convertors[conv].input_txt.strip().split()
            arg = txt[0]
            arg=re.split('GRAPH\(|\)',arg)
            while ('' in arg): arg.remove('')
            convertors[conv].Data=inputData(conv)
            if (arg[0] in convertors.keys()):
                convertors[conv].Data.xname=arg[0]
            else:
                convertors[conv].Data.xname='t'
            t=[];v=[]

            for i in range(1,len(txt)):
                txts = re.split('(,|\(|\))',txt[i])
                while ('' in txts): txts.remove('')
                while (',' in txts): txts.remove(',')
                if ('(' in txts):
                    t.append(float(txts[1]))
                elif(')' in txts):
                    v.append(float(txts[0]))

            #inputs[conv+'_Time']=t
            #inputs[conv+'_Value']=v
            convertors[conv].Data.x=t
            convertors[conv].Data.y=v
    ###################################
    # writing output
    if (not os.path.isdir(outPath)):
        os.mkdir(outPath)
    ff = open(outPath+"/"+outputName+".R",'w')
    ff.writelines(["if (require(deSolve) == F) {", 
                   "\tinstall.packages('deSolve', repos='http://cran.r-project.org');\n",
                   "if (require(deSolve) == F)\n"
                   "print ('Error: deSolve package is not installed on your machine')}\n"])

    ff.writelines(["model<-function(t,Y,parameters,...) { \n"])
    #-----
    ff.writelines(["\n"])
    ff.writelines(["\tTime <<- t\n"])
    Eqs = models.keys()
    for eq in Eqs:
        ff.writelines(["\t",eq," <- Y['",eq,"']\n"])

    ff.writelines(["\n"])
    #----
    convlist = list(convertors.keys())
    parms = []
    convT = []
    for conv in convlist:
        if (convertors[conv].value is not None):
            ff.writelines(["\t", conv, " <- parameters['", conv, "']\n"])
            parms.append(conv)
            convT.append(conv)

    for conv in convT: 
        convlist.remove(conv)

    ff.writelines(["\n"])
    convT2 = []
    for conv in convlist:
        if convertors[conv].isData:
            if (convertors[conv].Data.xname != 't'):
                if (convertors[conv].Data.xname not in convT2 and
                    convertors[conv].Data.xname in convlist):
                    convT2.append(convertors[conv].Data.xname)
                    convWrite(convertors[conv].Data.xname)
                ff.writelines(["\t",conv," <- inputData(",convertors[conv].Data.xname,", '",conv,"')\n"])
                convT2.append(conv)
            else:
                ff.writelines(["\t",conv," <- inputData(t, '",conv,"')\n"])
                convT2.append(conv)

    for conv in convT2: convlist.remove(conv)

    loop = True
    while (loop):
        len_conv = len(convlist)
        convT=[]
        for conv in convlist:
            if not bool(sum(map(lambda x: x in convlist, convertors[conv].in_convertors))):
                convT=convWrite(conv)
        for conv in convT: convlist.remove(conv)
        if len(convlist) == len_conv: loop=False

    #------
    flowlist = list(flows.keys())
    loop = True
    while (loop):
        len_flow = len(flowlist)
        flowT=[]
        for fl in flowlist:
            flowWrite=False
            if len(flows[fl].in_flows) > 0:
                for i in range(0,len(flows[fl].in_flows)):
                    if (flows[fl].in_flows[i] not in flowlist): flowWrite=True
            else:
                flowWrite=True
            if (flowWrite):
                if flows[fl].isIf:
                    iftemp = if_extract(flows[fl].Eq)
                    if ('=' in iftemp[0]): iftemp[0].replace('=','==')
                    if ('OR' in iftemp[0]): iftemp[0].replace('OR','|')
                    if ('AND' in iftemp[0]): iftemp[0].replace('AND','&')
                    ff.writelines(["\t",fl," <- ifelse(",iftemp[0],",\n"])
                    if HasFunction(iftemp[1]): iftemp[1]=TranslateFunctions(iftemp[1])
                    ff.writelines(["\t\t\t",iftemp[1],",\n"])
                    if HasFunction(iftemp[2]): iftemp[2]=TranslateFunctions(iftemp[2])
                    if (len(re.findall('(if\s+|IF\s+|If\s+|if|IF|If)',iftemp[2])) > 0):
                        iftemp2 = if_extract(iftemp[2])
                        if ('=' in iftemp2[0]): iftemp2[0].replace('=','==')
                        if ('OR' in iftemp2[0]): iftemp2[0].replace('OR','|')
                        if ('AND' in iftemp2[0]): iftemp2[0].replace('AND','&')
                        ff.writelines(["\t\t\t ifelse(",iftemp2[0],",",iftemp2[1],",",iftemp2[1],") )\n"])
                    else:
                        ff.writelines(["\t\t\t",iftemp[2],")\n"])
                    flowT.append(fl)
                elif HasFunction(flows[fl].Eq):
                    flows[fl].Eq=TranslateFunctions(flows[fl].Eq)
                    ff.writelines(["\t",fl," <- ",str(flows[fl].Eq),"\n"])
                    flowT.append(fl)
                else:
                    ff.writelines(["\t",fl," <- ",str(flows[fl].Eq),"\n"])
                    flowT.append(fl)


        for fl in flowT: flowlist.remove(fl)
        if len(flowlist) == len_flow: loop=False

    ff.writelines(["\n"])
    if (len(additional_lines.keys()) > 0):
        ff.writelines(["\n  ###------ The following lines have not been translated!!\n"])
        for i in range(0,len(additional_lines.keys())):
            ff.writelines(["\t#--- ",additional_lines[additional_lines.keys()[i]],"\n"])
        ff.writelines(["  ###-------------\n"])
    #------
    rank=1
    Eqs = list(models.keys())
    loop = True
    EqsL=[]
    while (loop):
        EqsT=[]

        for eq in Eqs:
            if (models[eq].rank == rank):
                ff.writelines(["\n\t d",eq," = "])
                if (len(models[eq].inflows) > 0):
                    ff.writelines([models[eq].inflows[0]," "])
                    if (len(models[eq].inflows) > 1):
                        for j in range(1,len(models[eq].inflows)):
                            ff.writelines([" + ",models[eq].inflows[j]," "])
                if (len(models[eq].outflows) > 0):
                        for j in range(0,len(models[eq].outflows)):
                            ff.writelines([" - ",models[eq].outflows[j]," "])
                rank+=1
                EqsT.append(eq)
                EqsL.append(eq)
        for eq in EqsT: Eqs.remove(eq)
        if len(Eqs) == 0: loop=False
    #----
    ff.writelines(["\n"])
    ff.writelines(["\t list(c(d",EqsL[0]])
    for i in range(1,len(EqsL)): ff.writelines([", d",EqsL[i]])
    ff.writelines(["))\n}"])
    ff.writelines(["\n##############################################\n"])
    ff.writelines(["##############################################\n"])
    ########################-----------------------------------------------------
    ff.writelines(["\n"])
    for eq in EqsL:
        if not is_number(models[eq].init):
            if models[eq].init in convertors.keys():
                models[eq].init = "parms['"+models[eq].init+"']"

    convlist = list(convertors.keys())
    conv=[]
    for c in convlist:
        if (convertors[c].value is not None): conv.append(c)

    if len(conv) > 0:
        ff.writelines(["parms <- c(",conv[0]," = ",str(convertors[conv[0]].value)])
        for i in range(1,len(conv)): ff.writelines([", ",conv[i]," = ",str(convertors[conv[i]].value) ])
        ff.writelines([")\n"])

    ff.writelines(["Y <- c(",EqsL[0]," = ",str(models[EqsL[0]].init)])
    for i in range(1,len(EqsL)): ff.writelines([", ",EqsL[i]," = ",str(models[EqsL[i]].init) ])
    ff.writelines([")\n"])

    #------------------
    ## writing data & support functions

    convlist = list(convertors.keys())
    conv=[]
    for c in convlist:
        if (convertors[c].isData): conv.append(c)
    fs=open(outPath+"/"+outputName+"_functions.R",'w')
    if len(conv) > 0:

        for i in range(0,len(conv)):
            fd=open(outputName+"/"+outputName+"_Data_"+convertors[conv[i]].Data.name+".csv",'w')
            fd.writelines([convertors[conv[i]].Data.xname,",",convertors[conv[i]].Data.name])
            for j in range(0,len(convertors[conv[i]].Data.x)):
                fd.writelines(["\n",str(convertors[conv[i]].Data.x[j]),",",str(convertors[conv[i]].Data.y[j])])
            fd.close()

        fs.writelines(["input.Data <- c()\n"])
        for i in range(0,len(conv)):

            fs.writelines([" temp <- read.csv('",outputName,"_Data_",convertors[conv[i]].Data.name,".csv')\n"])
            fs.writelines([" temp <- list(",convertors[conv[i]].Data.name," = temp)\n"])
            fs.writelines([" input.Data <- c(input.Data, temp)\n"])
        #fs.writelines(["}\n"])
        fs.writelines(["rm(temp)\n"])
    #----
        fs.writelines(["\ninputData <- function(x,name,datalist=input.Data) {\n"])
        fs.writelines(["\tdf=datalist[[name]]\n\tminT <- min(df[,1],na.rm=T)\n\tmaxT <- max(df[,1],na.rm=T)\n"])
        fs.writelines(["\tif (x < minT | x > maxT) {\n\t\tl <- lm(get(colnames(df)[2])~poly(get(colnames(df)[1]),3),data=df)\n\t\tdo <- data.frame(x); colnames(do) <- colnames(df)[1]\n\t\to <- predict(l,newdata=do)[[1]]"])
        fs.writelines(["\t} else {\n"])
        fs.writelines(["\tt1 <- max(df[which(df[,1] <= x),1])\n\tt2 <- min(df[which(df[,1] >= x),1])\n\tif (t1 == t2) {\n"])
        fs.writelines(["\t\to <- df[t1,2]}\n"])
        fs.writelines(["\telse {\n\t\tw1=1/abs(x-t1);w2=1/abs(x-t2)\n\to <- ((df[which( df[,1] == t1),2]*w1)+(df[which( df[,1] == t2),2]*w2)) / (w1+w2) } }\n  o }\n"])

    fs.writelines(["#----------------\nMOD <- function(x,y) {	x %% y }\n"])
    fs.writelines(["#----------------\nRANDOM <- function(x,y) { runif(1,x,y)}\n"])
    fs.writelines(["#----------------\nNORMAL <- function(x,y) { rnorm(1,x,y) }\n"])
    fs.writelines(["#----------------\nPOISON <- function(x)  { rpois(1,x) }\n"])
    fs.writelines(["#----------------\nLOGNORMAL <- function(x,y) { rlnorm(1,x,y) }\n"])
    fs.writelines(["#----------------\nEXPRAND <- function (x) { rxep(1,x) }\n"])
    fs.writelines(["#----------------\nSINWAVE <- function(x,y) { x * sin(2 * pi * Time / y) }\n"])
    fs.writelines(["#----------------\nCOSWAVE <- function(x,y) { x * cos(2 * pi * Time / y) }\n"])
    fs.writelines(["#----------------\nCOUNTER <- function(x,y) {\n"])
    fs.writelines(["\tif (Time == time[1]) COUNTER_TEMP <<- x\n"])
    fs.writelines(["\tif (!exists('COUNTER_TEMP')) COUNTER_TEMP <<- x\n"])
    fs.writelines(["\telse COUNTER_TEMP <- COUNTER_TEMP  + 1\n"])
    fs.writelines(["\tif (COUNTER_TEMP == y) COUNTER_TEMP  <<- x\n"])
    fs.writelines(["\treturn(COUNTER_TEMP)}\n"])
    fs.writelines(["#--------\nTREND <- function(x,y,z=0) {\n"])
    fs.writelines(["\tif (!exists('AVERAGE_INPUT')) AVERAGE_INPUT <<- z\n"])
    fs.writelines(["\tCHANGE_IN_AVERAGE <- (x - AVERAGE_INPUT) / y\n"])
    fs.writelines(["\tAVERAGE_INPUT <<- AVERAGE_INPUT + (DT * CHANGE_IN_AVERAGE)\n"])
    fs.writelines(["\tTREND_IN_INPUT <- (x - AVERAGE_INPUT) / (AVERAGE_INPUT * y)\n"])
    fs.writelines(["\tif (Time == time[length(time)]) rm(AVERAGE_INPUT,envir=environment(TREND))\n"])
    fs.writelines(["\tTREND_IN_INPUT}\n"])
    fs.writelines(["#-----------------\nDELAY <- function(x,y,z=NA) { x } # should be developed!\n"])
    fs.close()

    ff.writelines(["\nsource('",outputName,"_functions.R')\n"])
    ff.writelines(["DT <- 0.25\n"])
    ff.writelines(["time <- seq(0.001,100,DT)\n"])
    ff.writelines(["out <- ode(func=model,y=Y,times=time,parms=parms,method='rk4')\n"])
    ff.writelines(["plot(out)\n"])
    ff.close()
    print ("STELLA model is successfully translated into R!\n")
    print ("Babak Naimi (naimi@itc.nl)\n")
    print ("Alexey Voinov (voinov@itc.nl)\n")


if __name__ == '__main__':
    main()