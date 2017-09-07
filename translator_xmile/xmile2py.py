# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 23:45:45 2017

@author: Miguel
"""

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from bs4 import BeautifulSoup as bs
import sys
import os


def main():
    if len(sys.argv) < 3:
        print("Usage: xmile2py <input_stella.stmx> <outname>\n")
        print("\t -- input_stella.stmx. This file type is a xml variant" +
              " text file use in STELLA models from version 10 and up\n")
        print("\t -- outname is the name of an R project directory where" +
              " the main R script (outname.R), data files and other" +
              " functions will be stored\n")
        sys.exit()

    # exctract data from command line input
    model_file = sys.argv[1]
    outputName = sys.argv[2]
    outPath = os.path.abspath(outputName)
    outName = os.path.basename(outputName)

    # Prepare directory to store output
    if (not os.path.isdir(outPath)):
        os.mkdir(outPath)

# os.chdir("C:/Users/Miguel/Documents/1 Nube/Dropbox/Curso posgrado/2015/Día 2/Influenza/El modelo general")
# model_file = "SIR_HK (stella 10).stmx"
# os.chdir("C:/Users/Miguel/Documents/0 Versiones/2 Proyectos/pysd/translator_xmile")
# model_file = "SIR_HK.stmx"

    # Read model file
    with open(model_file) as fmodel:
        model_soup = bs(fmodel, "lxml")

    # Extract relevant information to build model
    model_name = model_soup.header.find("name").text
    print("Arg 1:", model_file)
    print("Arg 2:", outputName)
    print("Path:", outPath)
    print("BaseName:", outName)
    print(model_soup.prettify()[0:186])
    print("Nombre del modelo:", model_name)

    model_attributes = xmile_parser(model_soup)

    # output model file
    with open(outPath + "/" + outputName + ".R", "w") as foutput:
        model_output = XmileModel(model_name,
                                  model_attributes["stocks"],
                                  model_attributes["auxs"]).show()
        foutput.writelines(model_output)


# %%Model classes
class XmileStock:
    def __init__(self, stock_dict):
        self.name = stock_dict["stock_name"]
        self.units = stock_dict["units"]
        self.eqn = ""
        if "inflow" in stock_dict:
            self.inflow_name = stock_dict["inflow"]["flow_name"]
            self.inflow = stock_dict["inflow"]["eqn"]
            self.eqn = self.eqn + stock_dict["inflow"]["eqn"]
            self.inflow_units = stock_dict["inflow"]["units"]
        if "outflow" in stock_dict:
            self.outflow_name = stock_dict["outflow"]["flow_name"]
            self.outflow = stock_dict["outflow"]["eqn"]
            self.eqn = self.eqn + " - " + stock_dict["outflow"]["eqn"]
            self.outflow_units = stock_dict["outflow"]["units"]
        self.init = float(stock_dict["val_ini"])
        self.eqn = self.eqn.strip(" ")

    def show(self):
        stock_report = "".join(["STOCK",
                                "\n  Name:", self.name,
                                "\n  Units:", self.units,
                                "\n  Initial value:", str(self.init),
                                "\n  Equation:", self.eqn, "\n"])
        return stock_report


class XmileAux:
    def __init__(self, aux_dict):
        self.name = aux_dict["aux_name"]
        if "ypts" in aux_dict:
            self.eqn = aux_dict["eqn_str"]
            self.type = "series"
            step = (float(aux_dict["x_max"]) - float(aux_dict["x_min"]))\
                / (len(aux_dict["ypts"].split(",")) - 1)
            yval = list(map(float, (aux_dict["ypts"].split(","))))
            self.series = list(map(lambda x, y: (x, float(y)),
                                   [float(aux_dict["x_min"]) + step * x
                                    for x in range(0, len(yval))],
                               aux_dict["ypts"].split(",")))
        else:
            self.units = aux_dict["units"]
            self.value = float(aux_dict["eqn_str"])
            self.type = "value"

    def show(self):
        aux_report = "".join(["CONVERTER",
                              "\n  Name:", self.name,
                              "\n  Type:", self.type])
        if self.type is "value":
            aux_report = "".join([aux_report,
                                  "\n  Value:", self.value,
                                  "\n  Units:", self.units])
        else:
            aux_report = "".join([aux_report,
                                  "\n  Equation: GRAPH({0})".format(self.eqn),
                                  "\n  Data series:",
                                  ", ".join([str(x) for x in self.series])])

        return aux_report


class XmileModel:
    def __init__(self, name, stocks, auxs):
        self.name = name
        self.stocks = stocks
        self.auxs = auxs

    def rTranslation(self):
        pass

        def FunctionTranslator(x):
            return {
                'EXP': x.lower(),
                'MIN': x.lower(),
                'MAX': x.lower(),
                'MEAN': x.lower(),
                'SUM': x.lower(),
                'ABS': x.lower(),
                'SIN': x.lower(),
                'COS': x.lower(),
                'TAN': x.lower(),
                'LOG10': x.lower(),
                'SQRT': x.lower(),
                'ROUND': x.lower(),
                'LOGN': 'log',
                'ARCTAN': 'atan',
                'TIME': 't',
                'PI': x.lower(),
                'INT': 'floor',
                'DT': 'DT'
                }[x]

    def show(self):
        items = []
        for stk in self.stocks:
            eqn = stk.name + "(t)" + " = " + stk.name + " (t - dt)"
            fls = ""
            if hasattr(stk, "inflow"):
                fls = stk.inflow_name
            if hasattr(stk, "outflow"):
                fls = fls + " - " + stk.outflow_name
            item = [eqn + " + (" + fls + ") * dt"]
            item.extend(["    INIT " + stk.name + " = " + str(stk.init)])

            # Define flows
            if hasattr(stk, "inflow"):
                item.extend(["    INFLOWS:"])
                item.extend(["        " + stk.inflow_name +
                             " = " + stk.inflow])
            if hasattr(stk, "outflow"):
                item.extend(["    OUTFLOWS:"])
                item.extend(["        " + stk.outflow_name +
                             " = " + stk.outflow])
            items.append("\n".join(item))

        model_report = "".join(["Top-Level Model <{0}>:\n".format(self.name),
                                "\n".join(items)])

        for ax in self.auxs:
            if ax.type is "value":
                model_report = "".join([model_report,
                                        "\n{0} = {1}".format(ax.name,
                                                             ax.value)])
            else:
                model_report = "".join([model_report,
                                        "\n{0} = GRAPH({1})".format(ax.name,
                                                                    ax.eqn),
                                        ", ".join([str(x)
                                                   for x in ax.series])])

        # Metadata
        flows = set([])
        for flw in self.stocks:
            if hasattr(flw, "inflow_name"):
                flows.add(flw.inflow_name)
            if hasattr(flw, "outflow_name"):
                flows.add(flw.outflow_name)

        graphs, constants = 0, 0
        for cst in self.auxs:
            if "value" in cst.type:
                constants += 1
            if "series" in cst.type:
                graphs += 1

        num_var = len(self.stocks) + len(flows) + len(self.auxs)
        meta_data = "".join([
                "{ The model has ", str(num_var), " (",
                str(num_var), ") variables (array ",
                "expansion in parens).\n", "  In root model and 0",
                " additional modules with 0 sectors.\n", "  Stocks: ",
                str(len(self.stocks)), " (", str(len(self.stocks)),
                "), Flows: ", str(len(flows)), " (",
                str(len(flows)), "), Converters: ",
                str(len(self.auxs)), " (", str(len(self.auxs)), ")\n",
                "  Constants: ", str(constants), " (", str(constants), "),",
                " Equations: ", str(len(self.stocks)), " (",
                str(len(self.stocks)), "),",
                " Graphicals: ", str(graphs), " (", str(graphs), ")}\n"])

        model_report = "".join([model_report, meta_data])
        return model_report


# %% Parsers
def xmile_parser(xmile_soup):
    # Grammar deffinitions for stocks, flows and aux
    stock_grammar = r"""
        entry =  line+
        line = ("<stock" (ws stock_disp_name)? ws
                    "name=" qtm stock_name qtm ">")? nl?
                    (ws? "<eqn>" nl)? (ws val_ini nl)? (ws "</eqn>" nl)?
                    (ws? "<label>" nl ws label nl ws "</label>")?
                    (ws? "<inflow>"  nl ws inflow  nl ws "</inflow>" nl)?
                    (ws? "<outflow>" nl ws outflow nl ws "</outflow>" nl)?
                    (ws? "<non_negative>" nl ws "</non_negative>")?
                    (ws "<units>" nl ws units nl ws "</units>" nl)?
               ("</stock>" nl)?
        stock = "stock"
        stock_name = ~"[A-z]*"
        stock_disp_name = "isee:display_name=" qtm str qtm
        qtm = '"'
        val_ini = ~"[0-9.]*"
        inflow = ~"[A-z]*"
        val = ~"[0-9.]*"
        units = ~"[A-z]*"
        label = ~"[A-z_ ]*"
        outflow = ~"[A-z]*"
        str = ~"[A-z0-9_]*"
        ws = ~"\s"*
        nl = ~"\n"
        """

    flow_grammar = r"""\
        entry =  line+
        line = ("<flow" (ws flow_disp_name)? ws
                        "name=" qtm flow_name qtm ">")? nl?
                   (ws? "<eqn>" nl)? (ws eqn nl)? (ws "</eqn>" nl)?
                   (ws? "<non_negative>" nl ws "</non_negative>")?
                   (ws "<units>" nl ws units nl ws "</units>" nl)?
               ("</flow>" nl)?
        flow_disp_name = "isee:display_name=" qtm str qtm
        flow_name = ~"[A-z]*"
        eqn = ~"[A-z0-9.*/]*"
        val = ~"[0-9.]*"
        units = ~"[A-z/]*"
        str = ~"[A-z0-9_]*"
        qtm = '"'
        ws = ~"\s"*
        nl = ~"\n"
        """

    aux_grammar = r"""
        entry =  line+
        line = ("<aux" ws (aux_disp_name ws)? "name=" qtm aux_name qtm ">" nl)?
                   (ws "<eqn>" nl ws eqn_str nl ws "</eqn>" nl)?
                   (ws "<units>" nl ws units nl ws "</units>" nl)?
                   (ws "<gf>" nl)?
                       (ws "<xscale max=" qtm max qtm ws
                                   "min=" qtm min qtm ">" nl)?
                       (ws "</xscale>" nl)?
                       (ws "<yscale max=" str ws "min=" str ">" nl
                        ws "</yscale>" nl)?
                       (ws "<ypts>" nl ws ypts nl ws "</ypts>" nl)?
                   (ws "</gf>" nl)?
               (ws "</aux>" nl)?
        aux_disp_name = "isee:display_name=" str
        min = ~"[0-9.]*"
        max = ~"[0-9.]*"
        aux_name = ~"[A-z _]*"
        eqn_str = ~"[A-z0-9.*/]*"
        ypts = ~"[0-9.,-]*"
        units = ~"[A-z/ ]*"
        str = ~"[A-z0-9\"_]*"
        qtm = '"'
        ws = ~"\s*"
        nl = ~"\n"
        """

    # Visitor classes
    class StockParser(NodeVisitor):
        def __init__(self, grammar, text):
            self.entry = {}
            ast = Grammar(grammar).parse(text)
            self.visit(ast)

        def visit_stock_name(self, n, vc):
            self.entry['stock_name'] = n.text

        def visit_val_ini(self, n, vc):
            self.entry['val_ini'] = n.text

        def visit_inflow(self, n, vc):
            self.entry['inflow'] = n.text

        def visit_outflow(self, n, vc):
            self.entry['outflow'] = n.text

        def visit_units(self, n, vc):
            self.entry['units'] = n.text

        def generic_visit(self, n, vc):
            pass

    class FlowParser(NodeVisitor):
        def __init__(self, grammar, text):
            self.entry = {}
            ast = Grammar(grammar).parse(text)
            self.visit(ast)

        def visit_flow_name(self, n, vc):
            self.entry['flow_name'] = n.text

        def visit_eqn(self, n, vc):
            self.entry['eqn'] = n.text

        def visit_units(self, n, vc):
            self.entry['units'] = n.text

        def generic_visit(self, n, vc):
            pass

    class AuxParser(NodeVisitor):
        def __init__(self, grammar, text):
            self.entry = {}
            ast = Grammar(grammar).parse(text)
            self.visit(ast)

        def visit_aux_name(self, n, vc):
            self.entry['aux_name'] = n.text

        def visit_eqn_str(self, n, vc):
            self.entry['eqn_str'] = n.text

        def visit_units(self, n, vc):
            self.entry['units'] = n.text

        def visit_min(self, n, vc):
            self.entry['x_min'] = n.text

        def visit_max(self, n, vc):
            self.entry['x_max'] = n.text

        def visit_ypts(self, n, vc):
            self.entry['ypts'] = n.text

        def generic_visit(self, n, vc):
            pass

    # Extract equations block located in model/variables section of xmile file
    # First get flow equations
    flows = {}
    for flw in xmile_soup.variables.find_all("flow"):
        flw_parse = FlowParser(flow_grammar, flw.prettify()).entry
        name = flw_parse["flow_name"].replace(" ", "_")
        flows[name] = flw_parse

    # Extract stock collection attributes
    stocks_cls = []
    for stk in xmile_soup.variables.find_all("stock"):
        stock_parse = StockParser(stock_grammar, stk.prettify()).entry
        if stk.inflow:
            stock_parse["inflow"] = flows[stock_parse["inflow"]]
        if stk.outflow:
            stock_parse["outflow"] = flows[stock_parse["outflow"]]
        stocks_cls.append(XmileStock(stock_parse))

    # Get aux variables
    aux_dict = []
    for ax in xmile_soup.variables.find_all("aux"):
        aux_parse = AuxParser(aux_grammar, ax.prettify()).entry
        aux_dict.append(XmileAux(aux_parse))

    return {"stocks": stocks_cls, "auxs": aux_dict}


# %% Read xmile model (like stella 10 onwards)
if __name__ == '__main__':
    main()
