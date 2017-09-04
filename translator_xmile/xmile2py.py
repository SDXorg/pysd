# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 23:45:45 2017

@author: Miguel
"""

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from bs4 import BeautifulSoup as bs
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: xmile2py <input_stella.stmx> <outname>\n")
        print("\t -- input_stella.stmx is a xml variant text file " +
              " use in STELLA models from version 10 and up\n")
        print("\t -- outname is the name of R project where the main R" +
              " script (outname.R) as well as data files and other" +
              " functions are stored\n")
        sys.exit()

    fname1 = sys.argv[1]
    outputName = sys.argv[2]
    outPath = outputName
    outName_split = re.split('(\\\|/)', outputName)
    if len(outName_split) > 1:
        outputName = outName_split[-1]

    # Metadata
    num_var = len(stocks) + len(flows) + len(aux)
    uses_arrays = int(soup.header.smile.attrs["uses_arrays"])
    meta_data = "".join([
            "{ The model has ", str(num_var), " (",
            str(num_var * uses_arrays), ") variables (array ",
            "expansion in parens).\n", "  In root model and 0",
            " additional modules with 0 sectors.\n", "  Stocks ",
            str(len(stocks)), " (", str(len(stocks) * uses_arrays),
            "), Flows: ", str(len(flows)), " (",
            str(len(flows) * uses_arrays), "), Converters: ",
            str(len(aux)), " (", str(len(aux) * uses_arrays), ")\n}"])

    print("Top-levelmodel:")
    for st in stocks:
        print("\n".join(st))
    print("\n".join(aux))
    print(meta_data)


# %%Model classes

class stock:
    def __init__(self, stock_dict):
        self.name = stock_dict["stock_name"]
        self.eqn = ""
        if "inflow" in stock_dict:
            self.inflow = stock_dict["inflow"]["eqn"]
            self.eqn = self.eqn + stock_dict["inflow"]["eqn"]
        if "outflow" in stock_dict:
            self.outflow = stock_dict["outflow"]["eqn"]
            self.eqn = self.eqn  + " - " + stock_dict["outflow"]["eqn"]
        self.init = float(stock_dict["val_ini"])

    def show(self):
        print("Stock")
        print("name:", self.name)
        print("Equation: ", self.eqn)
        print("Init value: ", self.init)

class aux:
    def __init__(self, aux_dict):
        self.name = aux_dict["aux_name"]
        if "ypts" in aux_dict:
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
            self.type = "value"


class meta_data:
    def __init__(self, stock_dict, aux):


# %% Parsers
def xmile_parser(xmile_file):
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

        def visit_eqn(self, n, vc):
            self.entry['eqn'] = n.text

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
    for flw in soup.variables.find_all("flow"):
        flw_parse = FlowParser(flow_grammar, flw.prettify()).entry
        name = flw_parse.pop("flow_name").replace(" ", "_")
        flows[name] = flw_parse

    stocks, stocks_dict = [], []
    for stk in soup.variables.find_all("stock"):
        stock_parse = StockParser(stock_grammar, stk.prettify()).entry
        if stk.inflow:
            stock_parse["inflow"] = flows[stock_parse["inflow"]]
        if stk.outflow:
            stock_parse["outflow"] = flows[stock_parse["outflow"]]
        stocks_dict.append(stock_parse)

        eqn = stk.attrs["name"] + "(t)" + " = " + stk.attrs["name"] +\
            " (t - dt)"
        fls = ""
        if stk.inflow:
            fls = stk.inflow.text
        if stk.outflow:
            fls = fls + " - " + stk.outflow.text
        item = [eqn + " + (" + fls + ") * dt"]
        item.extend(["    INIT " + stk.attrs["name"] + " = " + stk.eqn.text])

        # Define flows
        if stk.inflow:
            item.extend(["    INFLOWS:"])
            item.extend(["        " + flows[stk.inflow.text]["eqn"]])
        if stk.outflow:
            item.extend(["    OUTFLOWS:"])
            item.extend(["        " + flows[stk.outflow.text]["eqn"]])

        stocks.append(item)

        print(StockParser(stock_grammar, stk.prettify()).entry)

    # Get aux variables
    aux_dict, aux = [], []
    for ax in soup.variables.find_all("aux"):
        aux_parse = AuxParser(aux_grammar, ax.prettify()).entry
        aux_dict.append(aux_parse)

        if ax.gf:
            rge = float(ax.xscale["max"]) - float(ax.xscale["min"])
            gf = ax.gf.ypts.text.split(",")
            step = rge / (len(gf) - 1)
            i, series = float(ax.xscale["min"]), []
            for g in gf:
                g_xy = "(" + str(float(i)) + ", " + str(float(g)) + ")"
                series.extend([g_xy])
                i += step
            aux.append(ax.attrs["name"] + " = GRAPH(" + ax.eqn.text + ")\n" +
                       ", ".join(series))

    return {"flows": flows, "stocks": stocks_dict, "auxs": aux_dict}

# %% Read xmile model (like stella 10 onwards)
if __name__ == '__main__':
    main()
    with open("SIR_HK (stella 10).stmx") as fp:
        soup = bs(fp, "lxml")

attributs = xmile_parser (soup)
attributs["auxs"][2]

stk1 = stock (attributs["stocks"][0])
stk1.show()

ax1 = aux(attributs["auxs"][2])
ax1.name
ax1.series
if hasattr(ax1, 'units'):
   ax1.units
