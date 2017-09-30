# -*- coding: utf-8 -*-
"""
@author: Miguel Equihua
email: equihuam@gmail.com
Location: Instituto de Ecología, AC (Xalapa, Veracruz México)
Version: 0.1

Use: xmile2py <xmile file> <output directory>
History
--------
Created on Sun Aug 27, 2017
First fully operational version Thursday 7 Sep, 2017

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
        print("\t -- outname is the name of a directory where" +
              " the main translated model, data files and other" +
              " functions will be stored\n")
        sys.exit()

    # exctract data from command line input
    model_file = sys.argv[1]
    outputName = sys.argv[2]
    outPath = os.path.abspath(outputName)

    # Prepare directory to store output
    if (not os.path.isdir(outPath)):
        os.mkdir(outPath)

    # Extract relevant information to build model
    model_attributes = xmile_parser(model_file)
    model_translation = XmileModel(model_attributes["model_spec"],
                                   model_attributes["stocks"],
                                   model_attributes["auxs"])

    # output model file
    with open(outPath + "/" + model_translation.name + ".txt", "w") as foutput:
        model_output = model_translation.show()
        foutput.writelines(model_output)

    with open(outPath + "/" + model_translation.name + ".R", "w") as foutput:
        model_output = model_translation.build_R_script()
        foutput.writelines(model_output)

    print("\n\n" + "*"*28 + "\nModel translation successful")
    print("Model processed:", model_translation.name)
    print("Translation can be found at:\n  ", outPath)
    print("\n\n")


# %%Model classes
class XmileStock:
    """Class that stores Stock definition
       as translated from an *xmile* model specification

    Attributes
    ----------
    :name: -- Name of the stock
    :units: -- Units of measure for the stock contents
    :eqn: --  Equation describing stock dynamics
    :init: -- Initial condition of the stock
    :inflow: -- List of input flow attributes (eqn, flow_name and units)
    :outflow: -- List of output flow attributes (eqn, flow_name and units)

    Methods
    ---------
    :__init__ -- Takes a stock dictionary as its only parameter (no default)
    :show: -- Display the main features of the stock instance.
    """
    def __init__(self, stock_dict):
        self.name = stock_dict["stock_name"]
        self.units = stock_dict["units"]
        self.eqn = ""
        if "inflow" in stock_dict:
            self.inflow = stock_dict["inflow"]
            self.eqn = " + ".join([f["flow_name"] for f in self.inflow])
        if "outflow" in stock_dict:
            self.outflow = stock_dict["outflow"]
            self.eqn = self.eqn + " - " +\
                " - ".join([f["flow_name"] for f in self.outflow])
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
    """Class that stores converters deffinition
       as translated from an original *xmile* model specification

    Attributes
    ----------
    :name: -- Name of the converter.
    :type: -- Either value if holding a constant or a data series if holding
              a *graphical function*
    :units: -- Units of measure for the converter constant.
    :eqn: -- If converter holds a *graphical function*, then this holds an
             equation related to that function.
    :value: -- If converter is a constant this is its value.

    Methods
    ---------
    :__init__: -- Takes an aux dictionary as its only parameter (no default)
    :show: -- Display the main features of the converter instance.
    """
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
        if self.type == "value":
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
    """Class that stores full model deffinition
       as translated from an original stella or *xmile* compliant
       model specification

    Attributes
    ----------
    :name: -- Name of the model.
    :step: -- Time step to run the model.
    :start: -- Starting point of the time frame to run the model.
    :stop: -- Ending point of the time frame to run the model.
    :stocks: -- List of *stock class* objects.
    :auxs: -- List of *aux class* objects.
    :units: -- Units of measure for the converter constant.
    :eqn: -- If converter holds a *graphical function*, then this holds an
             equation related to that function.
    :value: -- If converter is a constant this is its value.

    Methods
    ---------
    :__init__: -- Assemble full model descriptor from a spec dictionary, a
                  list of stock class objects and a list of aux class
                  objects (no default).
    :show: -- Display the main features of the converter instance.
    :build_R_script: -- Prepares an scrip for R use with **deSolve** library.
    """
    def __init__(self, spec, stocks, auxs):
        """Takes an spec dictionary with model data, a list of stock class
           and a list of aux class objects that fully describe the original
           Stella or *xmile* model(no default)
        """
        self.name = spec["name"]
        self.step = spec["step"]
        self.start = spec["start"]
        self.stop = spec["stop"]
        self.stocks = stocks
        self.auxs = auxs

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

    def build_R_script(self):
        """Prepares an scrip for R use with **deSolve** library.
        """
        # Initialization section
        r_script = "".join(["# Prepare libraries needed\n"
                            "if (require(deSolve) == F) \n{\n",
                            "    tall.packages('deSolve', ",
                            "repos='http://cran.r-project.org')\n",
                            "    if (require(deSolve) == F)\n",
                            "      print ('Error: deSolve is not installed",
                            " on your machine')\n",
                            "}\n\n",
                            "if (require(minpack.lm) == F) \n{\n",
                            "    tall.packages('minpack.lm', ",
                            "repos='http://cran.r-project.org')\n",
                            "    if (require(minpack.lm) == F)\n",
                            "      print ('Error: minpack.lm is not installed",
                            " on your machine')\n",
                            "}\n\n"])

        # Definition of main model function
        r_script = "".join([r_script,
                            "# Function specifying full model\n",
                            "model <- function(t, Y, ",
                            "parameters, ...) \n{\n",
                            "    # Time and other model variables\n",
                            "    Time <<- t\n"])

        stk_names = [s.name for s in self.stocks]
        stk_names.sort()
        lines = "".join(["    {0} <- Y['{0}']\n".format(stk)
                         for stk in stk_names])
        r_script = "".join([r_script, lines, "\n    # Model parameters"])

        convertors = [a.name for a in self.auxs]
        convertors.sort()
        lines = "".join(["    {0} <- parameters['{0}']\n".format(aux)
                         for aux in convertors])
        r_script = "".join([r_script, "\n", lines, "\n"])

        # Recover flow information and prepare suitable R commands
        flw_names = set()
        for stk in self.stocks:
            if hasattr(stk, "inflow"):
                flw_names = flw_names.union(
                          {"\n    " + f["flow_name"] + " <- " + f["eqn"]
                           for f in stk.inflow})
            if hasattr(stk, "outflow"):
                flw_names = flw_names.union(
                          {"\n    " + f["flow_name"] + " <- " + f["eqn"]
                           for f in stk.outflow})

        lines = "".join(flw_names)
        r_script = "".join([r_script, "    # flow equations", lines, "\n\n",
                            "    # Differential equations"])

        # diferential equation specification
        lines, d_func = "", []
        for deqn in self.stocks:
            lines = "".join([lines,
                             "d_", deqn.name, " <- ",
                             deqn.eqn, "\n    "])
            d_func.append("d_" + deqn.name)

        d_func = "    list(c(" + ", ".join(d_func) + "))\n}\n"

        r_script = "".join([r_script, "\n    ", lines, "\n",
                            "    # Model output list\n",
                            d_func,
                            "\n", "#" * 50])

        # Define R object to store model parameters
        series, items = [], []
        for ax in self.auxs:
            if ax.type == "value":
                items.append("".join(["{0} = {1}".format(ax.name,
                                                          str(ax.value))]))
            if ax.type == "series":
                series.append("".join(series) + ax.name +
                              " <- data.frame(matrix(c(" +
                              ", ".join(["c(" + str(x) + ", " + str(y) + ")"
                                         for (x, y) in ax.series]) +
                              "), ncol = 2, byrow = T))\n" +
                              "names(" + ax.name + ") <- c(" + "'" +
                              ax.eqn.lower() + "'" + ", '" + ax.name + "')\n")

        if items:
            r_script = "".join([r_script, "\n\n",
                                "# Paremeters and initial condition to" +
                                " solve model\n", "parms <- c(" +
                                ", ".join(items) + ")\n"])
        if series:
            r_script = "".join([r_script, "".join(series) + "\n"])

        # Includes initial conditions for the stocks
        items = []
        for stk in self.stocks:
            items.append(stk.name + " = " + str(stk.init))

        # Assemble command call to run the model in R
        lines = "Y <- c(" + ", ".join(items) + ")\n\n"
        lines = "".join([lines, "# Call for auxiliary components\n",
                         "source('", self.name, "_r_functions.R')\n\n",
                         "# Numerical integration specs\n"
                         "DT <- ", str(self.step), "\n",
                         "time <- seq(", str(self.start), " ,",
                         str(self.stop), " , DT)\n\n",
                         "# Use 'ode' function to solve model\n",
                         "out <- ode(func = model, y = Y, times = time,",
                         " parms = parms, method = 'rk4')\n\n",
                         "# Plot model numerical solution\n"
                         "plot(out)\n\n",
                         "# Error function to fit model with 'nls.lm'\n",
                         "ssq <- function(parms, t, data, y0, varList)\n{",
                         "    # solve ODE for a given set of parameters\n",
                         "    sol <- ode(func = model, y = y0, times = t,",
                         " parms = parms, method = 'rk4')\n\n",
                         "    # Match data to model output\n",
                         "    solDF <- data.frame(sol)\n",
                         "    solDF <- solDF[solDF$time %in% data$time, ]\n\n",
                         "    # Difference fitted vs observed \n",
                         "    solDF <- unlist(solDF[, varList])\n",
                         "    obsDF <- unlist(data[, -1])\n",
                         "    ssqres <- solDF - obsDF\n\n",
                         "    # return predicted vs experimental residual\n",
                         "    return(ssqres)\n}\n\n",
                         "# Provide data, dataset and time sequence\n\n",
                         "# parameter fitting using levenberg marquart",
                         " algorithm. Provide parms as initial guess\n",
                         "fittedModel <- nls.lm(par=parms, fn=ssq, y0 = Y, ",
                         "t = time, data = datos, varList = varList)\n",
                         "summary(fittedModel)\n",
                         "fittedVals <- ode(func = model, y = Y, ",
                         "times = time, parms = fittedModel$par, ",
                         "method = 'rk4')\n\n"])

        r_script = "".join([r_script, lines])

        return r_script

    def show(self):
        """Display the main features of the full model.
        """

        items = []
        for stk in self.stocks:
            eqn = stk.name + "(t)" + " = " + stk.name + " (t - dt)"
            fls = ""
            if hasattr(stk, "inflow"):
                fls = " + ".join([f["flow_name"] for f in stk.inflow])
            if hasattr(stk, "outflow"):
                fls = fls + "- " + " - ".join(
                        [f["flow_name"] for f in stk.outflow])
            item = [eqn + " + (" + fls + ") * dt"]
            item.extend(["    INIT " + stk.name + " = " + str(stk.init)])

            # Define flows
            if hasattr(stk, "inflow"):
                item.extend(["    INFLOWS:"])
                item.extend(["        " + f["flow_name"] + " = " +
                             f["eqn"] for f in stk.inflow])
            if hasattr(stk, "outflow"):
                item.extend(["    OUTFLOWS:"])
                item.extend(["        " + f["flow_name"] + " = " +
                             f["eqn"] for f in stk.outflow])
            items.append("\n".join(item))

        model_report = "".join(["Top-Level Model <{0}>:\n".format(self.name),
                                "\n".join(items)])

        for ax in self.auxs:
            if ax.type == "value":
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
        flows = set()
        for flw in self.stocks:
            if hasattr(flw, "inflow"):
                flows.update({f["flow_name"] for f in flw.inflow})
            if hasattr(flw, "outflow"):
                flows.update({f["flow_name"] for f in flw.outflow})

        graphs, constants = 0, 0
        for cst in self.auxs:
            if "value" in cst.type:
                constants += 1
            if "series" in cst.type:
                graphs += 1

        num_var = len(self.stocks) + len(flows) + len(self.auxs)
        meta_data = "".join([
                "\n{ The model has ", str(num_var), " (",
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
def xmile_parser(model_file):
    """ Parse a model from Stella
        as used from version 10 and up (type "\*.stmx") or some other
        *xmile* compliant file type. It uses **BeautifullSoup** and
        **parsimonious** to implement the parsing of *xmile* format.

    Keyword arguments:
    ----------
    :model_file: -- Name of a model file of an **xmile** compliant
                    type (no default)
    """
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
                    (ws? "<units>" nl ws units nl ws "</units>" nl)?
                    (ws? "<display" skip nl)?
                    (ws? "<label_side>" nl ws str nl ws "</label_side>" nl)?
                    (ws? "<label_angle>" nl ws str nl ws "</label_angle>" nl)?
                    (ws? "</display>" nl)?
               ("</stock>" nl)?
        stock = "stock"
        stock_name = ~"[A-Za-z0-9$_ ]*"
        stock_disp_name = "isee:display_name=" qtm str qtm
        qtm = '"'
        val_ini = ~"[0-9.]*"
        inflow = ~"[A-Za-z0-9$_ ]*"
        val = ~"[0-9.]*"
        units = ~"[A-z]*"
        label = ~"[A-z_ ]*"
        outflow = ~"[A-Za-z0-9$_ ]*"
        str = ~"[A-z0-9_]*"
        skip = ~"."*
        ws = ~"\s"*
        nl = ~"\n"
        """

    flow_grammar = r"""
        entry =  line+
        line = ("<flow" (ws flow_disp_name)? ws
                        "name=" qtm flow_name qtm ">")? nl?
                   (ws? "<eqn>" nl)? (ws eqn nl)? (skip nl)?
                   (ws? "<non_negative>" nl? ws? "</non_negative>")?
                   (ws? "<units>" nl ws units nl ws "</units>" nl)?
                   (ws? "<display" skip nl)?
                   (ws? "<label_side>" nl ws str nl ws "</label_side>" nl)?
                   (ws? "<label_angle>" nl ws str nl ws "</label_angle>" nl)?
                   (ws? "<pts>" ws? nl)?
                   (ws? "<pt x=" skip nl)?
                   (ws? "</pt>" ws? nl)?
                   (ws? "</pts>" ws? nl)?
                   (ws? "</display>" ws? nl)?
               ("</flow>" nl)?
        flow_disp_name = "isee:display_name=" qtm str qtm
        flow_name = ~"[A-Za-z0-9$_ ]*"
        eqn = ~"[A-z0-9.*/\(\)_\+ ]*"
        val = ~"[0-9.]*"
        units = ~"[A-z/]*"
        str = ~"[A-z0-9_]*"
        qtm = '"'
        skip = ~"."*
        ws = ~"\s"*
        nl = ~"\n"
        """

    aux_grammar = r"""
        entry =  line+
        line = ("<aux" ws (aux_disp_name ws)? "name=" qtm aux_name qtm ">" nl)?
                   (ws? "<eqn>" nl ws eqn_str nl ws "</eqn>" nl)?
                   (ws? "<units>" nl ws units nl ws "</units>" nl)?
                   (ws? "<display" skip nl)?
                   (ws? "<format" skip nl)?
                   (ws? "</format>" nl)?
                   (ws? "<gf>" nl)?
                       (ws "<xscale max=" qtm max qtm ws
                                   "min=" qtm min qtm ">" nl)?
                       (ws? "</xscale>" nl)?
                       (ws? "<yscale max=" str ws "min=" str ">" nl
                        ws? "</yscale>" nl)?
                       (ws? "<ypts>" nl ws ypts nl ws "</ypts>" nl)?
                   (ws? "</gf>" nl)?
                   (ws? "</display>" skip? nl)?
               (ws? "</aux>" skip? nl)?
        aux_disp_name = "isee:display_name=" str
        min = ~"[0-9.]*"
        max = ~"[0-9.]*"
        aux_name = ~"[A-Za-z0-9$_ ]*"
        eqn_str = ~"[A-z0-9.*/\(\)]*"
        ypts = ~"[0-9.,-]*"
        units = ~"[A-z/ ]*"
        str = ~"[A-z0-9\"_]*"
        qtm = '"'
        skip = ~"."*
        ws = ~"\s"*
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

    # Read model file
    with open(model_file, encoding="utf-8") as fmodel:
        xmile_soup = bs(fmodel, "lxml")

    # Extract relevant information to build model
    model_spec = {"name": xmile_soup.header.find("name").text,
                  "step": float(xmile_soup.find("dt").text),
                  "start": int(xmile_soup.find("start").text),
                  "stop": int(xmile_soup.find("stop").text)}
    if model_spec["name"] == "":
        model_spec["name"] = model_file.split(".")[0]

    # Extract equations block located in model/variables section of xmile file
    # First get flow equations
    flows = {}
    for flw in xmile_soup.model.find_all("flow"):
        if "eqn" in flw.prettify():
            flw_parse = FlowParser(flow_grammar, flw.prettify()).entry
            if "units" not in flw_parse:
                flw_parse["units"] = ""
            name = flw_parse["flow_name"].replace(" ", "_")
            name = name.replace("á", "a").replace("é", "e").replace("í",
                                "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
            flw_parse["flow_name"] = name
            flows[name] = flw_parse

    # Extract stock collection attributes
    stocks_cls = []
    for stk in xmile_soup.model.find_all("stock"):
        if "eqn" in stk.prettify():
            stock_parse = StockParser(stock_grammar, stk.prettify()).entry
            name = stock_parse["stock_name"].replace(" ", "_")
            stock_parse["stock_name"] = name
            if "units" not in stock_parse:
                stock_parse["units"] = ""
            if stk.inflow:
                stock_parse["inflow"] = [flows[f.text] for f in
                                         stk.find_all("inflow")]
            if stk.outflow:
                stock_parse["outflow"] = [flows[f.text] for f in
                                          stk.find_all("outflow")]
            stocks_cls.append(XmileStock(stock_parse))

    # Get aux variables
    aux_list = []
    for ax in xmile_soup.model.find_all("aux"):
        if "eqn" in ax.prettify():
            aux_parse = AuxParser(aux_grammar, ax.prettify()).entry
            name = aux_parse["aux_name"].replace(" ", "_")
            aux_parse["aux_name"] = name
            if "units" not in aux_parse:
                aux_parse["units"] = ""
            aux_list.append(XmileAux(aux_parse))

    return {"model_spec": model_spec, "stocks": stocks_cls, "auxs": aux_list}


# %% Read xmile model (like stella 10 onwards)
if __name__ == '__main__':
    main()
