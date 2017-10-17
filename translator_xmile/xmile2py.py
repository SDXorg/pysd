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
import re


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
    model_translation = xmile_parser(model_file)

    # output model file
    with open(outPath + "/" + model_translation.name + ".txt", "w") as foutput:
        model_output = model_translation.show()
        foutput.writelines(model_output)

    r_model_solver, r_model_calibrate = model_translation.build_R_script
    file_name = model_translation.name + "_solver.R"
    with open(outPath + "/" + file_name, "w") as foutput:
        foutput.writelines(r_model_solver)

    r_model_calibrate = r_model_calibrate.replace("file_path", '"' + file_name + '"')
    with open(outPath + "/" + file_name.replace("solver", "calibrate"), "w") as foutput:
        foutput.writelines(r_model_calibrate)

    print("\n\n" + "*" * 28 + "\nModel translation successful")
    print("Model processed:", model_translation.name)
    print("Translation can be found at:\n  ", outPath)
    print("\n\n")


# test for eqn containing a single number
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


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
        if "inflow" in stock_dict:
            self.inflow = stock_dict["inflow"]
            self.eqn = " + ".join([f["flow_name"] for f in self.inflow])
        if "outflow" in stock_dict:
            self.outflow = stock_dict["outflow"]
            self.eqn = self.eqn + " - " + \
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
    :type: -- Either "value" if holding a constant or a "series" if holding
              a *graphical function* or an "equation"
    :is_parameter: -- Logic value to indicate it is used within equations
    :units: -- Units of measure for the converter constant.
    :value: -- If converter is a constant this is its value.
    :eqn: -- If converter is *graphical function*, then this holds an
             equation related to that function.

    Methods
    ---------
    :__init__: -- Takes an aux dictionary as its only parameter (no default)
    :show: -- Display the main features of the converter instance.
    """

    def __init__(self, aux_dict):
        self.name = aux_dict["aux_name"]
        self.is_parameter = aux_dict["is_parameter"]
        if "ypts" in aux_dict:
            self.eqn = aux_dict["eqn"]
            self.type = "series"
            self.x_max = aux_dict["x_max"]
            self.x_min = aux_dict["x_min"]
            step = (float(self.x_max) - float(self.x_min)) / (len(aux_dict["ypts"].split(",")) - 1)
            yval = list(map(float, (aux_dict["ypts"].split(","))))
            self.series = list(map(lambda x, y: (x, float(y)), [float(self.x_min) + step * x
                                                                for x in range(0, len(yval))],
                                   aux_dict["ypts"].split(",")))
        else:
            self.units = aux_dict["units"]
            if isfloat(aux_dict["eqn"]):
                self.type = "value"
                self.value = aux_dict["eqn"]
            else:
                self.type = "equation"
                self.eqn = aux_dict["eqn"]

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
    """Class that stores full model definition
       as translated from an original stella or *xmile* compliant
       model specification

    Attributes
    ----------
    :name: -- Name of the model.
    :step: -- Time step to run the model.
    :start: -- Starting point of the time frame to run the model.
    :stop: -- Ending point of the time frame to run the model.
    :stocks_set: -- Collection of stock names in the model
    :flows_set: -- Collection of flow function names in the model
    :auxs_set: -- Collection of all converter names in the model
    :parameters_set -- Collection of variable names used within equations
    :stocks: -- List of *stock class* objects.
    :auxs: -- Dictionary by name of *aux class* objects.
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
        self.stocks_set = set()
        self.flows_set = set()
        self.auxs_set = set()
        self.parameters_set = set()
        self.eqn = ""
        self.stocks = stocks
        self.auxs = auxs

    def build_R_script(self):
        """Prepares an scrip for R use with **deSolve** library.
        """
        import textwrap as tw


                # Translate Stella function names into R version
        Functions_R = {'EXP': "exp", 'MIN': "min", 'MAX': "max", 'MEAN': "mean", 'SUM': "sum",
            'ABS': "abs", 'SIN': "sin", 'COS': "cos", 'TAN': "tan", 'LOG10': "log10",
            'LOGN': 'log', 'SQRT': "sqrt", 'ROUND': "round", 'ARCTAN': 'atan', 'TIME': 't',
            'PI': "pi", 'INT': 'floor'}
        Time_independet = {"RANDOM" : "random"}

        # Initialization section
        r_script_slv = "".join(["# Prepare libraries needed\n"
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
        r_script_slv = "".join([r_script_slv,
                                "# Function fully specifying the translated model for R use\n",
                                "model <- function(t, Y, ", "parameters, ...) \n{\n",
                                "    # Time and other model variables\n", "    Time <<- t\n"])

        # Stock values given as input for each time step
        stk_names = [s.name for s in self.stocks]
        stk_names.sort()
        lines = "".join(["    {0} <- Y['{0}']\n".format(stk) for stk in stk_names])
        r_script_slv = "".join([r_script_slv, lines, "\n    # Model parameters"])

        # Model parameters specified as single values
        converters = [a.name for k, a in self.auxs.items() if a.is_parameter and a.type == "value"]
        converters.sort()

        lines = "".join(["    {0} <- parameters['{0}']\n".format(aux) for aux in converters])
        r_script_slv = "".join([r_script_slv, "\n", lines, "\n"])

        # Flow information necessary to prepare suitable R commands
        flw_equations = set()
        for stk in self.stocks:
            if hasattr(stk, "inflow"):
                flw_equations = flw_equations.union({"\n    " + f["flow_name"] + " <- " +
                                                     f["eqn"] for f in stk.inflow})
            if hasattr(stk, "outflow"):
                flw_equations = flw_equations.union({"\n    " + f["flow_name"] + " <- " +
                                                     f["eqn"] for f in stk.outflow})

        #TODO convert into function? identifying supporting functions needed
        #  Collects function names needed for supporting functions
        supporting_eqn = {}
        for eqn in flw_equations:
            eqn_items = re.split(r"[*+/\-\n< ]", eqn)
            eqn_items = [e for e in eqn_items if e != "" and e != ","]
            for eqn_i in eqn_items:
                if eqn_i in self.parameters_set:
                    if self.auxs[eqn_i].type == "series":
                        supporting_eqn.update({eqn_i : eqn_i + "_lkp(t) "})
                    elif self.auxs[eqn_i].type == "equation":
                        supporting_eqn.update({eqn_i : eqn_i + "(t) "})

        calibration_data = set()
        for ax_k, ax_v in self.auxs.items():
            if ax_v.type == "equation":
                eqn_items = re.split(r"[(){}\[*+/-]", ax_v.eqn)
                eqn_items = [e for e in eqn_items if e != "" and e != ","]
                for eqn_i in eqn_items:
                    if eqn_i in Functions_R:
                        supporting_eqn.update({eqn_i: Functions_R[eqn_i]})
                    elif not re.findall("^[0-9)^]", eqn_i) and  not eqn_i in self.auxs_set and eqn_i != "":
                            supporting_eqn.update({eqn_i: eqn_i.lower()})
                    else:
                        if eqn_i in self.auxs:
                            if self.auxs[eqn_i].type == "equation":
                                supporting_eqn.update({eqn_i: eqn_i})
            else:
                if ax_v.type == "series" and not ax_v.is_parameter:
                    calibration_data.update({ax_k})


        for eqn_k, eqn_v in supporting_eqn.items():
            flw_equations = {re.sub(eqn_k, eqn_v, eqn) for eqn in flw_equations}

        # Equation formatting in the model function block
        flw_equations = {re.sub(r"(?<!<)([*+/-])", r" \1 ", eqn) for eqn in flw_equations}

        lines = "".join(sorted(flw_equations))
        r_script_slv = "".join([r_script_slv, "    # flow equations", lines, "\n\n",
                                "    # Differential equations"])

        # differential equation specification
        lines, d_func = "", []
        for d_eqn in self.stocks:
            lines = "".join([lines, "d_", d_eqn.name, " <- ", d_eqn.eqn, "\n    "])
            d_func.append("d_" + d_eqn.name)

        d_func = "    list(c(" + ", ".join(d_func) + "))\n}\n"

        r_script_slv = "".join([r_script_slv, "\n    ", lines, "\n", "    # Model output list\n",
                                d_func, "\n", "#" * 50, "\n\n"])

        # Define R object to store model parameters and supporting functions
        sup_funcs, series, items = [], [], []
        for ax_k, ax_v in self.auxs.items():
            if ax_v.type == "equation":
                # Translate xmile functions to R names from supporting_eqn keys
                new_eqn = ax_v.eqn
                for i_k, i_v in supporting_eqn.items():
                    if i_k in ax_v.eqn:
                        if i_k in Functions_R or i_k in Time_independet:
                            new_eqn = re.sub(i_k, i_v, new_eqn)
                        elif i_k + "(" in ax_v.eqn.replace(" ", ""):
                            new_eqn = re.sub(i_k + "\(", i_v + "(t,", new_eqn)
                        else:
                            new_eqn = re.sub(i_k, i_v + "(t)", new_eqn)

                # A little bit of equation formatting
                new_eqn = re.sub(r"([*+/-])", r" \1 ", new_eqn)
                new_eqn = re.sub(r"(,)", r"\1 ", new_eqn)
                new_eqn = re.sub("[ ]{2,}", " ", new_eqn)
                sup_funcs.append("".join([ax_v.name, " <- function(t)\n{\n    ", new_eqn, "\n}\n"]))

            if ax_v.is_parameter:
                if ax_v.type == "value":
                    items.append("".join([ax_v.name, " = ", ax_v.value]))
                if ax_v.type == "series":
                    if ax_v.eqn == "TIME":
                        sup_funcs.append("".join([ax_v.name, "_lkp <- function(t)\n{\n    ",
                                                  ax_v.name, "[floor((t - 1) %% ", ax_v.x_max, ") + 1, 2]\n}\n"]))
                    data_series = ", ".join(["c(" + str(x) + ", " + str(y) + ")" for (x, y) in ax_v.series])
                    if len(data_series) > 70:
                        data_series = "\n".join(tw.wrap(data_series, 80))
                    series.append(ax_v.name + " <- data.frame(matrix(c(" + data_series +
                                  "), ncol = 2, byrow = T))\n" + "names(" + ax_v.name + ") <- c(" + "'" +
                                  ax_v.eqn.lower() + "'" + ", '" + ax_v.name + "')\n")

        for fnc_def in supporting_eqn.keys() - self.auxs.keys() - Functions_R.keys():
            if fnc_def == "SINWAVE":
                sup_funcs.append("\nsinwave <- function(t, a, ph)\n{\n    " + "  a * sin(2 * t * pi / ph )" + "\n}\n")
            elif fnc_def == "RANDOM":
                sup_funcs.append("\nrandom <- function(min, max)\n{\n     runif(1, min, max)" + "\n}\n")
            else:
                sup_funcs.append("\nMissing definition for: " + fnc_def + "\n")

        if sup_funcs:
            r_script_slv = "".join([r_script_slv, "\n# Supporting functions\n", "\n".join(sup_funcs), "\n"])
        if series:
            r_script_slv = "".join([r_script_slv, "".join(series) + "\n"])
        if items:
            parms_str = "c(" + ", ".join(items) + ")"
            parms_str = parms_str[:70] + parms_str[70:].replace(", ", ",\n" + " "*9, 1)
            r_script_slv = "".join([r_script_slv, "# Paremeters and initial condition to solve model\n",
                                    "parms <- ", parms_str, "\n"])

        # Includes initial conditions for the stocks
        items = []
        for stk in self.stocks:
            items.append(stk.name + " = " + str(stk.init))

        # Assemble command call to run the model in R
        lines = "Y <- c(" + ", ".join(items) + ")\n\n"
        lines = "".join([lines, "# Call for auxiliary components\n",
                         "# source('", self.name, "_r_functions.R')\n\n",
                         "# Numerical integration specs\n"
                         "DT <- ", str(self.step), "\n",
                         "time <- seq(", str(self.start), " ,",
                         str(self.stop), " , DT)\n\n",
                         "# Use 'ode' function to solve model\n",
                         "out <- ode(func = model, y = Y, times = time,",
                         " parms = parms, method = 'rk4')\n\n",
                         "# Plot model numerical solution\n"
                         "plot(out)\n\n",
                         "par(mfrow=c(1,1))\n"])
        r_script_slv = "".join([r_script_slv, lines])

        #TODO Have to add provided calibration data from Stella model
        r_script_cal = "".join(["# Sckeleton for model calibration\n",
                                "# ********************************\n\n",
                                "# Error function required to fit the model with 'nls.lm'\n",
                                "ssq <- function(parms, t, data, y0, varList)\n{",
                                "    # solve ODE for a given set of parameters\n",
                                "    sol <- ode(func = model, y = y0, times = t,",
                                " parms = parms, method = 'rk4')\n\n",
                                "    # Match data to model output\n",
                                "    solDF <- data.frame(sol)\n",
                                "    solDF <- solDF[solDF$time %in% data$time, ]\n\n",
                                "    # Difference fitted vs observed \n",
                                "    solDF <- unlist(solDF[, varList])\n",
                                "    obsDF <- unlist(data[, varList])\n",
                                "    ssqres <- solDF - obsDF\n\n",
                                "    # return predicted vs experimental residual\n",
                                "    return(ssqres)\n}\n# ********************************\n\n\n",
                                "# Provide data, dataset and time sequence\n"])

        # Adding calibration data if available: series not in the parameters list
        series = []
        for d in calibration_data:
            ax = self.auxs[d]
            data_series = ", ".join(["c(" + str(x) + ", " + str(y) + ")" for (x, y) in ax.series])
            if len(data_series) > 70:
                data_series = "\n".join(tw.wrap(data_series, 80))

            series.append(ax.name +
                          " <- data.frame(matrix(c(" + data_series +
                          "), ncol = 2, byrow = T))\n" +
                          "names(" + ax.name + ") <- c(" + "'" +
                          ax.eqn.lower() + "'" + ", '" + ax.name + "')\n\n")

        if series:
            r_script_cal = "".join([r_script_cal, "".join(series) + "\n"])

        time = self.auxs[next(iter(calibration_data))].name + "$time"
        r_script_cal = "".join([r_script_cal,
                            "# include the names of all variables with calibration data\n",
                            "varList <- c("+ ", ".join(['"'+ d +'"' for d in sorted(self.stocks_set)]) +")\n\n",
                            "# Set-up data into a data.frame with time column. \n",
                            "# Verify correspondence between data and variable name in the model\n",
                            "data <- data.frame(time=", time, ",\n    ",
                            ",\n    ".join([d + "$" + d for d in sorted(calibration_data)]), ")\n\n",
                            "# Verify this varList for proper calibration data match\n",
                            "names(data) <- c(\"time\", varList)\n\n",
                            "# load translated model function\n",
                            "source (file_path)\n\n",
                            "# Fit model to calibration data using levenberg marquart\n",
                            "# algorithm. Please, provide parameter initial guess values\n",
                            "Y0 <- c(" + ", ".join(items) + ")\n",
                            "parm.0 <- " + parms_str + "\n",
                            "fittedModel <- nls.lm(par=parm.0, fn=ssq, y0 = Y0, ",
                            "t = time, data = data, varList = varList)\n\n",
                            "# Calibration results\n",
                            "summary(fittedModel)\n",
                            "parameters <- fittedModel$par  # store calibrated parameters\n\n",
                            "# Fitted values and plotting\n",
                            "fittedVals <- ode(func = model, y = Y, times = time, parms = parameters, ",
                            "method = 'rk4')\n\n",
                            "# Organizing fitted data in a data.frame\n",
                            "fitted.data.df <- data.frame(fittedVals)\n\n",
                            "# Plotting fitted & calibration data together\n"
                            "plot(fitted.data.df[, 1], fitted.data.df[, 2], type = 'l', col = 'blue', lwd=2)\n",
                            "points('<data to compare to fit>')  # Series to provide x, y data\n\n"])

        return r_script_slv, r_script_cal

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

        for ax_k, ax_v in self.auxs.items():
            if ax_v.type == "value":
                model_report = "".join([model_report, "\n", ax_v.name, " = ", ax_v.value, "\n"])
            elif ax_v.type == "equation":
                model_report = "".join([model_report, "\n", ax_v.name, " = ", ax_v.eqn, "\n"])
            else:
                model_report = "".join([model_report, "\n{0} = GRAPH({1})".format(ax_v.name, ax_v.eqn),
                                        ", ".join([str(x) for x in ax_v.series])])

        # Metadata
        flows = set()
        for flw in self.stocks:
            if hasattr(flw, "inflow"):
                flows.update({f["flow_name"] for f in flw.inflow})
            if hasattr(flw, "outflow"):
                flows.update({f["flow_name"] for f in flw.outflow})

        graphs, constants = 0, 0
        for cst_k, cst_v in self.auxs.items():
            if "value" in cst_v.type:
                constants += 1
            if "series" in cst_v.type:
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
                    (ws? "<scale " skip nl)?
                    (ws? "</scale>" nl)?
                    (ws? "<non_negative>" nl ws "</non_negative>")?
                    (ws? "<units>" nl ws units nl ws "</units>" nl)?
                    (ws? "<display" skip nl)?
                    (ws? "<label_side>" nl ws str nl ws "</label_side>" nl)?
                    (ws? "<label_angle>" nl ws str nl ws "</label_angle>" nl)?
                    (ws? "</display>" nl)?
               ("</stock>" nl)?
        stock = "stock"
        stock_name = ~"[A-Za-z0-9$_ áéíóúñ]*"
        stock_disp_name = "isee:display_name=" qtm str qtm
        qtm = '"'
        val_ini = ~"[0-9.]*"
        inflow = ~"[A-Za-z0-9$_ áéíóúñ]*"
        val = ~"[0-9.]*"
        units = ~"[A-z]*"
        label = ~"[A-z_ ]*"
        outflow = ~"[A-Za-z0-9$_ áéíóúñ]*"
        str = ~"[A-z0-9_]*"
        skip = ~"."*
        ws = ~"\s"*
        nl = ~"\n"
        """

    flow_grammar = r"""
        entry =  line+
        line = ("<flow" (ws flow_disp_name)? ws
                        "name=" qtm flow_name qtm ">")? nl?
                   (ws? "<eqn>" nl eqn? "</eqn>" nl)?
                   (ws? "<scale " skip nl)?
                   (ws? "</scale>" nl)?
                   (ws? "<non_negative>" nl? ws? "</non_negative>")?
                   (ws? "<units>" nl ws units nl ws "</units>" nl)?
                   (ws? "<display" skip nl)?
                   (ws? "<label_side>" nl? ws? str nl? ws? "</label_side>" nl?)?
                   (ws? "<label_angle>" nl? ws? str nl? ws? "</label_angle>" nl?)?
                   (ws? "<pts>" ws? nl?)?
                   (ws? "<pt x=" skip nl?)?
                   (ws? "</pt>" ws? nl?)?
                   (ws? "</pts>" ws? nl?)?
                   (ws? "</display>" ws? nl?)?
               ("</flow>" nl)?
        flow_disp_name = "isee:display_name=" qtm str qtm
        flow_name = ~"[A-Za-z0-9$_ áéíóúñ]*"
        eqn = ~"[ A-z0-9.,;&+*/áéíóúñ\-\^\(\)\n]*"
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
                   (ws? "<eqn>" nl eqn? "</eqn>" nl)?
                   (ws? "<units>" nl ws units nl ws "</units>" nl)?
                   (ws? "<display" skip nl)?
                   (ws? "<format" skip nl)?
                   (ws? "</format>" nl)?
                   (ws? "<scale " skip nl)?
                   (ws? "</scale>" nl)?
                   (ws? "<doc>" skip nl skip? nl? ws? "</doc>" nl)?
                   (ws? "<label_side>" nl? ws? str nl? ws? "</label_side>" nl?)?
                   (ws? "<label_angle>" nl? ws? str nl? ws? "</label_angle>" nl?)?
                   (ws? "<thous_separator" skip nl)?
                   (ws? "true" nl)?
                   (ws? "</thous_separator>" nl)?
                   (ws? "<gf>" nl)?
                   (ws? "<gf discrete=" qtm ("false" / "true") qtm ">" nl)? 
                       (ws "<xscale max=" qtm max qtm ws
                                   "min=" qtm min qtm ">" nl)?
                       (ws? "</xscale>" nl)?
                       (ws? "<yscale max=" str ws "min=" str ">" nl
                        ws? "</yscale>" nl)?
                       (ws? "<ypts>" nl ws ypts nl ws "</ypts>" nl)?
                       (ws? "<xpts>" nl ws skip nl ws "</xpts>" nl)?                       
                   (ws? "</gf>" nl)?
                   (ws? "</display>" skip? nl)?
               (ws? "</aux>" skip? nl)?
        aux_disp_name = "isee:display_name=" str
        min = ~"[0-9.]*"
        max = ~"[0-9.]*"
        aux_name = ~"[A-Za-z0-9$_ áéíóúñ]*"
        eqn = ~"[ A-z0-9.,;&+*/áéíóúñ\-\^\(\)\n]*"
        ypts = ~"[0-9.,-]*"
        units = ~"[A-z/ ]*"
        str = ~"[A-z0-9\"_]*"
        qtm = '"'
        skip = ~".*"
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

    # Read model file
    with open(model_file, encoding="utf-8") as fmodel:
        xmile_soup = bs(fmodel, "lxml")

    # Extract relevant information to build the model reflecting XMILE version
    model_spec = {"name": xmile_soup.header.find("name").text,
                  "step": float(xmile_soup.find("dt").text),
                  "start": int(xmile_soup.find("start").text),
                  "stop": int(xmile_soup.find("stop").text)}
    if model_spec["name"] == "":
        model_spec["name"] = model_file.split(".")[0]

    # Extract equations block located in model/variables section of xmile file
    # First get flow equations
    flows, parameters_set, flows_set = {}, set(), set()
    for flw in xmile_soup.model.find_all("flow"):
        if "eqn" in flw.prettify():
            flw_parse = FlowParser(flow_grammar, flw.prettify()).entry
            flw_parse["eqn"] = flw_parse["eqn"].replace("\n", " ").strip(" ")
            if "units" not in flw_parse:
                flw_parse["units"] = ""
            name = flw_parse["flow_name"].replace(" ", "_")
            name = name.replace("á", "a").replace("é", "e").replace("í", "i").\
                        replace("ó", "o").replace("ú", "u").replace("ñ", "n")
            flw_parse["flow_name"] = name
            flows_set = flows_set.union([name])
            flows[name] = flw_parse
            if not isfloat(flw_parse["eqn"]):
                parameters_set = parameters_set.union(re.split(r"[*/+-]", flw_parse["eqn"]))

    # Extract stock collection attributes
    stocks_list, stocks_set = [], set()
    for stk in xmile_soup.model.find_all("stock"):
        if "eqn" in stk.prettify():
            stock_parse = StockParser(stock_grammar, stk.prettify()).entry
            name = stock_parse["stock_name"].replace(" ", "_")
            name = name.replace("á", "a").replace("é", "e").replace("í", "i")
            name = name.replace("ó", "o").replace("ú", "u").replace("ñ", "n")
            stock_parse["stock_name"] = name
            stocks_set = stocks_set.union([name])
            parameters_set = parameters_set - stocks_set - flows_set
            if "units" not in stock_parse:
                stock_parse["units"] = ""
            if stk.inflow:
                all_flows = [item.text.replace("á", "a").replace("é", "e").
                                       replace("í","i").replace("ó", "o").replace("ú", "u").
                                       replace("ñ", "n") for item in stk.find_all("inflow")]
                stock_parse["inflow"] = [flows[f] for f in all_flows]

            if stk.outflow:
                all_flows = [item.text.replace("á", "a").replace("é", "e").
                                       replace("í", "i").replace("ó", "o").replace("ú", "u").
                                       replace("ñ", "n") for item in stk.find_all("outflow")]
                stock_parse["outflow"] = [flows[f] for f in all_flows]
            stocks_list.append(XmileStock(stock_parse))

    # Get aux variables
    auxs_dict, auxs_set = {}, parameters_set
    for ax in xmile_soup.model.find_all("aux"):
        if "eqn" in ax.prettify():
            aux_parse = AuxParser(aux_grammar, ax.prettify()).entry
            name = aux_parse["aux_name"].replace(" ", "_")
            aux_parse["aux_name"] = name
            aux_parse["eqn"] = aux_parse["eqn"].replace("\n", " ").strip(" ")
            auxs_set = auxs_set.union([name])
            if "units" not in aux_parse:
                aux_parse["units"] = ""
            if name in parameters_set:
                aux_parse["is_parameter"] = True
            else:
                aux_parse["is_parameter"] = False
            auxs_dict.update({name: XmileAux(aux_parse)})

    model_translation = XmileModel(model_spec, stocks_list, auxs_dict)
    model_translation.stocks_set = stocks_set
    model_translation.flows_set = flows_set
    model_translation.parameters_set = parameters_set
    model_translation.auxs_set = auxs_set

    return model_translation


# %% Read xmile model (like stella 10 onwards)
if __name__ == '__main__':
    main()
