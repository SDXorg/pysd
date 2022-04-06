
class ImportsManager():
    """
    Class to save the imported modules information for intelligent import
    """
    _external_libs = {"numpy": "np", "xarray": "xr"}
    _external_submodules = ["scipy"]
    _internal_libs = [
        "functions", "statefuls", "external", "data", "lookups", "utils"
    ]

    def __init__(self):
        self._numpy, self._xarray = False, False
        self._functions, self._statefuls, self._external, self._data,\
            self._lookups, self._utils, self._scipy =\
            set(), set(), set(), set(), set(), set(), set()

    def add(self, module, function=None):
        """
        Add a function from module.

        Parameters
        ----------
        module: str
          module name.

        function: str or None
          function name. If None module will be set to true.

        """
        if function:
            getattr(self, f"_{module}").add(function)
        else:
            setattr(self, f"_{module}", True)

    def get_header(self, outfile):
        """
        Returns the importing information to print in the model file

        Parameters
        ----------
        outfile: str
            Name of the outfile to print in the header.

        Returns
        -------
        text: str
            Header of the translated model file.

        """
        text =\
            f'"""\nPython model \'{outfile}\'\nTranslated using PySD\n"""\n\n'

        text += "from pathlib import Path\n"

        for module, shortname in self._external_libs.items():
            if getattr(self, f"_{module}"):
                text += f"import {module} as {shortname}\n"

        for module in self._external_submodules:
            if getattr(self, f"_{module}"):
                text += "from %(module)s import %(submodules)s\n" % {
                    "module": module,
                    "submodules": ", ".join(getattr(self, f"_{module}"))}

        text += "\n"

        for module in self._internal_libs:
            if getattr(self, f"_{module}"):
                text += "from pysd.py_backend.%(module)s import %(methods)s\n"\
                        % {
                            "module": module,
                            "methods": ", ".join(getattr(self, f"_{module}"))}

        text += "from pysd import component\n"

        return text
