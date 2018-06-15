"""
Translates vensim .mdl file to pieces needed by the builder module to write a python version of the
model. Everything that requires knowledge of vensim syntax should be in this file.
"""
from __future__ import absolute_import
import re
import parsimonious
from ...py_backend import builder
from ...py_backend import utils
import textwrap
import numpy as np
import os


def get_file_sections(file_str):
    """
    This is where we separate out the macros from the rest of the model file.
    Working based upon documentation at: https://www.vensim.com/documentation/index.html?macros.htm

    Macros will probably wind up in their own python modules eventually.

    Parameters
    ----------
    file_str

    Returns
    -------
    entries: list of dictionaries
        Each dictionary represents a different section of the model file, either a macro,
        or the main body of the model file. The dictionaries contain various elements:
        - returns: list of strings
            represents what is returned from a macro (for macros) or empty for main model
        - params: list of strings
            represents what is passed into a macro (for macros) or empty for main model
        - name: string
            the name of the macro, or 'main' for main body of model
        - string: string
            string representing the model section
    Examples
    --------
    >>> get_file_sections(r'a~b~c| d~e~f| g~h~i|')
    [{'returns': [], 'params': [], 'name': 'main', 'string': 'a~b~c| d~e~f| g~h~i|'}]

    """
    file_structure_grammar = r"""
    file = encoding? (macro / main)+
    macro = ":MACRO:" _ name _ "(" _ (name _ ","? _)+ _ ":"? _ (name _ ","? _)* _ ")" ~r".+?(?=:END OF MACRO:)" ":END OF MACRO:"
    main = !":MACRO:" ~r".+(?!:MACRO:)"

    name = basic_id / escape_group
    
    # This takes care of models with Unicode variable names
    basic_id = id_start (id_continue / ~r"[\'\$\s]")*
    id_start = ~r"[A-Z]" / ~r"[a-z]" / "\u00AA" / "\u00B5" / "\u00BA" / ~r"[\u00C0-\u00D6]" / ~r"[\u00D8-\u00F6]" / ~r"[\u00F8-\u01BA]" / "\u01BB" / ~r"[\u01BC-\u01BF]" / ~r"[\u01C0-\u01C3]" / ~r"[\u01C4-\u0241]" / ~r"[\u0250-\u02AF]" / ~r"[\u02B0-\u02C1]" / ~r"[\u02C6-\u02D1]" / ~r"[\u02E0-\u02E4]" / "\u02EE" / "\u037A" / "\u0386" / ~r"[\u0388-\u038A]" / "\u038C" / ~r"[\u038E-\u03A1]" / ~r"[\u03A3-\u03CE]" / ~r"[\u03D0-\u03F5]" / ~r"[\u03F7-\u0481]" / ~r"[\u048A-\u04CE]" / ~r"[\u04D0-\u04F9]" / ~r"[\u0500-\u050F]" / ~r"[\u0531-\u0556]" / "\u0559" / ~r"[\u0561-\u0587]" / ~r"[\u05D0-\u05EA]" / ~r"[\u05F0-\u05F2]" / ~r"[\u0621-\u063A]" / "\u0640" / ~r"[\u0641-\u064A]" / ~r"[\u066E-\u066F]" / ~r"[\u0671-\u06D3]" / "\u06D5" / ~r"[\u06E5-\u06E6]" / ~r"[\u06EE-\u06EF]" / ~r"[\u06FA-\u06FC]" / "\u06FF" / "\u0710" / ~r"[\u0712-\u072F]" / ~r"[\u074D-\u076D]" / ~r"[\u0780-\u07A5]" / "\u07B1" / ~r"[\u0904-\u0939]" / "\u093D" / "\u0950" / ~r"[\u0958-\u0961]" / "\u097D" / ~r"[\u0985-\u098C]" / ~r"[\u098F-\u0990]" / ~r"[\u0993-\u09A8]" / ~r"[\u09AA-\u09B0]" / "\u09B2" / ~r"[\u09B6-\u09B9]" / "\u09BD" / "\u09CE" / ~r"[\u09DC-\u09DD]" / ~r"[\u09DF-\u09E1]" / ~r"[\u09F0-\u09F1]" / ~r"[\u0A05-\u0A0A]" / ~r"[\u0A0F-\u0A10]" / ~r"[\u0A13-\u0A28]" / ~r"[\u0A2A-\u0A30]" / ~r"[\u0A32-\u0A33]" / ~r"[\u0A35-\u0A36]" / ~r"[\u0A38-\u0A39]" / ~r"[\u0A59-\u0A5C]" / "\u0A5E" / ~r"[\u0A72-\u0A74]" / ~r"[\u0A85-\u0A8D]" / ~r"[\u0A8F-\u0A91]" / ~r"[\u0A93-\u0AA8]" / ~r"[\u0AAA-\u0AB0]" / ~r"[\u0AB2-\u0AB3]" / ~r"[\u0AB5-\u0AB9]" / "\u0ABD" / "\u0AD0" / ~r"[\u0AE0-\u0AE1]" / ~r"[\u0B05-\u0B0C]" / ~r"[\u0B0F-\u0B10]" / ~r"[\u0B13-\u0B28]" / ~r"[\u0B2A-\u0B30]" / ~r"[\u0B32-\u0B33]" / ~r"[\u0B35-\u0B39]" / "\u0B3D" / ~r"[\u0B5C-\u0B5D]" / ~r"[\u0B5F-\u0B61]" / "\u0B71" / "\u0B83" / ~r"[\u0B85-\u0B8A]" / ~r"[\u0B8E-\u0B90]" / ~r"[\u0B92-\u0B95]" / ~r"[\u0B99-\u0B9A]" / "\u0B9C" / ~r"[\u0B9E-\u0B9F]" / ~r"[\u0BA3-\u0BA4]" / ~r"[\u0BA8-\u0BAA]" / ~r"[\u0BAE-\u0BB9]" / ~r"[\u0C05-\u0C0C]" / ~r"[\u0C0E-\u0C10]" / ~r"[\u0C12-\u0C28]" / ~r"[\u0C2A-\u0C33]" / ~r"[\u0C35-\u0C39]" / ~r"[\u0C60-\u0C61]" / ~r"[\u0C85-\u0C8C]" / ~r"[\u0C8E-\u0C90]" / ~r"[\u0C92-\u0CA8]" / ~r"[\u0CAA-\u0CB3]" / ~r"[\u0CB5-\u0CB9]" / "\u0CBD" / "\u0CDE" / ~r"[\u0CE0-\u0CE1]" / ~r"[\u0D05-\u0D0C]" / ~r"[\u0D0E-\u0D10]" / ~r"[\u0D12-\u0D28]" / ~r"[\u0D2A-\u0D39]" / ~r"[\u0D60-\u0D61]" / ~r"[\u0D85-\u0D96]" / ~r"[\u0D9A-\u0DB1]" / ~r"[\u0DB3-\u0DBB]" / "\u0DBD" / ~r"[\u0DC0-\u0DC6]" / ~r"[\u0E01-\u0E30]" / ~r"[\u0E32-\u0E33]" / ~r"[\u0E40-\u0E45]" / "\u0E46" / ~r"[\u0E81-\u0E82]" / "\u0E84" / ~r"[\u0E87-\u0E88]" / "\u0E8A" / "\u0E8D" / ~r"[\u0E94-\u0E97]" / ~r"[\u0E99-\u0E9F]" / ~r"[\u0EA1-\u0EA3]" / "\u0EA5" / "\u0EA7" / ~r"[\u0EAA-\u0EAB]" / ~r"[\u0EAD-\u0EB0]" / ~r"[\u0EB2-\u0EB3]" / "\u0EBD" / ~r"[\u0EC0-\u0EC4]" / "\u0EC6" / ~r"[\u0EDC-\u0EDD]" / "\u0F00" / ~r"[\u0F40-\u0F47]" / ~r"[\u0F49-\u0F6A]" / ~r"[\u0F88-\u0F8B]" / ~r"[\u1000-\u1021]" / ~r"[\u1023-\u1027]" / ~r"[\u1029-\u102A]" / ~r"[\u1050-\u1055]" / ~r"[\u10A0-\u10C5]" / ~r"[\u10D0-\u10FA]" / "\u10FC" / ~r"[\u1100-\u1159]" / ~r"[\u115F-\u11A2]" / ~r"[\u11A8-\u11F9]" / ~r"[\u1200-\u1248]" / ~r"[\u124A-\u124D]" / ~r"[\u1250-\u1256]" / "\u1258" / ~r"[\u125A-\u125D]" / ~r"[\u1260-\u1288]" / ~r"[\u128A-\u128D]" / ~r"[\u1290-\u12B0]" / ~r"[\u12B2-\u12B5]" / ~r"[\u12B8-\u12BE]" / "\u12C0" / ~r"[\u12C2-\u12C5]" / ~r"[\u12C8-\u12D6]" / ~r"[\u12D8-\u1310]" / ~r"[\u1312-\u1315]" / ~r"[\u1318-\u135A]" / ~r"[\u1380-\u138F]" / ~r"[\u13A0-\u13F4]" / ~r"[\u1401-\u166C]" / ~r"[\u166F-\u1676]" / ~r"[\u1681-\u169A]" / ~r"[\u16A0-\u16EA]" / ~r"[\u16EE-\u16F0]" / ~r"[\u1700-\u170C]" / ~r"[\u170E-\u1711]" / ~r"[\u1720-\u1731]" / ~r"[\u1740-\u1751]" / ~r"[\u1760-\u176C]" / ~r"[\u176E-\u1770]" / ~r"[\u1780-\u17B3]" / "\u17D7" / "\u17DC" / ~r"[\u1820-\u1842]" / "\u1843" / ~r"[\u1844-\u1877]" / ~r"[\u1880-\u18A8]" / ~r"[\u1900-\u191C]" / ~r"[\u1950-\u196D]" / ~r"[\u1970-\u1974]" / ~r"[\u1980-\u19A9]" / ~r"[\u19C1-\u19C7]" / ~r"[\u1A00-\u1A16]" / ~r"[\u1D00-\u1D2B]" / ~r"[\u1D2C-\u1D61]" / ~r"[\u1D62-\u1D77]" / "\u1D78" / ~r"[\u1D79-\u1D9A]" / ~r"[\u1D9B-\u1DBF]" / ~r"[\u1E00-\u1E9B]" / ~r"[\u1EA0-\u1EF9]" / ~r"[\u1F00-\u1F15]" / ~r"[\u1F18-\u1F1D]" / ~r"[\u1F20-\u1F45]" / ~r"[\u1F48-\u1F4D]" / ~r"[\u1F50-\u1F57]" / "\u1F59" / "\u1F5B" / "\u1F5D" / ~r"[\u1F5F-\u1F7D]" / ~r"[\u1F80-\u1FB4]" / ~r"[\u1FB6-\u1FBC]" / "\u1FBE" / ~r"[\u1FC2-\u1FC4]" / ~r"[\u1FC6-\u1FCC]" / ~r"[\u1FD0-\u1FD3]" / ~r"[\u1FD6-\u1FDB]" / ~r"[\u1FE0-\u1FEC]" / ~r"[\u1FF2-\u1FF4]" / ~r"[\u1FF6-\u1FFC]" / "\u2071" / "\u207F" / ~r"[\u2090-\u2094]" / "\u2102" / "\u2107" / ~r"[\u210A-\u2113]" / "\u2115" / "\u2118" / ~r"[\u2119-\u211D]" / "\u2124" / "\u2126" / "\u2128" / ~r"[\u212A-\u212D]" / "\u212E" / ~r"[\u212F-\u2131]" / ~r"[\u2133-\u2134]" / ~r"[\u2135-\u2138]" / "\u2139" / ~r"[\u213C-\u213F]" / ~r"[\u2145-\u2149]" / ~r"[\u2160-\u2183]" / ~r"[\u2C00-\u2C2E]" / ~r"[\u2C30-\u2C5E]" / ~r"[\u2C80-\u2CE4]" / ~r"[\u2D00-\u2D25]" / ~r"[\u2D30-\u2D65]" / "\u2D6F" / ~r"[\u2D80-\u2D96]" / ~r"[\u2DA0-\u2DA6]" / ~r"[\u2DA8-\u2DAE]" / ~r"[\u2DB0-\u2DB6]" / ~r"[\u2DB8-\u2DBE]" / ~r"[\u2DC0-\u2DC6]" / ~r"[\u2DC8-\u2DCE]" / ~r"[\u2DD0-\u2DD6]" / ~r"[\u2DD8-\u2DDE]" / "\u3005" / "\u3006" / "\u3007" / ~r"[\u3021-\u3029]" / ~r"[\u3031-\u3035]" / ~r"[\u3038-\u303A]" / "\u303B" / "\u303C" / ~r"[\u3041-\u3096]" / ~r"[\u309B-\u309C]" / ~r"[\u309D-\u309E]" / "\u309F" / ~r"[\u30A1-\u30FA]" / ~r"[\u30FC-\u30FE]" / "\u30FF" / ~r"[\u3105-\u312C]" / ~r"[\u3131-\u318E]" / ~r"[\u31A0-\u31B7]" / ~r"[\u31F0-\u31FF]" / ~r"[\u3400-\u4DB5]" / ~r"[\u4E00-\u9FBB]" / ~r"[\uA000-\uA014]" / "\uA015" / ~r"[\uA016-\uA48C]" / ~r"[\uA800-\uA801]" / ~r"[\uA803-\uA805]" / ~r"[\uA807-\uA80A]" / ~r"[\uA80C-\uA822]" / ~r"[\uAC00-\uD7A3]" / ~r"[\uF900-\uFA2D]" / ~r"[\uFA30-\uFA6A]" / ~r"[\uFA70-\uFAD9]" / ~r"[\uFB00-\uFB06]" / ~r"[\uFB13-\uFB17]" / "\uFB1D" / ~r"[\uFB1F-\uFB28]" / ~r"[\uFB2A-\uFB36]" / ~r"[\uFB38-\uFB3C]" / "\uFB3E" / ~r"[\uFB40-\uFB41]" / ~r"[\uFB43-\uFB44]" / ~r"[\uFB46-\uFBB1]" / ~r"[\uFBD3-\uFD3D]" / ~r"[\uFD50-\uFD8F]" / ~r"[\uFD92-\uFDC7]" / ~r"[\uFDF0-\uFDFB]" / ~r"[\uFE70-\uFE74]" / ~r"[\uFE76-\uFEFC]" / ~r"[\uFF21-\uFF3A]" / ~r"[\uFF41-\uFF5A]" / ~r"[\uFF66-\uFF6F]" / "\uFF70" / ~r"[\uFF71-\uFF9D]" / ~r"[\uFF9E-\uFF9F]" / ~r"[\uFFA0-\uFFBE]" / ~r"[\uFFC2-\uFFC7]" / ~r"[\uFFCA-\uFFCF]" / ~r"[\uFFD2-\uFFD7]" / ~r"[\uFFDA-\uFFDC]"
    id_continue = id_start / ~r"[0-9]" / ~r"[\u0300-\u036F]" / ~r"[\u0483-\u0486]" / ~r"[\u0591-\u05B9]" / ~r"[\u05BB-\u05BD]" / "\u05BF" / ~r"[\u05C1-\u05C2]" / ~r"[\u05C4-\u05C5]" / "\u05C7" / ~r"[\u0610-\u0615]" / ~r"[\u064B-\u065E]" / ~r"[\u0660-\u0669]" / "\u0670" / ~r"[\u06D6-\u06DC]" / ~r"[\u06DF-\u06E4]" / ~r"[\u06E7-\u06E8]" / ~r"[\u06EA-\u06ED]" / ~r"[\u06F0-\u06F9]" / "\u0711" / ~r"[\u0730-\u074A]" / ~r"[\u07A6-\u07B0]" / ~r"[\u0901-\u0902]" / "\u0903" / "\u093C" / ~r"[\u093E-\u0940]" / ~r"[\u0941-\u0948]" / ~r"[\u0949-\u094C]" / "\u094D" / ~r"[\u0951-\u0954]" / ~r"[\u0962-\u0963]" / ~r"[\u0966-\u096F]" / "\u0981" / ~r"[\u0982-\u0983]" / "\u09BC" / ~r"[\u09BE-\u09C0]" / ~r"[\u09C1-\u09C4]" / ~r"[\u09C7-\u09C8]" / ~r"[\u09CB-\u09CC]" / "\u09CD" / "\u09D7" / ~r"[\u09E2-\u09E3]" / ~r"[\u09E6-\u09EF]" / ~r"[\u0A01-\u0A02]" / "\u0A03" / "\u0A3C" / ~r"[\u0A3E-\u0A40]" / ~r"[\u0A41-\u0A42]" / ~r"[\u0A47-\u0A48]" / ~r"[\u0A4B-\u0A4D]" / ~r"[\u0A66-\u0A6F]" / ~r"[\u0A70-\u0A71]" / ~r"[\u0A81-\u0A82]" / "\u0A83" / "\u0ABC" / ~r"[\u0ABE-\u0AC0]" / ~r"[\u0AC1-\u0AC5]" / ~r"[\u0AC7-\u0AC8]" / "\u0AC9" / ~r"[\u0ACB-\u0ACC]" / "\u0ACD" / ~r"[\u0AE2-\u0AE3]" / ~r"[\u0AE6-\u0AEF]" / "\u0B01" / ~r"[\u0B02-\u0B03]" / "\u0B3C" / "\u0B3E" / "\u0B3F" / "\u0B40" / ~r"[\u0B41-\u0B43]" / ~r"[\u0B47-\u0B48]" / ~r"[\u0B4B-\u0B4C]" / "\u0B4D" / "\u0B56" / "\u0B57" / ~r"[\u0B66-\u0B6F]" / "\u0B82" / ~r"[\u0BBE-\u0BBF]" / "\u0BC0" / ~r"[\u0BC1-\u0BC2]" / ~r"[\u0BC6-\u0BC8]" / ~r"[\u0BCA-\u0BCC]" / "\u0BCD" / "\u0BD7" / ~r"[\u0BE6-\u0BEF]" / ~r"[\u0C01-\u0C03]" / ~r"[\u0C3E-\u0C40]" / ~r"[\u0C41-\u0C44]" / ~r"[\u0C46-\u0C48]" / ~r"[\u0C4A-\u0C4D]" / ~r"[\u0C55-\u0C56]" / ~r"[\u0C66-\u0C6F]" / ~r"[\u0C82-\u0C83]" / "\u0CBC" / "\u0CBE" / "\u0CBF" / ~r"[\u0CC0-\u0CC4]" / "\u0CC6" / ~r"[\u0CC7-\u0CC8]" / ~r"[\u0CCA-\u0CCB]" / ~r"[\u0CCC-\u0CCD]" / ~r"[\u0CD5-\u0CD6]" / ~r"[\u0CE6-\u0CEF]" / ~r"[\u0D02-\u0D03]" / ~r"[\u0D3E-\u0D40]" / ~r"[\u0D41-\u0D43]" / ~r"[\u0D46-\u0D48]" / ~r"[\u0D4A-\u0D4C]" / "\u0D4D" / "\u0D57" / ~r"[\u0D66-\u0D6F]" / ~r"[\u0D82-\u0D83]" / "\u0DCA" / ~r"[\u0DCF-\u0DD1]" / ~r"[\u0DD2-\u0DD4]" / "\u0DD6" / ~r"[\u0DD8-\u0DDF]" / ~r"[\u0DF2-\u0DF3]" / "\u0E31" / ~r"[\u0E34-\u0E3A]" / ~r"[\u0E47-\u0E4E]" / ~r"[\u0E50-\u0E59]" / "\u0EB1" / ~r"[\u0EB4-\u0EB9]" / ~r"[\u0EBB-\u0EBC]" / ~r"[\u0EC8-\u0ECD]" / ~r"[\u0ED0-\u0ED9]" / ~r"[\u0F18-\u0F19]" / ~r"[\u0F20-\u0F29]" / "\u0F35" / "\u0F37" / "\u0F39" / ~r"[\u0F3E-\u0F3F]" / ~r"[\u0F71-\u0F7E]" / "\u0F7F" / ~r"[\u0F80-\u0F84]" / ~r"[\u0F86-\u0F87]" / ~r"[\u0F90-\u0F97]" / ~r"[\u0F99-\u0FBC]" / "\u0FC6" / "\u102C" / ~r"[\u102D-\u1030]" / "\u1031" / "\u1032" / ~r"[\u1036-\u1037]" / "\u1038" / "\u1039" / ~r"[\u1040-\u1049]" / ~r"[\u1056-\u1057]" / ~r"[\u1058-\u1059]" / "\u135F" / ~r"[\u1369-\u1371]" / ~r"[\u1712-\u1714]" / ~r"[\u1732-\u1734]" / ~r"[\u1752-\u1753]" / ~r"[\u1772-\u1773]" / "\u17B6" / ~r"[\u17B7-\u17BD]" / ~r"[\u17BE-\u17C5]" / "\u17C6" / ~r"[\u17C7-\u17C8]" / ~r"[\u17C9-\u17D3]" / "\u17DD" / ~r"[\u17E0-\u17E9]" / ~r"[\u180B-\u180D]" / ~r"[\u1810-\u1819]" / "\u18A9" / ~r"[\u1920-\u1922]" / ~r"[\u1923-\u1926]" / ~r"[\u1927-\u1928]" / ~r"[\u1929-\u192B]" / ~r"[\u1930-\u1931]" / "\u1932" / ~r"[\u1933-\u1938]" / ~r"[\u1939-\u193B]" / ~r"[\u1946-\u194F]" / ~r"[\u19B0-\u19C0]" / ~r"[\u19C8-\u19C9]" / ~r"[\u19D0-\u19D9]" / ~r"[\u1A17-\u1A18]" / ~r"[\u1A19-\u1A1B]" / ~r"[\u1DC0-\u1DC3]" / ~r"[\u203F-\u2040]" / "\u2054" / ~r"[\u20D0-\u20DC]" / "\u20E1" / ~r"[\u20E5-\u20EB]" / ~r"[\u302A-\u302F]" / ~r"[\u3099-\u309A]" / "\uA802" / "\uA806" / "\uA80B" / ~r"[\uA823-\uA824]" / ~r"[\uA825-\uA826]" / "\uA827" / "\uFB1E" / ~r"[\uFE00-\uFE0F]" / ~r"[\uFE20-\uFE23]" / ~r"[\uFE33-\uFE34]" / ~r"[\uFE4D-\uFE4F]" / ~r"[\uFF10-\uFF19]" / "\uFF3F"


    # between quotes, either escaped quote or character that is not a quote
    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    encoding = ~r"\{[^\}]*\}"

    _ = ~r"[\s\\]*"  # whitespace character
    """  # the leading 'r' for 'raw' in this string is important for handling backslashes properly

    parser = parsimonious.Grammar(file_structure_grammar)
    tree = parser.parse(file_str)

    class FileParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.entries = []
            self.visit(ast)

        def visit_main(self, n, vc):
            self.entries.append({'name': '_main_',
                                 'params': [],
                                 'returns': [],
                                 'string': n.text.strip()})

        def visit_macro(self, n, vc):
            name = vc[2]
            params = vc[6]
            returns = vc[10]
            text = vc[13]
            self.entries.append({'name': name,
                                 'params': [x.strip() for x in params.split(',')] if params else [],
                                 'returns': [x.strip() for x in
                                             returns.split(',')] if returns else [],
                                 'string': text.strip()})

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text or ''

    return FileParser(tree).entries


def get_model_elements(model_str):
    """
    Takes in a string representing model text and splits it into elements

    I think we're making the assumption that all newline characters are removed...

    Parameters
    ----------
    model_str : string


    Returns
    -------
    entries : array of dictionaries
        Each dictionary contains the components of a different model element, separated into the
        equation, units, and docstring.

    Examples
    --------

    # Basic Parsing:
    >>> get_model_elements(r'a~b~c| d~e~f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]

    # Special characters are escaped within double-quotes:
    >>> get_model_elements(r'a~b~c| d~e"~"~f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e"~"', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]
    >>> get_model_elements(r'a~b~c| d~e~"|"f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': '"|"f', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]

    # Double-quotes within escape groups are themselves escaped with backslashes:
    >>> get_model_elements(r'a~b~c| d~e"\\\"~"~f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e"\\\\"~"', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]
    >>> get_model_elements(r'a~b~c| d~e~"\\\"|"f| g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': '"\\\\"|"f', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]
    >>> get_model_elements(r'a~b~c| d~e"x\\nx"~f| g~h~|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e"x\\\\nx"', 'eqn': 'd'}, {'doc': '', 'unit': 'h', 'eqn': 'g'}]

    # Todo: Handle model-level or section-level documentation
    >>> get_model_elements(r'*** .model doc ***~ Docstring!| d~e~f| g~h~i|')
    [{'doc': 'Docstring!', 'unit': '', 'eqn': ''}, {'doc': 'f', 'unit': 'e', 'eqn': 'd'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]

    # Handle control sections, returning appropriate docstring pieces
    >>> get_model_elements(r'a~b~c| ****.Control***~ Simulation Control Parameters | g~h~i|')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'i', 'unit': 'h', 'eqn': 'g'}]

    # Handle the model display elements (ignore them)
    >>> get_model_elements(r'a~b~c| d~e~f| \\\---///junk|junk~junk')
    [{'doc': 'c', 'unit': 'b', 'eqn': 'a'}, {'doc': 'f', 'unit': 'e', 'eqn': 'd'}]


    Notes
    -----
    - Tildes and pipes are not allowed in element docstrings, but we should still handle them there

    """

    model_structure_grammar = r"""
    model = (entry / section)+ sketch?
    entry = element "~" element "~" element ("~" element)? "|"
    section = element "~" element "|"
    sketch = ~r".*"  #anything

    # Either an escape group, or a character that is not tilde or pipe
    element = (escape_group / ~r"[^~|]")*

    # between quotes, either escaped quote or character that is not a quote
    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    """
    parser = parsimonious.Grammar(model_structure_grammar)
    tree = parser.parse(model_str)

    class ModelParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.entries = []
            self.visit(ast)

        def visit_entry(self, n, vc):
            units, lims = parse_units(vc[2].strip())
            self.entries.append({'eqn': vc[0].strip(),
                                 'unit': units,
                                 'lims': str(lims),
                                 'doc': vc[4].strip(),
                                 'kind': 'entry'})

        def visit_section(self, n, vc):
            if vc[2].strip() != "Simulation Control Parameters":
                self.entries.append({'eqn': '',
                                     'unit': '',
                                     'lims': '',
                                     'doc': vc[2].strip(),
                                     'kind': 'section'})

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text or ''

    return ModelParser(tree).entries


def get_equation_components(equation_str):
    """
    Breaks down a string representing only the equation part of a model element.
    Recognizes the various types of model elements that may exist, and identifies them.

    Parameters
    ----------
    equation_str : basestring
        the first section in each model element - the full equation.

    Returns
    -------
    Returns a dictionary containing the following:

    real_name: basestring
        The name of the element as given in the original vensim file

    subs: list of strings
        list of subscripts or subscript elements

    expr: basestring

    kind: basestring
        What type of equation have we found?
        - *component* - normal model expression or constant
        - *lookup* - a lookup table
        - *subdef* - a subscript definition

    Examples
    --------
    >>> get_equation_components(r'constant = 25')
    {'expr': '25', 'kind': 'component', 'subs': [], 'real_name': 'constant'}

    Notes
    -----
    in this function we dont create python identifiers, we use real names.
    This is so that when everything comes back together, we can manage
    any potential namespace conflicts properly
    """

    component_structure_grammar = r"""
    entry = component / subscript_definition / lookup_definition
    component = name _ subscriptlist? _ "=" _ expression
    subscript_definition = name _ ":" _ subscript _ ("," _ subscript)*
    lookup_definition = name _ &"(" _ expression  # uses lookahead assertion to capture whole group

    name = basic_id / escape_group
    subscriptlist = '[' _ subscript _ ("," _ subscript)* _ ']'
    expression = ~r".*"  # expression could be anything, at this point.

    subscript = basic_id / escape_group

    # This takes care of models with Unicode variable names
    basic_id = id_start (id_continue / ~r"[\'\$\s]")*
    id_start = ~r"[A-Z]" / ~r"[a-z]" / "\u00AA" / "\u00B5" / "\u00BA" / ~r"[\u00C0-\u00D6]" / ~r"[\u00D8-\u00F6]" / ~r"[\u00F8-\u01BA]" / "\u01BB" / ~r"[\u01BC-\u01BF]" / ~r"[\u01C0-\u01C3]" / ~r"[\u01C4-\u0241]" / ~r"[\u0250-\u02AF]" / ~r"[\u02B0-\u02C1]" / ~r"[\u02C6-\u02D1]" / ~r"[\u02E0-\u02E4]" / "\u02EE" / "\u037A" / "\u0386" / ~r"[\u0388-\u038A]" / "\u038C" / ~r"[\u038E-\u03A1]" / ~r"[\u03A3-\u03CE]" / ~r"[\u03D0-\u03F5]" / ~r"[\u03F7-\u0481]" / ~r"[\u048A-\u04CE]" / ~r"[\u04D0-\u04F9]" / ~r"[\u0500-\u050F]" / ~r"[\u0531-\u0556]" / "\u0559" / ~r"[\u0561-\u0587]" / ~r"[\u05D0-\u05EA]" / ~r"[\u05F0-\u05F2]" / ~r"[\u0621-\u063A]" / "\u0640" / ~r"[\u0641-\u064A]" / ~r"[\u066E-\u066F]" / ~r"[\u0671-\u06D3]" / "\u06D5" / ~r"[\u06E5-\u06E6]" / ~r"[\u06EE-\u06EF]" / ~r"[\u06FA-\u06FC]" / "\u06FF" / "\u0710" / ~r"[\u0712-\u072F]" / ~r"[\u074D-\u076D]" / ~r"[\u0780-\u07A5]" / "\u07B1" / ~r"[\u0904-\u0939]" / "\u093D" / "\u0950" / ~r"[\u0958-\u0961]" / "\u097D" / ~r"[\u0985-\u098C]" / ~r"[\u098F-\u0990]" / ~r"[\u0993-\u09A8]" / ~r"[\u09AA-\u09B0]" / "\u09B2" / ~r"[\u09B6-\u09B9]" / "\u09BD" / "\u09CE" / ~r"[\u09DC-\u09DD]" / ~r"[\u09DF-\u09E1]" / ~r"[\u09F0-\u09F1]" / ~r"[\u0A05-\u0A0A]" / ~r"[\u0A0F-\u0A10]" / ~r"[\u0A13-\u0A28]" / ~r"[\u0A2A-\u0A30]" / ~r"[\u0A32-\u0A33]" / ~r"[\u0A35-\u0A36]" / ~r"[\u0A38-\u0A39]" / ~r"[\u0A59-\u0A5C]" / "\u0A5E" / ~r"[\u0A72-\u0A74]" / ~r"[\u0A85-\u0A8D]" / ~r"[\u0A8F-\u0A91]" / ~r"[\u0A93-\u0AA8]" / ~r"[\u0AAA-\u0AB0]" / ~r"[\u0AB2-\u0AB3]" / ~r"[\u0AB5-\u0AB9]" / "\u0ABD" / "\u0AD0" / ~r"[\u0AE0-\u0AE1]" / ~r"[\u0B05-\u0B0C]" / ~r"[\u0B0F-\u0B10]" / ~r"[\u0B13-\u0B28]" / ~r"[\u0B2A-\u0B30]" / ~r"[\u0B32-\u0B33]" / ~r"[\u0B35-\u0B39]" / "\u0B3D" / ~r"[\u0B5C-\u0B5D]" / ~r"[\u0B5F-\u0B61]" / "\u0B71" / "\u0B83" / ~r"[\u0B85-\u0B8A]" / ~r"[\u0B8E-\u0B90]" / ~r"[\u0B92-\u0B95]" / ~r"[\u0B99-\u0B9A]" / "\u0B9C" / ~r"[\u0B9E-\u0B9F]" / ~r"[\u0BA3-\u0BA4]" / ~r"[\u0BA8-\u0BAA]" / ~r"[\u0BAE-\u0BB9]" / ~r"[\u0C05-\u0C0C]" / ~r"[\u0C0E-\u0C10]" / ~r"[\u0C12-\u0C28]" / ~r"[\u0C2A-\u0C33]" / ~r"[\u0C35-\u0C39]" / ~r"[\u0C60-\u0C61]" / ~r"[\u0C85-\u0C8C]" / ~r"[\u0C8E-\u0C90]" / ~r"[\u0C92-\u0CA8]" / ~r"[\u0CAA-\u0CB3]" / ~r"[\u0CB5-\u0CB9]" / "\u0CBD" / "\u0CDE" / ~r"[\u0CE0-\u0CE1]" / ~r"[\u0D05-\u0D0C]" / ~r"[\u0D0E-\u0D10]" / ~r"[\u0D12-\u0D28]" / ~r"[\u0D2A-\u0D39]" / ~r"[\u0D60-\u0D61]" / ~r"[\u0D85-\u0D96]" / ~r"[\u0D9A-\u0DB1]" / ~r"[\u0DB3-\u0DBB]" / "\u0DBD" / ~r"[\u0DC0-\u0DC6]" / ~r"[\u0E01-\u0E30]" / ~r"[\u0E32-\u0E33]" / ~r"[\u0E40-\u0E45]" / "\u0E46" / ~r"[\u0E81-\u0E82]" / "\u0E84" / ~r"[\u0E87-\u0E88]" / "\u0E8A" / "\u0E8D" / ~r"[\u0E94-\u0E97]" / ~r"[\u0E99-\u0E9F]" / ~r"[\u0EA1-\u0EA3]" / "\u0EA5" / "\u0EA7" / ~r"[\u0EAA-\u0EAB]" / ~r"[\u0EAD-\u0EB0]" / ~r"[\u0EB2-\u0EB3]" / "\u0EBD" / ~r"[\u0EC0-\u0EC4]" / "\u0EC6" / ~r"[\u0EDC-\u0EDD]" / "\u0F00" / ~r"[\u0F40-\u0F47]" / ~r"[\u0F49-\u0F6A]" / ~r"[\u0F88-\u0F8B]" / ~r"[\u1000-\u1021]" / ~r"[\u1023-\u1027]" / ~r"[\u1029-\u102A]" / ~r"[\u1050-\u1055]" / ~r"[\u10A0-\u10C5]" / ~r"[\u10D0-\u10FA]" / "\u10FC" / ~r"[\u1100-\u1159]" / ~r"[\u115F-\u11A2]" / ~r"[\u11A8-\u11F9]" / ~r"[\u1200-\u1248]" / ~r"[\u124A-\u124D]" / ~r"[\u1250-\u1256]" / "\u1258" / ~r"[\u125A-\u125D]" / ~r"[\u1260-\u1288]" / ~r"[\u128A-\u128D]" / ~r"[\u1290-\u12B0]" / ~r"[\u12B2-\u12B5]" / ~r"[\u12B8-\u12BE]" / "\u12C0" / ~r"[\u12C2-\u12C5]" / ~r"[\u12C8-\u12D6]" / ~r"[\u12D8-\u1310]" / ~r"[\u1312-\u1315]" / ~r"[\u1318-\u135A]" / ~r"[\u1380-\u138F]" / ~r"[\u13A0-\u13F4]" / ~r"[\u1401-\u166C]" / ~r"[\u166F-\u1676]" / ~r"[\u1681-\u169A]" / ~r"[\u16A0-\u16EA]" / ~r"[\u16EE-\u16F0]" / ~r"[\u1700-\u170C]" / ~r"[\u170E-\u1711]" / ~r"[\u1720-\u1731]" / ~r"[\u1740-\u1751]" / ~r"[\u1760-\u176C]" / ~r"[\u176E-\u1770]" / ~r"[\u1780-\u17B3]" / "\u17D7" / "\u17DC" / ~r"[\u1820-\u1842]" / "\u1843" / ~r"[\u1844-\u1877]" / ~r"[\u1880-\u18A8]" / ~r"[\u1900-\u191C]" / ~r"[\u1950-\u196D]" / ~r"[\u1970-\u1974]" / ~r"[\u1980-\u19A9]" / ~r"[\u19C1-\u19C7]" / ~r"[\u1A00-\u1A16]" / ~r"[\u1D00-\u1D2B]" / ~r"[\u1D2C-\u1D61]" / ~r"[\u1D62-\u1D77]" / "\u1D78" / ~r"[\u1D79-\u1D9A]" / ~r"[\u1D9B-\u1DBF]" / ~r"[\u1E00-\u1E9B]" / ~r"[\u1EA0-\u1EF9]" / ~r"[\u1F00-\u1F15]" / ~r"[\u1F18-\u1F1D]" / ~r"[\u1F20-\u1F45]" / ~r"[\u1F48-\u1F4D]" / ~r"[\u1F50-\u1F57]" / "\u1F59" / "\u1F5B" / "\u1F5D" / ~r"[\u1F5F-\u1F7D]" / ~r"[\u1F80-\u1FB4]" / ~r"[\u1FB6-\u1FBC]" / "\u1FBE" / ~r"[\u1FC2-\u1FC4]" / ~r"[\u1FC6-\u1FCC]" / ~r"[\u1FD0-\u1FD3]" / ~r"[\u1FD6-\u1FDB]" / ~r"[\u1FE0-\u1FEC]" / ~r"[\u1FF2-\u1FF4]" / ~r"[\u1FF6-\u1FFC]" / "\u2071" / "\u207F" / ~r"[\u2090-\u2094]" / "\u2102" / "\u2107" / ~r"[\u210A-\u2113]" / "\u2115" / "\u2118" / ~r"[\u2119-\u211D]" / "\u2124" / "\u2126" / "\u2128" / ~r"[\u212A-\u212D]" / "\u212E" / ~r"[\u212F-\u2131]" / ~r"[\u2133-\u2134]" / ~r"[\u2135-\u2138]" / "\u2139" / ~r"[\u213C-\u213F]" / ~r"[\u2145-\u2149]" / ~r"[\u2160-\u2183]" / ~r"[\u2C00-\u2C2E]" / ~r"[\u2C30-\u2C5E]" / ~r"[\u2C80-\u2CE4]" / ~r"[\u2D00-\u2D25]" / ~r"[\u2D30-\u2D65]" / "\u2D6F" / ~r"[\u2D80-\u2D96]" / ~r"[\u2DA0-\u2DA6]" / ~r"[\u2DA8-\u2DAE]" / ~r"[\u2DB0-\u2DB6]" / ~r"[\u2DB8-\u2DBE]" / ~r"[\u2DC0-\u2DC6]" / ~r"[\u2DC8-\u2DCE]" / ~r"[\u2DD0-\u2DD6]" / ~r"[\u2DD8-\u2DDE]" / "\u3005" / "\u3006" / "\u3007" / ~r"[\u3021-\u3029]" / ~r"[\u3031-\u3035]" / ~r"[\u3038-\u303A]" / "\u303B" / "\u303C" / ~r"[\u3041-\u3096]" / ~r"[\u309B-\u309C]" / ~r"[\u309D-\u309E]" / "\u309F" / ~r"[\u30A1-\u30FA]" / ~r"[\u30FC-\u30FE]" / "\u30FF" / ~r"[\u3105-\u312C]" / ~r"[\u3131-\u318E]" / ~r"[\u31A0-\u31B7]" / ~r"[\u31F0-\u31FF]" / ~r"[\u3400-\u4DB5]" / ~r"[\u4E00-\u9FBB]" / ~r"[\uA000-\uA014]" / "\uA015" / ~r"[\uA016-\uA48C]" / ~r"[\uA800-\uA801]" / ~r"[\uA803-\uA805]" / ~r"[\uA807-\uA80A]" / ~r"[\uA80C-\uA822]" / ~r"[\uAC00-\uD7A3]" / ~r"[\uF900-\uFA2D]" / ~r"[\uFA30-\uFA6A]" / ~r"[\uFA70-\uFAD9]" / ~r"[\uFB00-\uFB06]" / ~r"[\uFB13-\uFB17]" / "\uFB1D" / ~r"[\uFB1F-\uFB28]" / ~r"[\uFB2A-\uFB36]" / ~r"[\uFB38-\uFB3C]" / "\uFB3E" / ~r"[\uFB40-\uFB41]" / ~r"[\uFB43-\uFB44]" / ~r"[\uFB46-\uFBB1]" / ~r"[\uFBD3-\uFD3D]" / ~r"[\uFD50-\uFD8F]" / ~r"[\uFD92-\uFDC7]" / ~r"[\uFDF0-\uFDFB]" / ~r"[\uFE70-\uFE74]" / ~r"[\uFE76-\uFEFC]" / ~r"[\uFF21-\uFF3A]" / ~r"[\uFF41-\uFF5A]" / ~r"[\uFF66-\uFF6F]" / "\uFF70" / ~r"[\uFF71-\uFF9D]" / ~r"[\uFF9E-\uFF9F]" / ~r"[\uFFA0-\uFFBE]" / ~r"[\uFFC2-\uFFC7]" / ~r"[\uFFCA-\uFFCF]" / ~r"[\uFFD2-\uFFD7]" / ~r"[\uFFDA-\uFFDC]"
    id_continue = id_start / ~r"[0-9]" / ~r"[\u0300-\u036F]" / ~r"[\u0483-\u0486]" / ~r"[\u0591-\u05B9]" / ~r"[\u05BB-\u05BD]" / "\u05BF" / ~r"[\u05C1-\u05C2]" / ~r"[\u05C4-\u05C5]" / "\u05C7" / ~r"[\u0610-\u0615]" / ~r"[\u064B-\u065E]" / ~r"[\u0660-\u0669]" / "\u0670" / ~r"[\u06D6-\u06DC]" / ~r"[\u06DF-\u06E4]" / ~r"[\u06E7-\u06E8]" / ~r"[\u06EA-\u06ED]" / ~r"[\u06F0-\u06F9]" / "\u0711" / ~r"[\u0730-\u074A]" / ~r"[\u07A6-\u07B0]" / ~r"[\u0901-\u0902]" / "\u0903" / "\u093C" / ~r"[\u093E-\u0940]" / ~r"[\u0941-\u0948]" / ~r"[\u0949-\u094C]" / "\u094D" / ~r"[\u0951-\u0954]" / ~r"[\u0962-\u0963]" / ~r"[\u0966-\u096F]" / "\u0981" / ~r"[\u0982-\u0983]" / "\u09BC" / ~r"[\u09BE-\u09C0]" / ~r"[\u09C1-\u09C4]" / ~r"[\u09C7-\u09C8]" / ~r"[\u09CB-\u09CC]" / "\u09CD" / "\u09D7" / ~r"[\u09E2-\u09E3]" / ~r"[\u09E6-\u09EF]" / ~r"[\u0A01-\u0A02]" / "\u0A03" / "\u0A3C" / ~r"[\u0A3E-\u0A40]" / ~r"[\u0A41-\u0A42]" / ~r"[\u0A47-\u0A48]" / ~r"[\u0A4B-\u0A4D]" / ~r"[\u0A66-\u0A6F]" / ~r"[\u0A70-\u0A71]" / ~r"[\u0A81-\u0A82]" / "\u0A83" / "\u0ABC" / ~r"[\u0ABE-\u0AC0]" / ~r"[\u0AC1-\u0AC5]" / ~r"[\u0AC7-\u0AC8]" / "\u0AC9" / ~r"[\u0ACB-\u0ACC]" / "\u0ACD" / ~r"[\u0AE2-\u0AE3]" / ~r"[\u0AE6-\u0AEF]" / "\u0B01" / ~r"[\u0B02-\u0B03]" / "\u0B3C" / "\u0B3E" / "\u0B3F" / "\u0B40" / ~r"[\u0B41-\u0B43]" / ~r"[\u0B47-\u0B48]" / ~r"[\u0B4B-\u0B4C]" / "\u0B4D" / "\u0B56" / "\u0B57" / ~r"[\u0B66-\u0B6F]" / "\u0B82" / ~r"[\u0BBE-\u0BBF]" / "\u0BC0" / ~r"[\u0BC1-\u0BC2]" / ~r"[\u0BC6-\u0BC8]" / ~r"[\u0BCA-\u0BCC]" / "\u0BCD" / "\u0BD7" / ~r"[\u0BE6-\u0BEF]" / ~r"[\u0C01-\u0C03]" / ~r"[\u0C3E-\u0C40]" / ~r"[\u0C41-\u0C44]" / ~r"[\u0C46-\u0C48]" / ~r"[\u0C4A-\u0C4D]" / ~r"[\u0C55-\u0C56]" / ~r"[\u0C66-\u0C6F]" / ~r"[\u0C82-\u0C83]" / "\u0CBC" / "\u0CBE" / "\u0CBF" / ~r"[\u0CC0-\u0CC4]" / "\u0CC6" / ~r"[\u0CC7-\u0CC8]" / ~r"[\u0CCA-\u0CCB]" / ~r"[\u0CCC-\u0CCD]" / ~r"[\u0CD5-\u0CD6]" / ~r"[\u0CE6-\u0CEF]" / ~r"[\u0D02-\u0D03]" / ~r"[\u0D3E-\u0D40]" / ~r"[\u0D41-\u0D43]" / ~r"[\u0D46-\u0D48]" / ~r"[\u0D4A-\u0D4C]" / "\u0D4D" / "\u0D57" / ~r"[\u0D66-\u0D6F]" / ~r"[\u0D82-\u0D83]" / "\u0DCA" / ~r"[\u0DCF-\u0DD1]" / ~r"[\u0DD2-\u0DD4]" / "\u0DD6" / ~r"[\u0DD8-\u0DDF]" / ~r"[\u0DF2-\u0DF3]" / "\u0E31" / ~r"[\u0E34-\u0E3A]" / ~r"[\u0E47-\u0E4E]" / ~r"[\u0E50-\u0E59]" / "\u0EB1" / ~r"[\u0EB4-\u0EB9]" / ~r"[\u0EBB-\u0EBC]" / ~r"[\u0EC8-\u0ECD]" / ~r"[\u0ED0-\u0ED9]" / ~r"[\u0F18-\u0F19]" / ~r"[\u0F20-\u0F29]" / "\u0F35" / "\u0F37" / "\u0F39" / ~r"[\u0F3E-\u0F3F]" / ~r"[\u0F71-\u0F7E]" / "\u0F7F" / ~r"[\u0F80-\u0F84]" / ~r"[\u0F86-\u0F87]" / ~r"[\u0F90-\u0F97]" / ~r"[\u0F99-\u0FBC]" / "\u0FC6" / "\u102C" / ~r"[\u102D-\u1030]" / "\u1031" / "\u1032" / ~r"[\u1036-\u1037]" / "\u1038" / "\u1039" / ~r"[\u1040-\u1049]" / ~r"[\u1056-\u1057]" / ~r"[\u1058-\u1059]" / "\u135F" / ~r"[\u1369-\u1371]" / ~r"[\u1712-\u1714]" / ~r"[\u1732-\u1734]" / ~r"[\u1752-\u1753]" / ~r"[\u1772-\u1773]" / "\u17B6" / ~r"[\u17B7-\u17BD]" / ~r"[\u17BE-\u17C5]" / "\u17C6" / ~r"[\u17C7-\u17C8]" / ~r"[\u17C9-\u17D3]" / "\u17DD" / ~r"[\u17E0-\u17E9]" / ~r"[\u180B-\u180D]" / ~r"[\u1810-\u1819]" / "\u18A9" / ~r"[\u1920-\u1922]" / ~r"[\u1923-\u1926]" / ~r"[\u1927-\u1928]" / ~r"[\u1929-\u192B]" / ~r"[\u1930-\u1931]" / "\u1932" / ~r"[\u1933-\u1938]" / ~r"[\u1939-\u193B]" / ~r"[\u1946-\u194F]" / ~r"[\u19B0-\u19C0]" / ~r"[\u19C8-\u19C9]" / ~r"[\u19D0-\u19D9]" / ~r"[\u1A17-\u1A18]" / ~r"[\u1A19-\u1A1B]" / ~r"[\u1DC0-\u1DC3]" / ~r"[\u203F-\u2040]" / "\u2054" / ~r"[\u20D0-\u20DC]" / "\u20E1" / ~r"[\u20E5-\u20EB]" / ~r"[\u302A-\u302F]" / ~r"[\u3099-\u309A]" / "\uA802" / "\uA806" / "\uA80B" / ~r"[\uA823-\uA824]" / ~r"[\uA825-\uA826]" / "\uA827" / "\uFB1E" / ~r"[\uFE00-\uFE0F]" / ~r"[\uFE20-\uFE23]" / ~r"[\uFE33-\uFE34]" / ~r"[\uFE4D-\uFE4F]" / ~r"[\uFF10-\uFF19]" / "\uFF3F"

    escape_group = "\"" ( "\\\"" / ~r"[^\"]" )* "\""
    _ = ~r"[\s\\]*"  # whitespace character
    """

    # replace any amount of whitespace  with a single space
    equation_str = equation_str.replace('\\t', ' ')
    equation_str = re.sub(r"\s+", ' ', equation_str)

    parser = parsimonious.Grammar(component_structure_grammar)
    tree = parser.parse(equation_str)

    class ComponentParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.subscripts = []
            self.real_name = None
            self.expression = None
            self.kind = None
            self.visit(ast)

        def visit_subscript_definition(self, n, vc):
            self.kind = 'subdef'

        def visit_lookup_definition(self, n, vc):
            self.kind = 'lookup'

        def visit_component(self, n, vc):
            self.kind = 'component'

        def visit_name(self, n, vc):
            (name,) = vc
            self.real_name = name.strip()

        def visit_subscript(self, n, vc):
            (subscript,) = vc
            self.subscripts.append(subscript.strip())

        def visit_expression(self, n, vc):
            self.expression = n.text.strip()

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

        def visit__(self, n, vc):
            return ' '

    parse_object = ComponentParser(tree)

    return {'real_name': parse_object.real_name,
            'subs': parse_object.subscripts,
            'expr': parse_object.expression,
            'kind': parse_object.kind}


def parse_units(units_str):
    """
    Extract and parse the units
    Extract the bounds over which the expression is assumed to apply.

    Parameters
    ----------
    units_str

    Returns
    -------

    Examples
    --------
    >>> parse_units('Widgets/Month [-10,10,1]')
    ('Widgets/Month', (-10,10,1))

    >>> parse_units('Month [0,?]')
    ('Month', [-10, None])

    >>> parse_units('Widgets [0,100]')
    ('Widgets', (0, 100))

    >>> parse_units('Widgets')
    ('Widgets', (None, None))

    >>> parse_units('[0, 100]')
    ('', (0, 100))

    """
    if not len(units_str):
        return units_str, (None, None)

    if units_str[-1] == ']':
        units, lims = units_str.rsplit('[')  # type: str, str
    else:
        units = units_str
        lims = '?, ?]'

    lims = tuple([float(x) if x.strip() != '?' else None for x in lims.strip(']').split(',')])

    return units.strip(), lims


functions = {
    # element-wise functions
    "abs": "abs",
    "integer": "int",
    "exp": "np.exp",
    "sin": "np.sin",
    "cos": "np.cos",
    "sqrt": "np.sqrt",
    "tan": "np.tan",
    "lognormal": "np.random.lognormal",
    "random normal":
        "functions.bounded_normal",
    "poisson": "np.random.poisson",
    "ln": "np.log",
    "log": "functions.log",
    "exprnd": "np.random.exponential",
    "random uniform": "functions.random_uniform",
    "sum": "np.sum",
    "arccos": "np.arccos",
    "arcsin": "np.arcsin",
    "arctan": "np.arctan",
    "if then else": "functions.if_then_else",
    "step": {
        "name": "functions.step",
        "require_time": True
    },
    "modulo": "np.mod",
    "pulse": {
        "name": "functions.pulse",
        "require_time": True
    },
    "pulse train": {
        "name": "functions.pulse_train",
        "require_time": True
    },
    "ramp": {
        "name": "functions.ramp",
        "require_time": True
    },
    "min": "np.minimum",
    "max": "np.maximum",
    "active initial": {
        "name": "functions.active_initial",
        "require_time": True
    },
    "xidz": "functions.xidz",
    "zidz": "functions.zidz",
    "game": "",  # In the future, may have an actual `functions.game` pass through

    # vector functions
    "vmin": "np.min",
    "vmax": "np.max",
    "prod": "np.prod"
}

builders = {
    "integ": lambda element, subscript_dict, args: builder.add_stock(
        identifier=element['py_name'],
        subs=element['subs'],
        expression=args[0],
        initial_condition=args[1],
        subscript_dict=subscript_dict
    ),

    "delay1": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[0],
        order='1',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "delay1i": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
        order='1',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "delay3": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[0],
        order='3',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "delay3i": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
        order='3',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "delay n": lambda element, subscript_dict, args: builder.add_n_delay(
        delay_input=args[0],
        delay_time=args[1],
        initial_value=args[2],
        order=args[3],
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smooth": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[0],
        order='1',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smoothi": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[2],
        order='1',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smooth3": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[0],
        order='3',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smooth3i": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[2],
        order='3',
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "smooth n": lambda element, subscript_dict, args: builder.add_n_smooth(
        smooth_input=args[0],
        smooth_time=args[1],
        initial_value=args[2],
        order=args[3],
        subs=element['subs'],
        subscript_dict=subscript_dict
    ),

    "trend": lambda element, subscript_dict, args: builder.add_n_trend(
        trend_input=args[0],
        average_time=args[1],
        initial_trend=args[2],
        subs=element['subs'],
        subscript_dict=subscript_dict),

    "initial": lambda element, subscript_dict, args: builder.add_initial(args[0]),

    "a function of": lambda element, subscript_dict, args: builder.add_incomplete(
        element['real_name'], args)
}


def parse_general_expression(element, namespace=None, subscript_dict=None, macro_list=None):
    """
    Parses a normal expression
    # its annoying that we have to construct and compile the grammar every time...

    Parameters
    ----------
    element: dictionary

    namespace : dictionary

    subscript_dict : dictionary

    macro_list: list of dictionaries
        [{'name': 'M', 'py_name':'m', 'filename':'path/to/file', 'args':['arg1', 'arg2']}]

    Returns
    -------
    translation

    new_elements: list of dictionaries
        If the expression contains builder functions, those builders will create new elements
        to add to our running list (that will eventually be output to a file) such as stock
        initialization and derivative funcs, etc.


    Examples
    --------
    >>> parse_general_expression({'expr': 'INTEG (FlowA, -10)',
    ...                           'py_name':'test_stock',
    ...                           'subs':None},
    ...                          {'FlowA': 'flowa'}),
    ({'kind': 'component', 'py_expr': "_state['test_stock']"},
     [{'kind': 'implicit',
       'subs': None,
       'doc': 'Provides initial conditions for test_stock function',
       'py_name': 'init_test_stock',
       'real_name': None,
       'unit': 'See docs for test_stock',
       'py_expr': '-10'},
      {'py_name': 'dtest_stock_dt',
       'kind': 'implicit',
       'py_expr': 'flowa',
       'real_name': None}])

    """
    if namespace is None:
        namespace = {}
    if subscript_dict is None:
        subscript_dict = {}

    in_ops = {
        "+": "+", "-": "-", "*": "*", "/": "/", "^": "**", "=": "==", "<=": "<=", "<>": "!=",
        "<": "<", ">=": ">=", ">": ">",
        ":and:": " and ", ":or:": " or "}  # spaces important for word-based operators

    pre_ops = {
        "-": "-", ":not:": " not ",  # spaces important for word-based operators
        "+": " "  # space is important, so that and empty string doesn't slip through generic
    }

    # in the following, if lists are empty use non-printable character
    # everything needs to be escaped before going into the grammar, in case it includes quotes
    sub_names_list = [re.escape(x) for x in subscript_dict.keys()] or ['\\a']
    sub_elems_list = [re.escape(y) for x in subscript_dict.values() for y in x] or ['\\a']
    ids_list = [re.escape(x) for x in namespace.keys()] or ['\\a']
    in_ops_list = [re.escape(x) for x in in_ops.keys()]
    pre_ops_list = [re.escape(x) for x in pre_ops.keys()]
    if macro_list is not None and len(macro_list) > 0:
        macro_names_list = [x['name'] for x in macro_list]
    else:
        macro_names_list = ['\\a']

    expression_grammar = r"""
    expr_type = array / expr / empty
    expr = _ pre_oper? _ (lookup_def / build_call / macro_call / lookup_call / call / parens / number / reference) _ (in_oper _ expr)?

    lookup_def = ~r"(WITH\ LOOKUP)"I _ "(" _ expr _ "," _ "(" _  ("[" ~r"[^\]]*" "]" _ ",")?  ( "(" _ expr _ "," _ expr _ ")" _ ","? _ )+ _ ")" _ ")"
    lookup_call = id _ "(" _ (expr _ ","? _)* ")"  # these don't need their args parsed...
    call = func _ "(" _ (expr _ ","? _)* ")"  # these don't need their args parsed...
    build_call = builder _ "(" _ arguments _ ")"
    macro_call = macro _ "(" _ arguments _ ")"
    parens   = "(" _ expr _ ")"

    arguments = (expr _ ","? _)*

    reference = id _ subscript_list?
    subscript_list = "[" _ ((sub_name / sub_element) _ ","? _)+ "]"

    array = (number _ ("," / ";")? _)+ !~r"."  # negative lookahead for anything other than an array
    number = ~r"\d+\.?\d*(e[+-]\d+)?"

    id = ~r"(%(ids)s)"I
    sub_name = ~r"(%(sub_names)s)"I  # subscript names (if none, use non-printable character)
    sub_element = ~r"(%(sub_elems)s)"I  # subscript elements (if none, use non-printable character)

    func = ~r"(%(funcs)s)"I  # functions (case insensitive)
    in_oper = ~r"(%(in_ops)s)"I  # infix operators (case insensitive)
    pre_oper = ~r"(%(pre_ops)s)"I  # prefix operators (case insensitive)
    builder = ~r"(%(builders)s)"I  # builder functions (case insensitive)
    macro = ~r"(%(macros)s)"I  # macros from model file (if none, use non-printable character)

    _ = ~r"[\s\\]*"  # whitespace character
    empty = "" # empty string
    """ % {
        # In the following, we have to sort keywords in decreasing order of length so that the
        # peg parser doesn't quit early when finding a partial keyword
        'sub_names': '|'.join(reversed(sorted(sub_names_list, key=len))),
        'sub_elems': '|'.join(reversed(sorted(sub_elems_list, key=len))),
        'ids': '|'.join(reversed(sorted(ids_list, key=len))),
        'funcs': '|'.join(reversed(sorted(functions.keys(), key=len))),
        'in_ops': '|'.join(reversed(sorted(in_ops_list, key=len))),
        'pre_ops': '|'.join(reversed(sorted(pre_ops_list, key=len))),
        'builders': '|'.join(reversed(sorted(builders.keys(), key=len))),
        'macros': '|'.join(reversed(sorted(macro_names_list, key=len)))
    }

    class ExpressionParser(parsimonious.NodeVisitor):
        # Todo: at some point, we could make the 'kind' identification recursive on expression,
        # so that if an expression is passed into a builder function, the information
        # about whether it is a constant, or calls another function, goes with it.
        def __init__(self, ast):
            self.translation = ""
            self.kind = 'constant'  # change if we reference anything else
            self.new_structure = []
            self.visit(ast)

        def visit_expr_type(self, n, vc):
            s = ''.join(filter(None, vc)).strip()
            self.translation = s

        def visit_expr(self, n, vc):
            s = ''.join(filter(None, vc)).strip()
            self.translation = s
            return s

        def visit_call(self, n, vc):
            self.kind = 'component'
            function_name = vc[0].lower()
            arguments = [e.strip() for e in vc[4].split(",")]
            return builder.build_function_call(functions[function_name], arguments)

        def visit_in_oper(self, n, vc):
            return in_ops[n.text.lower()]

        def visit_pre_oper(self, n, vc):
            return pre_ops[n.text.lower()]

        def visit_reference(self, n, vc):
            self.kind = 'component'
            id_str = vc[0]
            return id_str + '()'

        def visit_id(self, n, vc):
            return namespace[n.text]

        def visit_lookup_def(self, n, vc):
            """ This exists because vensim has multiple ways of doing lookups.
            Which is frustrating."""
            x_val = vc[4]
            pairs = vc[11]
            mixed_list = pairs.replace('(', '').replace(')', '').split(',')
            xs = mixed_list[::2]
            ys = mixed_list[1::2]
            string = "functions.lookup(%(x)s, [%(xs)s], [%(ys)s])" % {
                'x': x_val,
                'xs': ','.join(xs),
                'ys': ','.join(ys)
            }
            return string

        def visit_array(self, n, vc):
            if 'subs' in element and element['subs']:  # first test handles when subs is not defined
                coords = utils.make_coord_dict(element['subs'], subscript_dict, terse=False)
                dims = [utils.find_subscript_name(subscript_dict, sub) for sub in element['subs']]
                shape = [len(coords[dim]) for dim in dims]
                if ';' in n.text or ',' in n.text:
                    text = n.text.strip(';').replace(' ', '').replace(';', ',')
                    data = np.array([float(s) for s in text.split(',')]).reshape(shape)
                else:
                    data = np.tile(float(n.text), shape)
                datastr = np.array2string(data, separator=',').replace('\n', '').replace(' ', '')
                return textwrap.dedent("""\
                    xr.DataArray(data=%(datastr)s,
                                 coords=%(coords)s,
                                 dims=%(dims)s )""" % {
                    'datastr': datastr,
                    'coords': repr(coords),
                    'dims': repr(dims)})

            else:
                return n.text.replace(' ', '')

        def visit_subscript_list(self, n, vc):
            refs = vc[2]
            subs = [x.strip() for x in refs.split(',')]
            coordinates = utils.make_coord_dict(subs, subscript_dict)
            if len(coordinates):
                return '.loc[%s]' % repr(coordinates)
            else:
                return ' '

        def visit_build_call(self, n, vc):
            call = vc[0]
            arglist = vc[4]
            self.kind = 'component'

            builder_name = call.strip().lower()
            name, structure = builders[builder_name](element, subscript_dict, arglist)
            self.new_structure += structure
            return name

        def visit_macro_call(self, n, vc):
            call = vc[0]
            arglist = vc[4]
            self.kind = 'component'
            py_name = utils.make_python_identifier(call)[0]
            macro = [x for x in macro_list if x['py_name'] == py_name][0]  # should match once
            name, structure = builder.add_macro(macro['py_name'], macro['file_name'],
                                                macro['params'], arglist)
            self.new_structure += structure
            return name

        def visit_arguments(self, n, vc):
            arglist = [x.strip(',') for x in vc]
            return arglist

        def visit__(self, n, vc):
            """ Handles whitespace characters"""
            return ''

        def visit_empty(self, n, vc):
            return 'None'

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

    parser = parsimonious.Grammar(expression_grammar)
    tree = parser.parse(element['expr'])
    parse_object = ExpressionParser(tree)

    return ({'py_expr': parse_object.translation,
             'kind': parse_object.kind,
             'arguments': ''},
            parse_object.new_structure)


def parse_lookup_expression(element):
    """ This syntax parses lookups that are defined with their own element """

    lookup_grammar = r"""
    lookup = _ "(" _ "[" ~r"[^\]]*" "]" _ "," _ ( "(" _ number _ "," _ number _ ")" _ ","? _ )+ ")"
    number = ("+"/"-")? ~r"\d+\.?\d*(e[+-]\d+)?"
    _ = ~r"[\s\\]*"  # whitespace character
    """
    parser = parsimonious.Grammar(lookup_grammar)
    tree = parser.parse(element['expr'])

    class LookupParser(parsimonious.NodeVisitor):
        def __init__(self, ast):
            self.translation = ""
            self.new_structure = []
            self.visit(ast)

        def visit__(self, n, vc):
            # remove whitespace
            return ''

        def visit_lookup(self, n, vc):
            pairs = vc[9]
            mixed_list = pairs.replace('(', '').replace(')', '').split(',')
            xs = mixed_list[::2]
            ys = mixed_list[1::2]
            string = "functions.lookup(x, [%(xs)s], [%(ys)s])" % {
                'xs': ','.join(xs),
                'ys': ','.join(ys)
            }
            self.translation = string

        def generic_visit(self, n, vc):
            return ''.join(filter(None, vc)) or n.text

    parse_object = LookupParser(tree)
    return {'py_expr': parse_object.translation,
            'arguments': 'x'}


def translate_section(section, macro_list):
    model_elements = get_model_elements(section['string'])

    # extract equation components
    model_docstring = ''
    for entry in model_elements:
        if entry['kind'] == 'entry':
            entry.update(get_equation_components(entry['eqn']))
        elif entry['kind'] == 'section':
            model_docstring += entry['doc']

    # make python identifiers and track for namespace conflicts
    namespace = {'TIME': 'time', 'Time': 'time'}  # Initialize with builtins
    # add macro parameters when parsing a macro section
    for param in section['params']:
        name, namespace = utils.make_python_identifier(param, namespace)

    # add macro functions to namespace
    for macro in macro_list:
        if macro['name'] is not '_main_':
            name, namespace = utils.make_python_identifier(macro['name'], namespace)

    # add model elements
    for element in model_elements:
        if element['kind'] not in ['subdef', 'section']:
            element['py_name'], namespace = utils.make_python_identifier(element['real_name'],
                                                                         namespace)

    # Create a namespace for the subscripts
    # as these aren't used to create actual python functions, but are just labels on arrays,
    # they don't actually need to be python-safe
    subscript_dict = {e['real_name']: e['subs'] for e in model_elements if e['kind'] == 'subdef'}

    # Parse components to python syntax.
    for element in model_elements:
        if element['kind'] == 'component' and 'py_expr' not in element:
            # Todo: if there is new structure, it should be added to the namespace...
            translation, new_structure = parse_general_expression(element,
                                                                  namespace=namespace,
                                                                  subscript_dict=subscript_dict,
                                                                  macro_list=macro_list)
            element.update(translation)
            model_elements += new_structure

        elif element['kind'] == 'lookup':
            element.update(parse_lookup_expression(element))

    # send the pieces to be built
    build_elements = [e for e in model_elements if e['kind'] not in ['subdef', 'section']]
    builder.build(build_elements,
                  subscript_dict,
                  namespace,
                  section['file_name'])

    return section['file_name']


def translate_vensim(mdl_file):
    """

    Parameters
    ----------
    mdl_file : basestring
        file path of a vensim model file to translate to python

    Returns
    -------

    Examples
    --------
    >>> translate_vensim('../tests/test-models/tests/subscript_3d_arrays/test_subscript_3d_arrays.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/abs/test_abs.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/exponentiation/exponentiation.mdl')

    #>>> translate_vensim('../../tests/test-models/tests/limits/test_limits.mdl')

    """
    with open(mdl_file, 'r', encoding='UTF-8') as in_file:
        text = in_file.read()

    outfile_name = mdl_file.replace('.mdl', '.py')
    out_dir = os.path.dirname(outfile_name)

    # extract model elements
    file_sections = get_file_sections(text.replace('\n', ''))
    for section in file_sections:
        if section['name'] == '_main_':
            section['file_name'] = outfile_name
        else:  # separate macro elements into their own files
            section['py_name'] = utils.make_python_identifier(section['name'])[0]
            section['file_name'] = out_dir + '/' + section['py_name'] + '.py'

    macro_list = [s for s in file_sections if s['name'] is not '_main_']

    for section in file_sections:
        translate_section(section, macro_list)

    return outfile_name
