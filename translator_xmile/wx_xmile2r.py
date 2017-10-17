#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# generated by wxGlade not found on Sun Oct  1 01:20:10 2017
#

import wx
import os
import xmile2py

# begin wxGlade: dependencies
import gettext
# end wxGlade

# begin wxGlade: extracode
# end wxGlade


class Xmile2RFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: Xmile2RFrame.__init__
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.text_ctrl_modelo = wx.TextCtrl(self, wx.ID_ANY, "")
        self.text_ctrl_resutado = wx.TextCtrl(self, wx.ID_ANY, "", style=wx.TE_MULTILINE)
        self.button_1 = wx.Button(self, wx.ID_ANY, _("Traducir"), style=wx.BU_AUTODRAW)
        self.button_2 = wx.Button(self, wx.ID_ANY, _("Cerrar"), style=wx.BU_AUTODRAW)

        self.__set_properties()
        self.__do_layout()

        # Localización de trabajo
        self.abspath = os.path.abspath(__file__)
        self.dname = os.path.dirname(self.abspath)
        os.chdir(self.dname)
        self.modelo = "<Da enter para elegir el modelo Stella>"
        self.text_ctrl_modelo.SetLabelText(self.modelo)

        self.Bind(wx.EVT_TEXT_ENTER, self.al_cambiar_texto, self.text_ctrl_modelo)
        self.Bind(wx.EVT_BUTTON, self.al_traducir, self.button_1)
        self.Bind(wx.EVT_BUTTON, self.al_cancelar, self.button_2)
        # end wxGlade

    def __set_properties(self):
        # begin wxGlade: Xmile2RFrame.__set_properties
        self.SetTitle(_("XMILE -> R"))
        _icon = wx.NullIcon
        _icon.CopyFromBitmap(wx.Bitmap("Butterfly.ico", wx.BITMAP_TYPE_ANY))
        self.SetIcon(_icon)
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: Xmile2RFrame.__do_layout
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        sizer_2.Add((10, 10), 1, wx.EXPAND, 1)
        sizer_3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_3.Add((10, 10), 1, wx.EXPAND, 1)
        sizer_4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_6 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_5 = wx.BoxSizer(wx.HORIZONTAL)
        label_modelo = wx.StaticText(self, wx.ID_ANY, _("Elige el modelo:"))
        sizer_3.Add(label_modelo, 1, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_3.Add(self.text_ctrl_modelo, 3, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_3.Add((10, 10), 1, wx.EXPAND, 0)
        sizer_2.Add(sizer_3, 0, wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 0)
        sizer_2.Add((10, 10), 0, wx.EXPAND, 0)
        sizer_5.Add((10, 10), 1, wx.EXPAND, 0)
        label_resultado = wx.StaticText(self, wx.ID_ANY, _("Resultado:"))
        sizer_5.Add(label_resultado, 10, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 1)
        sizer_5.Add((10, 10), 10, wx.EXPAND, 0)
        sizer_2.Add(sizer_5, 1, wx.EXPAND, 0)
        static_line_1 = wx.StaticLine(self, wx.ID_ANY)
        sizer_2.Add(static_line_1, 0, wx.EXPAND, 0)
        sizer_6.Add((5, 50), 1, wx.EXPAND, 0)
        sizer_6.Add(self.text_ctrl_resutado, 50, wx.ALL | wx.EXPAND,2)
        sizer_6.Add((5, 50), 1, wx.EXPAND, 0)
        sizer_2.Add(sizer_6, 6, wx.EXPAND, 0)
        static_line_2 = wx.StaticLine(self, wx.ID_ANY)
        sizer_2.Add(static_line_2, 0, wx.EXPAND, 0)
        sizer_2.Add((20, 13), 0, wx.EXPAND, 0)
        sizer_4.Add((20, 20), 1, wx.EXPAND, 0)
        sizer_4.Add(self.button_1, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_4.Add((20, 20), 1, wx.EXPAND, 0)
        sizer_4.Add(self.button_2, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_4.Add((20, 20), 1, wx.EXPAND, 0)
        sizer_2.Add(sizer_4, 0, wx.EXPAND, 0)
        sizer_2.Add((20, 10), 1, wx.EXPAND, 0)
        sizer_1.Add(sizer_2, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()
        # end wxGlade

    def al_cambiar_texto(self, event):  # wxGlade: Xmile2RFrame.<event_handler>
        openFileDialog = wx.FileDialog(self, "Abrir", "", "",
                                       "Archivos Stella (*.stmx)|*.stmx",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        openFileDialog.ShowModal()
        self.modelo = openFileDialog.GetPath()
        openFileDialog.Destroy()
        os.chdir(os.path.dirname(self.modelo))
        self.modelo = os.path.basename(self.modelo)
        self.text_ctrl_modelo.SetLabelText(self.modelo)


    def al_traducir(self, event):  # wxGlade: Xmile2RFrame.<event_handler>
        # exctract data from command line input
        outputName = os.path.abspath(".") + "\\" + self.modelo.split(".")[0]
        outPath = os.path.abspath(outputName)

        # Prepare directory to store output
        if (not os.path.isdir(outPath)):
            os.mkdir(outPath)

        # Extract relevant information to build model
        model_translation = xmile2py.xmile_parser(self.modelo)

        # output model file
        with open(outPath + "/" + model_translation.name + ".txt", "w") as f_output:
            model_output = model_translation.show()
            f_output.writelines(model_output)

        (r_model_solver, r_model_calibrate) = model_translation.build_R_script()
        r_model_setwd = "".join(["#Working directory\n", "setwd(\"", outPath.replace("\\", "/"), "\")", "\n\n"])
        file_name = model_translation.name + "_solver.R"
        with open(outPath + "/" + file_name, "w") as f_output:
            f_output.writelines(r_model_setwd + r_model_solver)

        r_model_calibrate = r_model_calibrate.replace("file_path", '"' + file_name + '"')
        with open(outPath + "/" + file_name.replace("solver", "calibrate"), "w") as f_output:
            f_output.writelines(r_model_setwd + r_model_calibrate)

        self.text_ctrl_resutado.SetValue("".join(["*" * 28,
                                                  "\nModel translation successful",
                                                  "\nModel processed:", model_translation.name,
                                                  "\nTranslation can be found at:\n  ", outPath,
                                                  "\n\n"]))

    def al_cancelar(self, event):  # wxGlade: Xmile2RFrame.<event_handler>
        self.Close()

# end of class Xmile2RFrame
class Wx_Xmile2Py(wx.App):
    def OnInit(self):
        self.frame = Xmile2RFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True

# end of class Wx_Xmile2Py

if __name__ == "__main__":
    gettext.install("app") # replace with the appropriate catalog name

    app = Wx_Xmile2Py(0)
    app.MainLoop()