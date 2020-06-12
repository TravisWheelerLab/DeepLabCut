from ..forward_backward import ForwardBackward
from .video_player import VideoPlayer, VideoController
from .probability_displayer import ProbabilityDisplayer
import wx
from wx.lib.statbmp import GenStaticBitmap
import cv2
import numpy as np


class SupervisedForwardBackward(ForwardBackward):
    # TODO: Implement!!!!
    pass


class FBEditor(wx.Frame):
    def __init__(self, parent, video_hdl: cv2.VideoCapture, data: np.ndarray, id=wx.ID_ANY, title="",
                 pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, name="FBEditor"):
        super().__init__(parent, id, title, pos, size, style, name)

        self._main_panel = wx.Panel(self)
        self._main_sizer = wx.BoxSizer(wx.VERTICAL)
        self._main_sizer.Add(self._main_panel)

        self._sub_sizer = wx.BoxSizer(wx.VERTICAL)
        self._video_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._plot_sizer = wx.BoxSizer(wx.VERTICAL)

        self.video_player = VideoPlayer(self._main_panel, video_hdl=video_hdl)
        self.video_controls = VideoController(self._main_panel, video_player=self.video_player)

        self.prob_displays = [ProbabilityDisplayer(self._main_panel, data=sub_data) for sub_data in data]

        self.plot_button = wx.Button(self._main_panel, label="Plot This Frame")
        self.plot_images = [GenStaticBitmap(self._main_panel, wx.ID_ANY, bitmap=wx.NullBitmap) for __ in data]

        self._plot_sizer.Add(self.plot_button)
        for img in self.plot_images:
           self._plot_sizer.Add(img, 0, wx.EXPAND)

        self._video_sizer.Add(self._plot_sizer, 0, wx.EXPAND)
        self._video_sizer.Add(self.video_player, 1, wx.EXPAND)

        self._sub_sizer.Add(self._video_sizer, 1, wx.EXPAND)
        for prob_display in self.prob_displays:
            self._sub_sizer.Add(prob_display, 0, wx.EXPAND)
        self._sub_sizer.Add(self.video_controls, 0, wx.EXPAND)

        self._main_panel.SetSizerAndFit(self._sub_sizer)
        self.SetSizerAndFit(self._main_sizer)