import wx
import cv2
import numpy as np
from typing import List, Any, Tuple, Optional, Sequence
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.point_edit import PointEditor, PointViewNEdit
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.probability_displayer import ProbabilityDisplayer
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.scroll_image_list import ScrollImageList
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.video_player import VideoController
from wx.lib.scrolledpanel import ScrolledPanel
from collections import deque


class FBEditor(wx.Frame):
    def __init__(self, parent, video_hdl: cv2.VideoCapture, data: np.ndarray, poses: Pose, names: List[str],
                 w_id=wx.ID_ANY, title="", pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE,
                 name="FBEditor"):
        super().__init__(parent, w_id, title, pos, size, style, name)

        self._main_panel = wx.Panel(self)
        self._main_sizer = wx.BoxSizer(wx.VERTICAL)
        self._main_sizer.Add(self._main_panel, 1, wx.EXPAND)
        self._splitter_sizer = wx.BoxSizer(wx.VERTICAL)

        self._main_splitter = wx.SplitterWindow(self._main_panel)
        self._sub_sizer = wx.BoxSizer(wx.VERTICAL)
        self._video_splitter = wx.SplitterWindow(self._main_splitter)
        self._side_sizer = wx.BoxSizer(wx.VERTICAL)
        self._sub_panel = wx.Panel(self._main_splitter)

        # Splitter specific settings...
        self._video_splitter.SetSashGravity(0.0)
        self._video_splitter.SetMinimumPaneSize(20)
        self._main_splitter.SetSashGravity(1.0)
        self._main_splitter.SetMinimumPaneSize(20)

        self.video_player = PointEditor(self._video_splitter, video_hdl=video_hdl, poses=poses, bp_names=names)
        self.video_controls = VideoController(self._sub_panel, video_player=self.video_player.video_viewer)

        self._prob_disp = MultiProbabilityDisplay(self._sub_panel, names, data)

        self._plot_panel = wx.Panel(self._video_splitter)

        self.plot_button = wx.Button(self._plot_panel, label="Plot This Frame")
        plot_imgs = [wx.Bitmap.FromRGBA(100, 100, 0, 0, 0, 0) for __ in data]
        self.plot_list = ScrollImageList(self._plot_panel, plot_imgs, wx.VERTICAL, size=wx.Size(400, -1))

        self._side_sizer.Add(self.plot_button, 0, wx.ALIGN_CENTER)
        self._side_sizer.Add(self.plot_list, 1, wx.EXPAND)
        self._plot_panel.SetSizerAndFit(self._side_sizer)

        self._video_splitter.SplitVertically(self._plot_panel, self.video_player, self._plot_panel.GetMinSize().GetWidth())

        self._sub_sizer.Add(self._prob_disp, 1, wx.EXPAND)
        self._sub_sizer.Add(self.video_controls, 0, wx.EXPAND)

        self._sub_panel.SetSizerAndFit(self._sub_sizer)

        self._main_splitter.SplitHorizontally(self._video_splitter, self._sub_panel, -self._sub_panel.GetMinSize().GetHeight())
        self._splitter_sizer.Add(self._main_splitter, 1, wx.EXPAND)

        self._main_panel.SetSizerAndFit(self._splitter_sizer)
        self.SetSizerAndFit(self._main_sizer)

        self.video_controls.Bind(PointViewNEdit.EVT_FRAME_CHANGE, self._on_frame_chg)

    def _build_toolbar(self):
        toolbar: wx.ToolBar = self.GetToolBar()

        # TODO: Need to actually make toolbar....

    def _on_frame_chg(self, evt: PointViewNEdit.FrameChangeEvent):
        for prob_disp in self.prob_displays:
            prob_disp.set_location(evt.frame)

    @property
    def prob_displays(self):
        return self._prob_disp.displays


class History:
    # TODO: Docs!!!
    class Element:
        def __init__(self, name: str, value: Any):
            self.name = name
            self.value = value

    def __init__(self, max_size: int = 100):
        self.history = deque(maxlen=max_size)
        self.future = deque(maxlen=max_size)

    def do(self, name: str, value: Any):
        self.future.clear()
        self.history.append(self.Element(name, value))

    def undo(self) -> Tuple[Optional[str], Optional[Any]]:
        if(self.can_undo()):
            result = self.history.pop()
            self.future.appendleft(result)
            return (result.name, result.value)
        else:
            return None, None

    def redo(self) -> Tuple[Optional[str], Optional[Any]]:
        if(self.can_redo()):
            result = self.future.popleft()
            self.history.append(result)
            return (result.name, result.value)
        else:
            return None, None

    def clear(self):
        self.future.clear()
        self.history.clear()

    def can_undo(self) -> bool:
        return (len(self.history) > 0)

    def can_redo(self) -> bool:
        return (len(self.future) > 0)


class MultiProbabilityDisplay(wx.Panel):
    # The number of probability displays to allow at max...
    MAX_HEIGHT_IN_WIDGETS = 4

    def __init__(self, parent, bp_names: List[str], data: np.ndarray, w_id=wx.ID_ANY, **kwargs):
        super().__init__(parent, w_id, **kwargs)

        self._main_sizer = wx.BoxSizer(wx.VERTICAL)

        self._scroll_panel = ScrolledPanel(self, style=wx.VSCROLL)
        self._scroll_sizer = wx.BoxSizer(wx.VERTICAL)

        self.displays = [ProbabilityDisplayer(self._scroll_panel, data=sub_data, text=name) for sub_data, name in
                         zip(data, bp_names)]

        for display in self.displays:
            self._scroll_sizer.Add(display, 0, wx.EXPAND)

        self._scroll_panel.SetSizer(self._scroll_sizer)
        self._scroll_panel.SetAutoLayout(True)
        self._scroll_panel.SetupScrolling()

        self._main_sizer.Add(self._scroll_panel, 1, wx.EXPAND)

        self.SetSizer(self._main_sizer)
        self.SetMinSize(wx.Size(
            max(disp.GetMinSize().GetWidth() for disp in self.displays),
            sum(disp.GetMinSize().GetHeight() for disp in self.displays[:self.MAX_HEIGHT_IN_WIDGETS]))
        )