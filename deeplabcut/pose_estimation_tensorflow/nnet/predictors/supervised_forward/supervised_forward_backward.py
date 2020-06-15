from typing import Union, List, Callable, Tuple, Any, Dict, Optional
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.forward_backward import ForwardBackward
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.video_player import VideoPlayer, VideoController
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.probability_displayer import ProbabilityDisplayer
import wx
from wx.lib.statbmp import GenStaticBitmap
import cv2
import numpy as np
import tqdm
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


class SupervisedForwardBackward(ForwardBackward):

    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int,
                 settings: Union[Dict[str, Any], None], video_metadata: Optional[Dict[str, Any]]):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)

        if(video_metadata["orig-video-path"] is None):
            raise ValueError("Unable to find the original video file, which is required by this plugin!")

        self._video_path = video_metadata["orig-video-path"]
        self._video_hdl: Optional[cv2.VideoCapture] = None  # Will soon become a cv2 videocapture
        self._final_probabilities = None
        self._fb_editor: Optional[FBEditor] = None
        self._bp_names = bodyparts


    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        progress_bar.reset(total=self._num_frames * (7 if (self.NEGATE_ON) else 3) - (self.NEGATE_ON * 2))
        self._complete_posterior_probs(progress_bar)

        if (self.NEGATE_ON):
            self._bp_negation_pass(progress_bar)
            self._post_forward_backward(progress_bar, self._gaussian_values_at)

        poses, indexes = self._get_maximums(progress_bar, include_indexes=True)
        probs = np.transpose(poses.get_all_prob())

        self._video_hdl = cv2.VideoCapture(self._video_path)

        app = wx.App()

        self._fb_editor = FBEditor(None, self._video_hdl, probs)

        self._fb_editor.plot_button.Bind(wx.EVT_BUTTON, self._make_plots)

        self._fb_editor.Show()

        app.MainLoop()

        return poses

    def _make_plots(self, evt):
        frame_idx = self._fb_editor.video_player.get_offset_count()
        for bp_idx, plot_img in zip(range(self._total_bp_count), self._fb_editor.plot_images):
            figure = plt.figure(figsize=(1.5, 1.15), dpi=100)
            axes = figure.gca()

            frames, edges = self._frame_probs[frame_idx][bp_idx], self._edge_vals[frame_idx, bp_idx]
            all_values = np.concatenate((frames, edges))
            axes.set_title(self._bp_names[bp_idx // self._num_outputs] + str((bp_idx % self._num_outputs) + 1))
            axes.hist(all_values, bins=10)

            plt.tight_layout()
            figure.canvas.draw()

            w, h = figure.canvas.get_width_height()
            new_bitmap = wx.Bitmap.FromBufferRGBA(w, h, figure.canvas.buffer_rgba())

            plot_img.SetBitmap(new_bitmap)
            plot_img.Refresh()

    @classmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        return super().get_settings()

    @classmethod
    def get_name(cls) -> str:
        return "supervised_forward_backward"

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True


class FBEditor(wx.Frame):
    def __init__(self, parent, video_hdl: cv2.VideoCapture, data: np.ndarray, id=wx.ID_ANY, title="",
                 pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, name="FBEditor"):
        super().__init__(parent, id, title, pos, size, style, name)

        self._main_panel = wx.Panel(self)
        self._main_sizer = wx.BoxSizer(wx.VERTICAL)
        self._main_sizer.Add(self._main_panel, 1, wx.EXPAND)

        self._sub_sizer = wx.BoxSizer(wx.VERTICAL)
        self._video_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._side_sizer = wx.BoxSizer(wx.VERTICAL)

        self.video_player = VideoPlayer(self._main_panel, video_hdl=video_hdl)
        self.video_controls = VideoController(self._main_panel, video_player=self.video_player)

        self.prob_displays = [ProbabilityDisplayer(self._main_panel, data=sub_data) for sub_data in data]

        self.plot_button = wx.Button(self._main_panel, label="Plot This Frame")
        self.plot_images = [GenStaticBitmap(self._main_panel, wx.ID_ANY, bitmap=wx.Bitmap.FromRGBA(100, 100, 0, 0, 0, 0)) for __ in data]

        self._side_sizer.Add(self.plot_button)
        for img in self.plot_images:
            self._side_sizer.Add(img, 0, wx.EXPAND)

        self._video_sizer.Add(self._side_sizer, 0, wx.EXPAND)
        self._video_sizer.Add(self.video_player, 1, wx.EXPAND)

        self._sub_sizer.Add(self._video_sizer, 1, wx.EXPAND)
        for prob_display in self.prob_displays:
            self._sub_sizer.Add(prob_display, 0, wx.EXPAND)
        self._sub_sizer.Add(self.video_controls, 0, wx.EXPAND)

        self._main_panel.SetSizerAndFit(self._sub_sizer)
        self.SetSizerAndFit(self._main_sizer)

        self.video_controls.Bind(VideoPlayer.EVT_FRAME_CHANGE, self._on_frame_chg)

    def _on_frame_chg(self, evt: VideoPlayer.FrameChangeEvent):
        for prob_disp in self.prob_displays:
            prob_disp.set_location(evt.frame)
