from typing import Union, List, Callable, Tuple, Any, Dict, Optional
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.forward_backward import ForwardBackward, SparseTrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.video_player import VideoPlayer, VideoController
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.probability_displayer import ProbabilityDisplayer
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.scroll_image_list import ScrollImageList
import wx
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

        self._fb_editor = FBEditor(None, self._video_hdl, probs, self._get_names())

        self._fb_editor.plot_button.Bind(wx.EVT_BUTTON, self._make_plots)

        self._fb_editor.Show()

        app.MainLoop()

        return poses

    def _get_names(self):
        return [self._bp_names[bp_idx // self._num_outputs] + str((bp_idx % self._num_outputs) + 1)
                for bp_idx in range(self._total_bp_count)]

    def _make_plots(self, evt):
        frame_idx = self._fb_editor.video_player.get_offset_count()

        new_bitmap_list = []

        for bp_idx in range(self._total_bp_count):
            bp_name = self._bp_names[bp_idx // self._num_outputs] + str((bp_idx % self._num_outputs) + 1)
            figure = plt.figure(figsize=(3.6, 2.8), dpi=100)
            axes = figure.gca()

            frames, edges = self._frame_probs[frame_idx][bp_idx], self._edge_vals[frame_idx, bp_idx]

            if(frames is None):
                continue

            all_values = np.concatenate((frames, edges))
            axes.set_title(bp_name)
            axes.hist(all_values, bins=np.arange(11) / 10)

            plt.tight_layout()
            figure.canvas.draw()

            w, h = figure.canvas.get_width_height()
            new_bitmap_list.append(wx.Bitmap.FromBufferRGBA(w, h, figure.canvas.buffer_rgba()))
            axes.cla()
            figure.clf()
            plt.close(figure)

            figure = plt.figure(figsize=(3.6, 2.8), dpi=100)
            axes = figure.gca()
            axes.set_title(bp_name)
            data = self._sparse_data[frame_idx][bp_idx].unpack()
            track_data = SparseTrackingData()
            track_data.pack(*data[:2], frames, *data[3:])
            h, w = self._gaussian_table.shape
            track_data = track_data.desparsify(w - 2, h - 2, 8)
            axes.pcolormesh(track_data.get_prob_table(0, 0))
            axes.set_ylim(axes.get_ylim()[::-1])
            plt.tight_layout()
            figure.canvas.draw()

            w, h = figure.canvas.get_width_height()
            new_bitmap_list.append(wx.Bitmap.FromBufferRGBA(w, h, figure.canvas.buffer_rgba()))
            axes.cla()
            figure.clf()
            plt.close(figure)

        self._fb_editor.plot_list.set_bitmaps(new_bitmap_list)

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
    def __init__(self, parent, video_hdl: cv2.VideoCapture, data: np.ndarray, names: List[str], w_id=wx.ID_ANY,
                 title="", pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, name="FBEditor"):
        super().__init__(parent, w_id, title, pos, size, style, name)

        self._main_panel = wx.Panel(self)
        self._main_sizer = wx.BoxSizer(wx.VERTICAL)
        self._main_sizer.Add(self._main_panel, 1, wx.EXPAND)

        self._sub_sizer = wx.BoxSizer(wx.VERTICAL)
        self._video_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._side_sizer = wx.BoxSizer(wx.VERTICAL)

        self.video_player = VideoPlayer(self._main_panel, video_hdl=video_hdl)
        self.video_controls = VideoController(self._main_panel, video_player=self.video_player)

        self.prob_displays = [ProbabilityDisplayer(self._main_panel, data=sub_data, text=name) for sub_data, name in zip(data, names)]

        self.plot_button = wx.Button(self._main_panel, label="Plot This Frame")
        plot_imgs = [wx.Bitmap.FromRGBA(100, 100, 0, 0, 0, 0) for __ in data]
        self.plot_list = ScrollImageList(self._main_panel, plot_imgs, wx.VERTICAL, size=wx.Size(400, -1))

        self._side_sizer.Add(self.plot_button, 0, wx.ALIGN_CENTER)
        self._side_sizer.Add(self.plot_list, 1, wx.EXPAND)

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
