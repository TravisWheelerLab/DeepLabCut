from typing import Union, List, Callable, Tuple, Any, Dict, Optional
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.forward_backward import ForwardBackward, SparseTrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.fb_editor import FBEditor
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

        self._vid_meta = video_metadata

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

        self._fb_editor = FBEditor(None, self._video_hdl, probs, poses, self._get_names())

        self._fb_editor.plot_button.Bind(wx.EVT_BUTTON, self._make_plots)

        self._fb_editor.Show()

        app.MainLoop()

        return poses

    def _get_names(self):
        return [self._bp_names[bp_idx // self._num_outputs] + str((bp_idx % self._num_outputs) + 1)
                for bp_idx in range(self._total_bp_count)]

    def _make_plots(self, evt):
        frame_idx = self._fb_editor.video_player.video_viewer.get_offset_count()

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
            axes.hist(all_values, bins=15)

            plt.tight_layout()
            figure.canvas.draw()

            w, h = figure.canvas.get_width_height()
            new_bitmap_list.append(wx.Bitmap.FromBufferRGBA(w, h, figure.canvas.buffer_rgba()))
            axes.cla()
            figure.clf()
            plt.close(figure)

            figure = plt.figure(figsize=(3.6, 2.8), dpi=100)
            axes = figure.gca()
            axes.set_title(bp_name + " Post Forward Backward")
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


            figure = plt.figure(figsize=(3.6, 2.8), dpi=100)
            axes = figure.gca()
            axes.set_title(bp_name + " Original Source Frame")
            h, w = self._gaussian_table.shape
            track_data = self._sparse_data[frame_idx][bp_idx].desparsify(w - 2, h - 2, 8)
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


