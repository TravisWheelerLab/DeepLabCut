import os
# We first check if this is a headless environment, and if so don't even allow this module to be imported...
if os.environ.get('DLClight', default=False) == 'True':
    raise ImportError("Can't use this module in DLClight mode!")

# For types in methods
from typing import Union, List, Tuple, Any, Dict, Callable, Optional

from matplotlib.patches import Circle
from tqdm import tqdm

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.fastviterbi import FastViterbi, SparseTrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose

# For computations
import numpy as np

# For custom point selection UI....
import wx
from matplotlib.backends.backend_wxagg import FigureFrameWxAgg
from matplotlib.figure import Figure
import matplotlib
from matplotlib.backend_tools import ToolBase, ToolToggleBase


class MouseButton:
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3

# Path manipulation...
from pathlib import Path

# Collecting video frames...
import cv2


from collections import namedtuple

# MAIN UI DISPLAY CODE BELOW:

# For storing frame info in the gui really just a convenient way of providing names to the different parts of the frame.
FrameInfo = namedtuple("FrameInfo", ["prev", "current", "next"])

class PointPicker:
    """
    Allows user to pick a point location using matplotlib...
    """

    APP = wx.App()

    def __init__(self, main_title: str, titles, frames, predicted_locations):
        matplotlib.rcParams["toolbar"] = "toolmanager"
        self._frames = FrameInfo(*frames)

        self._loc = FrameInfo(*predicted_locations)
        self._titles = FrameInfo(*titles)

        self._current_loc = self._loc.current

        self._figure = Figure()
        self._grid = self._figure.add_gridspec(4, 4)
        ax = (self._figure.add_subplot(self._grid[0, :2]),
              self._figure.add_subplot(self._grid[1:, :]),
              self._figure.add_subplot(self._grid[0, 2:]))

        self._axes = FrameInfo(*ax)

        self._figure.suptitle(main_title)

        circles = []

        for ax, title, image, point_loc, in zip(self._axes, self._titles, self._frames, self._loc):
            ax.set_title(title)
            if(image is not None):
                ax.imshow(image, aspect="equal")
            if(point_loc is None):
                point_loc = (-5, -5)

            circle = Circle(point_loc, 0.02 * self._frames.current.shape[0], clip_on=True, color=(0, 0, 1, 0.5),
                                picker=True)
            ax.add_artist(circle)
            circles.append(circle)

        self._figure.tight_layout()

        self._points = FrameInfo(*circles)

        self._wx_frame = FigureFrameWxAgg(-1, self._figure)

        # Grab the toolbar and tool manager, we will hack in our own button using it!
        self._toolmgr = self._wx_frame.toolmanager
        self._toolbar = self._wx_frame.toolbar

        self._mode = None

        def set_mode(mode):
            self._mode = mode

        class EditTool(ToolToggleBase):
            name = "Edit Point"
            description = "Edits a Point. Click to place and right click to delete..."
            default_keymap = "Ctrl+E"
            radio_group = "default"

            def enable(self, event):
                set_mode("Edit")

            def disable(self, event):
                set_mode(None)

        self._toolmgr.add_tool("Edit Point", EditTool)
        self._toolbar.add_tool(self._toolmgr.get_tool("Edit Point"), "Edit Point")

        class SubmitBtn(ToolBase):
            name = "Submit"
            description = "Submit point data now!"

            APP = self.APP
            WINDOW = self._wx_frame

            def trigger(self, sender, event, data=None):
                self.WINDOW.Destroy()
                self.APP.ExitMainLoop()

        self._toolmgr.add_tool("Submit", SubmitBtn)
        self._toolbar.add_tool(self._toolmgr.get_tool("Submit"), "Submit")

        def pick_evt(event):
            if (event.artist == self._points.current and
                    (event.mouseevent.button == MouseButton.RIGHT) and
                    (self._mode == "Edit")):
                self._points.current.set_visible(False)
                self._current_loc = None
                self._wx_frame.canvas.draw_idle()

        self._wx_frame.canvas.mpl_connect("pick_event", pick_evt)

        def release_evt(event):
            if (event.button == MouseButton.LEFT and event.inaxes == self._axes.current and self._mode == "Edit"):
                if (0 <= event.xdata <= self._frames.current.shape[1]):
                    if (0 <= event.ydata <= self._frames.current.shape[0]):
                        self._points.current.set_visible(True)
                        self._points.current.set_center((event.xdata, event.ydata))
                        self._current_loc = (event.xdata, event.ydata)
                        self._wx_frame.canvas.draw_idle()

        self._wx_frame.canvas.mpl_connect("button_release_event", release_evt)

    def show(self):
        self.APP.SetExitOnFrameDelete(True)
        self._wx_frame.Show(True)
        self.APP.SetTopWindow(self._wx_frame)
        self.APP.MainLoop()
        return self.get_selected_point()

    def get_selected_point(self):
        return self._current_loc


class SupervisedViterbi(FastViterbi):
    """
    A predictor that applies the viterbi algorithm to frames in order to predict poses.
    The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but
    is also more memory intensive and computationally expensive. This specific implementation
    asks for user feedback on frames it is unsure about and then reruns the algorithm for better
    results.
    """
    __TITLES = ["Previous Frame", "Click to move point, right click to delete.", "Next Frame"]

    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int,
                 settings: Union[Dict[str, Any], None], video_metadata: Dict[str, Any]):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)

        self.MODES = {
            "on_probability_drop": self._on_prob_drop,
            "threshold": self._on_low_prob
        }

        # The passed detection algorithm and it's arguments.
        self.DETECTION_ALGORITHM = settings["detection_algorithm"][0]
        self.DETECTION_ARGS = settings["detection_algorithm"][1:]

        if(self.DETECTION_ALGORITHM not in self.MODES):
            raise ValueError(f"'{self.DETECTION_ALGORITHM}' is not a known detection algorithm!")

        self.__bodyparts = bodyparts

        self.__video_path = Path(video_metadata["orig-video-path"])
        self.__frames = FrameInfo(None, None, None)

        # We need to read in 2 frames before __frame.current becomes frame 0.
        self.__current_video_frame = -2


    def _on_prob_drop(self, args: List[str], poses: Pose) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the frames and
        """
        drop_value = float(args[0]) if(len(args) > 0 and (0 < float(args[0]) < 1)) else 0.5
        probs = np.array(self._viterbi_probs)

        diff_forward = np.concatenate(np.zeros((1, probs.shape[1])), probs[1:] - probs[:-1], axis=1)
        diff_forward = np.nonzero(diff_forward >= drop_value)

        return diff_forward


    def _on_low_prob(self, args: List[str], poses: Pose) -> Tuple[np.ndarray, np.ndarray]:
        threshold = float(args[0]) if(len(args) > 0 and (0 < float(args[0]) < 1)) else 0.1
        probs = np.array(self._viterbi_probs)

        return np.nonzero(probs < threshold)


    def _eat_frames(self, capture: cv2.VideoCapture, goto_frame: int) -> FrameInfo:
        """
        Private method: Eat frames, storing in the frames structure until we reach the desired frame.....
        """
        while(self.__current_video_frame < goto_frame):
            prev = self.__frames.current
            current = self.__frames.next

            if(capture.isOpened()):
                # We ignore the return value....
                __, next_f = capture.read()
            else:
                next_f = None

            self.__frames = FrameInfo(prev, current, next_f)
            self.__current_video_frame += 1

        return self.__frames


    def on_end(self, progress_bar: tqdm) -> Optional[Pose]:
        poses = super()._backward(progress_bar)
        # Reset the current frame....
        self._current_frame = 0

        # Run the selected detection algorithm:
        print("Looking for poorly labeled frames: ")
        relabel_frames = np.transpose(self.MODES[self.DETECTION_ALGORITHM](self.DETECTION_ARGS, poses))
        relabel_frames = relabel_frames[np.argsort(relabel_frames[:, 0])]

        # Ask user to label bad frames, correct them....
        print("Labeling frames: ")

        cap = cv2.VideoCapture(str(self.__video_path))

        for frame, bp in tqdm(relabel_frames):
            frames = self._eat_frames(cap, frame)

            bp_name = self.__bodyparts[int(bp // self._num_outputs)]
            bp_number = (bp % self._num_outputs) + 1

            def limit(val):
                return min(max(0, val), len(self._sparse_data) - 1)

            locations = ((poses.get_x_at(limit(frame + off), bp), poses.get_y_at(limit(frame + off), bp)) for off in [-1, 0, 1])

            # Showing the point picker and getting user feedback...
            point_editor = PointPicker(f"Frame {frame}, {bp_name} {bp_number}", self.__TITLES, frames, locations)
            point = point_editor.show()

            if(point is None):
                # If user didn't select a point, duplicate prior frame, or if we are on the first frame set it to None.
                if(frame > 0):
                    self._sparse_data[frame][bp] = self._sparse_data[frame - 1][bp]
                else:
                    self._sparse_data[frame][bp] = SparseTrackingData()
                    self._sparse_data[frame][bp].pack(*[np.array([0]) for __ in range(5)])
            else:
                # User picked a spot, set it as the only point in this frame with 100% probability...
                x = np.array([int(point[0] // self._down_scaling)])
                y = np.array([int(point[1] // self._down_scaling)])
                off_x = np.array([(point[0] % self._down_scaling) - (0.5 * self._down_scaling)])
                off_y = np.array([(point[1] % self._down_scaling) - (0.5 * self._down_scaling)])
                prob = np.array([1])

                self._sparse_data[frame][bp].pack(y, x, prob, off_x, off_y)

        # Gotta release the video reader...
        cap.release()

        # Redo forward and backward:
        print("Rerunning Forward:")
        for __ in tqdm(range(self._num_frames)):
            self._forward_step(1)

        print("Rerunning Backward:")
        return self._backward(tqdm(total=self._num_frames))


    @classmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        return super().get_settings() + [
            ("detection_algorithm", "A list of values, determines the detection algorithm to use and the arguments "
                                    "to pass to it. The available options are:\n"
                                    " - ['threshold', (optional float 0 < x < 1)]: Select frames that fall below a"
                                    "certain probability.\n"
                                    " - ['on_probability_drop', (optional float 0 < x < 1)]: Select frames on which "
                                    "the probability has dropped by the specified amount.",
             ["threshold", 0.1])
        ]

    @staticmethod
    def get_name() -> str:
        return "supervised_viterbi"

    @staticmethod
    def get_description() -> str:
        return ("A predictor that applies the viterbi algorithm to frames in order to predict poses. "
                "The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but "
                "is also more memory intensive and computationally expensive. This specific implementation "
                "asks for user feedback on frames it is unsure about and then reruns the algorithm for better "
                "results.")

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        """
        Run tests for "fast_viterbi" to test this plugin...
        """
        print("Run tests for 'fast_viterbi' to test this plugin...")
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True
