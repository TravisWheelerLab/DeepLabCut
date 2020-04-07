# For types in methods
from typing import Union, List, Tuple, Any, Dict, Callable, Optional
from tqdm import tqdm

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.fastviterbi import FastViterbi
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose

# For computations
import numpy as np

# For custom point selection UI....
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from matplotlib.backend_bases import MouseButton
matplotlib.rcParams["toolbar"] = "toolmanager"

# Path manipulation...
from pathlib import Path

# Collecting video frames...
import cv2

from collections import namedtuple

# MAIN UI DISPLAY CODE BELOW:

# For storing frame info in the gui really just a convienent way of providing names to the different parts of the frame.
FrameInfo = namedtuple("FrameInfo", ["prev", "current", "next"])

class PointPicker:
    """
    Allows user to pick a point location using matplotlib...
    """

    def __init__(self, main_title: str, titles, frames, predicted_locations):
        self._frames = FrameInfo(*frames)

        self._loc = FrameInfo(*predicted_locations)
        self._titles = FrameInfo(*titles)

        self._current_loc = self._loc.current

        self._figure = plt.figure(constrained_layout=True)
        self._grid = self._figure.add_gridspec(4, 4)
        ax = (self._figure.add_subplot(self._grid[0, :2]),
              self._figure.add_subplot(self._grid[1:, :]),
              self._figure.add_subplot(self._grid[0, 2:]))

        self._axes = FrameInfo(*ax)

        self._figure.suptitle(main_title)

        circles = []

        for ax, title, image, point_loc, in zip(self._axes, self._titles, self._frames, self._loc):
            ax.set_title(title)
            if(ax is not None):
                ax.imshow(image, aspect="equal")
            if(point_loc is None):
                point_loc = (-5, -5)

            circle = plt.Circle(point_loc, 0.02 * self._frames.current.shape[0], clip_on=True, color=(0, 0, 1, 0.5),
                                picker=True)
            ax.add_artist(circle)
            circles.append(circle)

        self._figure.tight_layout()

        self._points = FrameInfo(*circles)

        # Grab the toolbar and tool manager, we will hack in our own button using it!
        self._toolmgr = self._figure.canvas.manager.toolmanager
        self._toolbar = self._figure.canvas.manager.toolbar

        self._mode = None

        def set_mode(mode):
            self._mode = mode

        class EditTool(ToolToggleBase):
            name = "Edit Point"
            description = "Edits a Point. Click to place and right click to delete..."
            default_keymap = "Ctrl+E"
            radio_group = "default"

            def enable(self, event):
                super().enable()
                set_mode("Edit")

            def disable(self, event):
                set_mode(None)

        self._toolmgr.add_tool("Edit Point", EditTool)
        self._toolbar.add_tool(self._toolmgr.get_tool("Edit Point"), "Edit Point")

        class SubmitBtn(ToolBase):
            name = "Submit"
            description = "Submit point data now!"

            def trigger(self, sender, event, data=None):
                plt.close(self._figure)

        self._toolmgr.add_tool("Submit", SubmitBtn)
        self._toolbar.add_tool(self._toolmgr.get_tool("Submit"), "Submit")

        def pick_evt(event):
            print(event.mouseevent.button)
            if (event.artist == self._points.current and
                    (event.mouseevent.button == MouseButton.RIGHT) and
                    (self._mode == "Edit")):
                self._points.current.set_visible(False)
                self._current_loc = None
                self._figure.canvas.draw_idle()

        self._figure.canvas.mpl_connect("pick_event", pick_evt)

        def release_evt(event):
            if (event.button == MouseButton.LEFT and event.inaxes == self._axes.current and self._mode == "Edit"):
                if (0 <= event.xdata <= self._frames.current.shape[1]):
                    if (0 <= event.ydata <= self._frames.current.shape[0]):
                        self._points.current.set_visible(True)
                        self._points.current.set_center((event.xdata, event.ydata))
                        self._current_loc = (event.xdata, event.ydata)
                        self._figure.canvas.draw_idle()

        self._figure.canvas.mpl_connect("button_release_event", release_evt)

    def show(self):
        plt.figure(self._figure.number)
        plt.show()

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
            "threshold_drop": self._on_low_prob
        }

        # The passed detection algorithm and it's arguments.
        self.DETECTION_ALGORITHM = settings["detection_algorithm"][0]
        self.DETECTION_ARGS = settings["detection_algorithm"][1:]

        if(self.DETECTION_ALGORITHM not in self.MODES):
            raise ValueError(f"'{self.DETECTION_ALGORITHM}' is not a known detection algorithm!")

        self.__bodyparts = bodyparts

        self.__video_path = Path(video_metadata["orig-video-path"])
        self.__frames = FrameInfo(None, None, None)
        self.__current_video_frame = -2


    def _on_prob_drop(self, args: List[str], poses: Pose) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the frames and
        """
        drop_value = float(args[0]) if(len(args) > 0 and (0 < float(args[0]) < 1)) else 0.5
        probs = np.array(super()._viterbi_probs)

        diff_forward = np.concatenate(np.zeros((1, probs.shape[1])), probs[1:] - probs[:-1], axis=1)
        diff_forward = np.nonzero(diff_forward >= drop_value)

        return diff_forward


    def _on_low_prob(self, args: List[str], poses: Pose) -> Tuple[np.ndarray, np.ndarray]:
        threshold = float(args[0]) if(len(args) > 0 and (0 < float(args[0]) < 1)) else 0.1
        probs = np.array(super()._viterbi_probs)

        return np.nonzero(probs < threshold)


    def _eat_frames(self, capture: cv2.VideoCapture, goto_frame: int) -> FrameInfo:
        print("Om nom num...")

        while(self.__current_video_frame < goto_frame):
            self.__frames.prev = self.__frames.current
            self.__frames.current = self.__frames.next

            if(capture.isOpen()):
                self.__frames.next = capture.read()
            else:
                self.__frames.next = None

            self.__current_video_frame += 1

        return self.__frames



    def on_end(self, progress_bar: tqdm) -> Optional[Pose]:
        poses = super()._backward(progress_bar)
        super()._current_frame = 0

        cap = cv2.VideoCapture(str(self.__video_path))

        print("Looking for poorly labeled frames: ")
        relabel_frames = np.transpose(self.MODES[self.DETECTION_ALGORITHM](self.DETECTION_ARGS, poses))
        relabel_frames = relabel_frames[np.argsort[relabel_frames[:, 0]]]

        print("Labeling frames: ")

        for frame, bp in tqdm(relabel_frames):
            frames = self._eat_frames(cap, frame)

            bp_name = self.__bodyparts[int(bp // self._num_outputs)]
            bp_number = (bp % self._num_outputs) + 1

            locations = (poses.get_x_at(frame, bp + off), poses.get_y_at(frame, bp + off) for off in [-1, 0, 1])

            point_editor = PointPicker(f"Frame {frame}, {bp_name} {bp_number}", self.__TITLES, frames, locations)
            point_editor.show()










    @classmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        return super(cls).get_settings() + [
            ("detection_algorithm", "A list of values, determines the detection algorithm to use and the arguments "
                                    "to pass to it. The available options are:\n"
                                    " - ['threshold', (optional float 0 < x < 1)]: Select frames that fall below a"
                                    "certain probability.\n"
                                    " - ['on_probability_drop', (optional float 0 < x < 1)]: Select frames on which "
                                    "the probability has dropped by the specified amount.",
             ["threshold", 0.5])
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
