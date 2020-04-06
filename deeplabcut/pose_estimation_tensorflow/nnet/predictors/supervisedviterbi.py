# For types in methods
from typing import Union, List, Tuple, Any, Dict, Callable, Optional
from numpy import ndarray
import tqdm

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.fastviterbi import FastViterbi
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose

# For computations
import numpy as np
from collections import deque

class SupervisedViterbi(FastViterbi):
    """
    A predictor that applies the viterbi algorithm to frames in order to predict poses.
    The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but
    is also more memory intensive and computationally expensive. This specific implementation
    asks for user feedback on frames it is unsure and then reruns the algorithm for better
    results.
    """
    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int,
                 settings: Union[Dict[str, Any], None], video_metadata: Dict[str, Any]):
        self.MODES = {
            "on_probability_drop": self._on_prob_drop
        }

        # The passed detection algorithm and it's arguments.
        self.DETECTION_ALGORITHM = settings["detection_algorithm"][0]
        self.DETECTION_ARGS = settings["detection_algorithm"][1:]

        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)


    def _on_prob_drop(self, args: List[str]):
        """
        Returns the frames and
        """
        drop_value = float(args[0]) if(len(args) > 0) else 0.5
        probs = np.array(super()._viterbi_probs)

        diff_forward = np.concatenate(np.zeros((1, probs.shape[1])), probs[1:] - probs[:-1], axis=1)
        diff_forward = np.nonzero(diff_forward >= drop_value)

        return diff_forward




    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        poses = super()._backward(progress_bar)
        super()._current_frame = 0

        print("Looking for poorly labeled frames:")
        poses.get_all_prob()


    @classmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        return super(cls).get_settings() + [
            ("detection_algorithm", "A list of values, determines the detection algorithm to use and the arguments "
                                    "to pass to it. The available options are:\n"
                                    " - ['on_probability_drop', (optional float 0 < x < 1)]: Select frames on which the "
                                    "probability has dropped by the specified amount.", ["on_probability_drop", 0.5])
        ]

    @staticmethod
    def get_name() -> str:
        return "supervised_viterbi"

    @staticmethod
    def get_description() -> str:
        return ("A predictor that applies the viterbi algorithm to frames in order to predict poses. "
                "The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but "
                "is also more memory intensive and computationally expensive. This specific implementation "
                "asks for user feedback on frames it is unsure and then reruns the algorithm for better "
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
