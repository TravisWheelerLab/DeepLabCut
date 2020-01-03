
# For types in methods
from typing import Union, List, Tuple, Any, Callable, Dict

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose


class SingleArgMaxPredict(Predictor):
    """
    Default processor for DeepLabCut, and the code originally used by DeepLabCut for prediction of points. Predicts
    the point from the probability frames simply by selecting the max probability in the frame.
    """
    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int, settings: None, video_metadata: Dict[str, Any]):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)
        self._num_outputs = num_outputs


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # Using new object library to get the max... Drastically simplified logic...
        return scmap.get_poses_for(scmap.get_max_scmap_points(num_max=self._num_outputs))

    def on_end(self, pbar) -> Union[None, Pose]:
        # Processing is done per frame, so return None.
        return None

    @staticmethod
    def get_settings() -> Union[List[Tuple[str, str, Any]], None]:
        return None

    @staticmethod
    def get_name() -> str:
        return "argmax"

    @staticmethod
    def get_description() -> str:
        return ("Default processor for DeepLabCut, and was the code originally used by DeepLabCut"
                "historically. Predicts the point from the probability frames simply by selecting"
                "the max probabilities in the frame.")

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True