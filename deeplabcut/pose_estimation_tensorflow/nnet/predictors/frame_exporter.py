"""
Package includes the frame exporter plugin. This plugin exports DeepLabCut probability maps to a binary format that can
be passed back into DeepLabCut again to perform frame predictions later. This allows for a video to be run through
the neural network (expensive) on a headless server or supercomputer, and then run a predictor with gui feedback on a
laptop or somewhere else. Below is the specification for the binary file format.

DEEPLABCUT FRAMESTORE BINARY FORMAT: (All fields are in little-endian format)
['DLCF'] -> DeepLabCut Framestore - 4 Bytes (file magic)

Header:
	['DLCH'] -> DeepLabCut Header
	[num_frames] - the number of frames. 8 Bytes (long unsigned integer)
	[num_bp] - number of bodyparts contained per frame. 4 Bytes (unsigned integer)
    [frame_height] - The height of a frame. 4 Bytes (unsigned integer)
	[frame_width] - The width of a frame. 4 Bytes (unsigned integer)
	[frame_rate] - The frame rate, in frames per second. 8 Bytes (double float).
	[stride] - The original video upscaling multiplier relative to current frame size. 4 Bytes (unsigned integer)

Bodypart Names:
    ['DBPN'] -> Deeplabcut Body Part Names
    (num_bp entries):
        [bp_len] - The length of the name of the bodypart. 2 Bytes (unsigned short)
        [DATA of length bp_len] - UTF8 Encoded name of the bodypart.

Frame data block:
	['FDAT'] -> Frame DATa
	Now the data (num_frames entries):
	0000000[sparce_fmt] - Single bit, last bit of the byte, determines if this is in the
	 						  sparce frame format. Sparce frames only store non-zero locations.
	Now Based on sparce_fmt flag:
		If it is false, frames are stored as 4 byte float arrays, row-by-row, as below:
			frame_bp-1,
			frame_bp-2,
			...,
			frame_bp-n
		Otherwise frames are stored in the format below
			Sparce Frame Format (num_bp entries):
				[num_entries] - Number of sparce entries in the frame, 4 bytes.
				[arr y] - list of 4 byte unsigned integers of length num_entries. Stores y coordinates of probabilities.
				[arr x] - list of 4 byte unsigned integers of length num_entries. Stores x coordinates of probabilities.
				[probs] - list of 4 byte floats, Stores probabilities specified at x and y coordinates above.
"""
from pathlib import Path
from typing import Union, List, Callable, Tuple, Any, Dict
import tqdm
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor, Pose, TrackingData
import numpy as np

# REQUIRED DATA TYPES: (With little endian encoding...)
luint16 = np.dtype(np.uint16).newbyteorder("<")
luint32 = np.dtype(np.uint32).newbyteorder("<")
luint64 = np.dtype(np.uint64).newbyteorder("<")
ldouble = np.dtype(np.float64).newbyteorder("<")
lfloat = np.dtype(np.float32).newbyteorder("<")


def to_bytes(obj: Any, dtype: np.dtype) -> bytes:
    return dtype.type(obj).to_bytes()


class FrameExporter(Predictor):
    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int,
                 settings: Union[Dict[str, Any], None], video_metadata: Dict[str, Any]):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)
        self._num_frames = num_frames
        self._bodyparts = bodyparts
        self._video_metadata = video_metadata
        self._num_outputs = num_outputs
        # Making the output file...
        orig_h5_path = Path(video_metadata["h5-file-name"])
        self._out_file = (orig_h5_path.parent / (orig_h5_path.stem + ".dlcf")).open("wb")

        self.SPARSIFY = settings["sparsify"]
        self.THRESHOLD = settings["threshold"]


    # noinspection PyTypeChecker
    def _write_header(self, scmap_width: int, scmap_height: int, stide: int):
        """
        Private, writes the header of the DeepLabCut Frame Store format...
        """
        # Write the file magic...
        self._out_file.write(b"DLCF")
        # Now we write the header label, then the header...
        self._out_file.write(b"DLCH")
        self._out_file.write(to_bytes(self._num_frames, luint64))  # The frame count
        self._out_file.write(to_bytes(len(self._bodyparts), luint32))  # The body part count
        self._out_file.write(to_bytes(scmap_height, luint32))  # The height of each frame
        self._out_file.write(to_bytes(scmap_width, luint32))  # The width of each frame
        self._out_file.write(to_bytes(self._video_metadata["fps"], ldouble))  # The frames per second
        self._out_file.write(to_bytes(stide, luint32))  # The video upscaling factor

    # noinspection PyTypeChecker
    def _write_body_part_names(self):
        """
        Private, writes the body part names chunk of the DeepLabCut Frame Store format.
        """
        self._out_file.write(b"DBPN")
        for bodypart in self._bodyparts:
            body_bytes = bodypart.encode("utf-8")
            self._out_file.write(to_bytes(len(body_bytes), luint16))
            self._out_file.write(body_bytes)


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        pass

    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        pass

    @staticmethod
    def get_name() -> str:
        return "file_exporter"

    @staticmethod
    def get_description() -> str:
        return ""

    @staticmethod
    def get_settings() -> Union[List[Tuple[str, str, Any]], None]:
        return [
            ("sparsify", "Boolean, specify whether optimize and store the data in a sparse format when it "
                         "saves storage", True),
            ("threshold", "The threshold used if sparsify is true. Any values which land below this threshold "
                          "probability won't be included in the frame.", 1e-6)
        ]

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True