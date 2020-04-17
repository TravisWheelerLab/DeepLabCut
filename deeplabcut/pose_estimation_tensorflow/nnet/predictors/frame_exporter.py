"""
Package includes the frame exporter plugin. This plugin exports DeepLabCut probability maps to a binary format that can
be passed back into DeepLabCut again to perform frame predictions later. This allows for a video to be run through
the neural network (expensive) on a headless server or supercomputer, and then run through a predictor with gui
feedback on a laptop or somewhere else. Below is the specification for the binary file format.

DEEPLABCUT FRAMESTORE BINARY FORMAT: (All multi-byte fields are in little-endian format)
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
	    Each sub-frame entry (num_bp entries):
            0000000[sparce_fmt] - Single bit, last bit of the byte, determines if this is in the
                                      sparce frame format. Sparce frames only store non-zero locations.
            Now Based on sparce_fmt flag:
                If it is false, frames are stored as 4 byte float arrays, row-by-row, as below (x, y order below):
                    prob(1, 1), prob(2, 1), prob(3, 1), ....., prob(x, 1)
                    prob(1, 2), prob(2, 2), prob(3, 2), ....., prob(x, 2)
                    .....................................................
                    prob(1, y), prob(2, y), prob(3, y), ....., prob(x, y)
                Otherwise frames are stored in the format below
                    Sparce Frame Format (num_bp entries):
                        [num_entries] - Number of sparce entries in the frame, 8 bytes, unsigned integer.
                        [arr y] - list of 4 byte unsigned integers of length num_entries. Stores y coordinates of probabilities.
                        [arr x] - list of 4 byte unsigned integers of length num_entries. Stores x coordinates of probabilities.
                        [probs] - list of 4 byte floats, Stores probabilities specified at x and y coordinates above.
"""
from pathlib import Path
from typing import Union, List, Callable, Tuple, Any, Dict, BinaryIO
import tqdm
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor, Pose, TrackingData
import numpy as np

# REQUIRED DATA TYPES: (With little endian encoding...)
luint8 = np.dtype(np.uint8).newbyteorder("<")
luint16 = np.dtype(np.uint16).newbyteorder("<")
luint32 = np.dtype(np.uint32).newbyteorder("<")
luint64 = np.dtype(np.uint64).newbyteorder("<")
ldouble = np.dtype(np.float64).newbyteorder("<")
lfloat = np.dtype(np.float32).newbyteorder("<")


def to_bytes(obj: Any, dtype: np.dtype) -> bytes:
    """
    Converts an object to bytes.

    :param obj: The object to convert to bytes.
    :param dtype: The numpy data type to interpret the object as when converting to bytes.
    :return: A bytes object, representing the object obj as type dtype.
    """
    return dtype.type(obj).to_bytes()


class FrameExporter(Predictor):
    """
    Exports DeepLabCut probability maps to a binary format that can be passed back into DeepLabCut again to perform
    frame predictions later. This allows for a video to be run through the neural network (expensive) on a headless
    server or supercomputer, and then run through a predictor with gui feedback on a laptop or somewhere else.
    """

    # The frame must become 1/3 or less its original size when sparsified to save space over the entire frame format,
    # so we check for this by dividing the original frame size by the sparse frame size and checking to see if it is
    # greater than this factor below...
    MIN_SPARSE_SAVING_FACTOR = 3
    # Magic...
    FILE_MAGIC = b"DLCF"
    # Chunk names...
    HEADER_CHUNK_MAGIC = b"DLCH"
    BP_NAME_CHUNK_MAGIC = b"DBPN"
    FRAME_DATA_CHUNK_MAGIC = b"FDAT"

    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int,
                 settings: Union[Dict[str, Any], None], video_metadata: Dict[str, Any]):
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)
        self._num_frames = num_frames
        self._bodyparts = bodyparts
        self._video_metadata = video_metadata
        self._num_outputs = num_outputs
        # Making the output file...
        orig_h5_path = Path(video_metadata["h5-file-name"])
        self._out_file: BinaryIO = (orig_h5_path.parent / (orig_h5_path.stem + ".dlcf")).open("wb")
        # Load in the settings....
        self.SPARSIFY = settings["sparsify"]
        self.THRESHOLD = settings["threshold"]
        # Initialize the frame counter...
        self._current_frame = 0


    # noinspection PyTypeChecker
    def _write_header(self, scmap_width: int, scmap_height: int, stide: int):
        """
        Private, writes the header of the DeepLabCut Frame Store format...
        """
        # Write the file magic...
        self._out_file.write(self.FILE_MAGIC)
        # Now we write the header label, then the header...
        self._out_file.write(self.HEADER_CHUNK_MAGIC)
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
        self._out_file.write(self.BP_NAME_CHUNK_MAGIC)
        for bodypart in self._bodyparts:
            body_bytes = bodypart.encode("utf-8")
            self._out_file.write(to_bytes(len(body_bytes), luint16))
            self._out_file.write(body_bytes)

    # noinspection PyTypeChecker
    def _write_prob_map(self, frame: np.ndarray):
        if(self.SPARSIFY):
            # Sparsify the data by removing everything below the threshold...
            sparse_y, sparse_x = np.nonzero(frame > self.THRESHOLD)
            probs = frame[(sparse_y, sparse_x)]
            # Check if we managed to strip out at least 2/3rds of the data, and if so write the frame using the sparse
            # format. Otherwise it is actually more memory efficient to just store the entire frame...
            if((len(frame) / len(sparse_y)) >= self.MIN_SPARSE_SAVING_FACTOR):
                self._out_file.write(to_bytes(1, luint8))  # Sparse indicator flag
                self._out_file.write(sparse_y.astype(luint32).tobytes('C'))  # Y coord data
                self._out_file.write(sparse_x.astype(luint32).tobytes('C'))  # X coord data
                self._out_file.write(probs.astype(lfloat).tobytes('C'))  # Probabilities

                return
        # If sparse optimization mode is off or the sparse format wasted more space, just write the entire frame...
        self._out_file.write(to_bytes(0, luint8))
        self._out_file.write(frame.astype(lfloat).tobytes('C'))


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        # If we are just starting, write the header, body part names chunk, and magic for frame data chunk...
        if(self._current_frame == 0):
            self._write_header(scmap.get_frame_width(), scmap.get_frame_height(), scmap.get_down_scaling())
            self._write_body_part_names()
            self._out_file.write(self.FRAME_DATA_CHUNK_MAGIC)

        # Writing all of the frames in this batch...
        for i in range(scmap.get_frame_count()):
            for j in range(scmap.get_bodypart_count()):
                self._write_prob_map(scmap.get_prob_table(i, j))

        return scmap.get_poses_for(scmap.get_max_scmap_points(num_max=self._num_outputs))


    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        self._out_file.flush()
        self._out_file.close()
        return None

    @staticmethod
    def get_name() -> str:
        return "file_exporter"

    @staticmethod
    def get_settings() -> Union[List[Tuple[str, str, Any]], None]:
        return [
            ("sparsify", "Boolean, specify whether optimize and store the data in a sparse format when it "
                         "saves storage", True),
            ("threshold", "A Float between 0 and 1. The threshold used if sparsify is true. Any values which land below "
                          "this threshold probability won't be included in the frame.", 1e-6)
        ]

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return None

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True