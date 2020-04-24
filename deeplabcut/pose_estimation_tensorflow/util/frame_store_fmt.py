"""
Contains 2 Utility Classes for reading and writing the DeepLabCut Frame Store format. The format allows for processing
videos using DeepLabCut and then running predictions on the probability map data later. Below is a specification for
the DeepLabCut Frame Store format...

DEEPLABCUT FRAMESTORE BINARY FORMAT: (All multi-byte fields are in little-endian format)
['DLCF'] -> DeepLabCut Frame store - 4 Bytes (file magic)

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

            0000000[sparse_fmt] - Single bit, Whether we are using the sparse format. See difference in storage below:
            [data_length] - The length of the compressed/uncompressed frame data, 8 Bytes (long unsigned integer)

            DATA (The below is compressed in the zlib format and must be uncompressed first). Based on 'sparse_fmt' flag:

                If it is false, frames are stored as 4 byte float arrays, row-by-row, as below (x, y order below):
                    prob(1, 1), prob(2, 1), prob(3, 1), ....., prob(x, 1)
                    prob(1, 2), prob(2, 2), prob(3, 2), ....., prob(x, 2)
                    .....................................................
                    prob(1, y), prob(2, y), prob(3, y), ....., prob(x, y)
                Length of the above data will be frame height * frame width...
                Otherwise frames are stored in the format below.

                Sparce Frame Format (num_bp entries):
                    [num_entries] - Number of sparce entries in the frame, 8 bytes, unsigned integer.
                    [arr y] - list of 4 byte unsigned integers of length num_entries. Stores y coordinates of probabilities.
                    [arr x] - list of 4 byte unsigned integers of length num_entries. Stores x coordinates of probabilities.
                    [probs] - list of 4 byte floats, Stores probabilities specified at x and y coordinates above.
"""
from collections import namedtuple
from io import BytesIO
from pathlib import Path
from typing import Union, List, Callable, Tuple, Any, Dict, BinaryIO, Optional, cast
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor, Pose, TrackingData
import numpy as np
import zlib

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
    return dtype.type(obj).tobytes()


def from_bytes(data: bytes, dtype: np.dtype) -> np.dtype:
    """
    Converts bytes to a single object depending on the passed data type.

    :param data: The bytes to convert to an object
    :param dtype: The numpy data type to convert the bytes to.
    :return: An object of the specified data type passed to this method.
    """
    return np.frombuffer(data, dtype=dtype)[0]


class DLCFSConstants:
    """
    Class stores some constants for the DLC Frame Store format.
    """
    # The frame must become 1/3 or less its original size when sparsified to save space over the entire frame format,
    # so we check for this by dividing the original frame size by the sparse frame size and checking to see if it is
    # greater than this factor below...
    MIN_SPARSE_SAVING_FACTOR = 3
    # Magic...
    FILE_MAGIC = b"DLCF"
    # Chunk names...
    HEADER_CHUNK_MAGIC = b"DLCH"
    # The header length, including the 'DLCH' magic
    HEADER_LENGTH = 36
    BP_NAME_CHUNK_MAGIC = b"DBPN"
    FRAME_DATA_CHUNK_MAGIC = b"FDAT"


def string_list(lister: list):
    """
    Casts object to a list of strings, enforcing type...

    :param lister: The list
    :return: A list of strings

    :raises: ValueError if the list doesn't contain strings...
    """
    lister = list(lister)

    for item in lister:
        if(not isinstance(item, str)):
            raise ValueError("Must be a list of strings!")

    return lister


class DLCFSHeader():
    """
    Stores some basic info about a frame store...

    Below are the fields in order, their names, types and default values:
        ("number_of_frames", int, 0),
        ("frame_height", int, 0),
        ("frame_width", int, 0),
        ("frame_rate", float, 0),
        ("stride", int, 0),
        ("bodypart_names", list of strings, [])
    """
    SUPPORTED_FIELDS = [
        ("number_of_frames", int, 0),
        ("frame_height", int, 0),
        ("frame_width", int, 0),
        ("frame_rate", float, 0),
        ("stride", int, 0),
        ("bodypart_names", string_list, [])
    ]

    GET_VAR_CAST = {name: var_cast for name, var_cast, __ in SUPPORTED_FIELDS}

    def __init__(self, *args, **kwargs):
        """
        Make a new DLCFrameStoreHeader. Supports tuple style construction and also supports setting the fields using
        keyword arguments. Look at the class documentation for all the fields.
        """
        # Make the fields.
        self._values = {}
        for name, var_caster, def_value in self.SUPPORTED_FIELDS:
            self._values[name] = def_value

        for new_val, (key, var_caster, __) in zip(args, self.SUPPORTED_FIELDS):
            self._values[key] = var_caster(new_val)

        for key, new_val in kwargs.items():
            if(key in self._values):
                self._values[key] = new_val

    def __getattr__(self, item):
        if(item == "_values"):
            return self.__dict__[item]
        return self._values[item]

    def __setattr__(self, key, value):
        if(key == "_values"):
            self.__dict__["_values"] = value
            return
        self.__dict__["_values"][key] = self.GET_VAR_CAST[key](value)

    def __str__(self):
        return str(self._values)

    def to_list(self) -> List[Any]:
        return [self._values[key] for key, __, __ in self.SUPPORTED_FIELDS]


class DLCFSReader():
    """
    A DeepLabCut Frame Store Reader. Allows for reading ".dlcf" files.
    """

    HEADER_DATA_TYPES = [luint64, luint32, luint32, luint32, ldouble, luint32]
    HEADER_OFFSETS = np.cumsum([4] + [dtype.itemsize for dtype in HEADER_DATA_TYPES])[:-1]

    def _assert_true(self, assertion: bool, error_msg: str):
        """
        Private method, if the assertion is false, throws a ValueError.
        """
        if(not assertion):
            raise ValueError(error_msg)

    def __init__(self, file: BinaryIO):
        """
        Create a new DeepLabCut Frame Store Reader.

        :param file: The binary file object to read a frame store from, file opened with 'rb'.
        """
        self._assert_true(file.read(4) == DLCFSConstants.FILE_MAGIC,
                          "File is not of the DLC Frame Store Format!")
        # Check for valid header...
        header_bytes = file.read(DLCFSConstants.HEADER_LENGTH)
        self._assert_true(header_bytes[0:4] == DLCFSConstants.HEADER_CHUNK_MAGIC,
                          "First Chunk must be the Header ('DLCH')!")
        # Read the header into a DLC header...
        parsed_data = [from_bytes(header_bytes[off:(off + dtype.itemsize)], dtype) for off, dtype in
                       zip(self.HEADER_OFFSETS, self.HEADER_DATA_TYPES)]
        self._header = DLCFSHeader(parsed_data[0], *parsed_data[2:])
        body_parts = [None] * parsed_data[1]
        # Read the body part chunk...
        self._assert_true(file.read(4) == DLCFSConstants.BP_NAME_CHUNK_MAGIC, "Body part chunk must come second!")
        for i in range(len(body_parts)):
            length = from_bytes(file.read(2), luint16)
            body_parts[i] = file.read(length).decode("utf-8")
        # Add the list of body parts to the header...
        self._header.bodypart_names = body_parts

    def get_header(self) -> DLCFSHeader:
        return DLCFSHeader(*self._header.to_list())

    def read_frames(self, num_frames: int) -> TrackingData:
        # TODO: Write
        pass


class DLCFSWriter():
    """
    A DeepLabCut Frame Store Writer. Allows for writing ".dlcf" files.
    """
    def __init__(self, file: BinaryIO, header: DLCFSHeader, threshold: Optional[float] = 1e6,
                 compression_level: int = 6):
        """
        Create a new DeepLabCut Frame Store Writer.

        :param file: The file to write to, a file opened in 'wb' mode.
        :param header: The DLCFrameStoreHeader, with all properties filled out.
        :param threshold: A float between 0 and 1, the threshold at which to filter out any probabilities which fall
                          below it. The default value is 1e6, and it can be set to None to force all frames to be
                          stored in the non-sparse format.
        :param compression_level: The compression of the data. 0 is no compression, 9 is max compression but is slow.
                                  The default is 6.
        """
        self._out_file = file
        self._header = header
        self._threshold = threshold if (threshold is None or 0 <= threshold <= 1) else 1e6
        self._compression_level = compression_level if(0 <= compression_level <= 9) else 6
        self._current_frame = -1

        # Write the file magic...
        self._out_file.write(DLCFSConstants.FILE_MAGIC)
        # Now we write the header:
        self._out_file.write(DLCFSConstants.HEADER_CHUNK_MAGIC)
        self._out_file.write(to_bytes(header.number_of_frames, luint64))  # The frame count
        self._out_file.write(to_bytes(len(header.bodypart_names), luint32))  # The body part count
        self._out_file.write(to_bytes(header.frame_height, luint32))  # The height of each frame
        self._out_file.write(to_bytes(header.frame_width, luint32))  # The width of each frame
        self._out_file.write(to_bytes(header.frame_rate, ldouble))  # The frames per second
        self._out_file.write(to_bytes(header.stride, luint32))  # The video upscaling factor

        # Now we write the body part name chunk:
        self._out_file.write(DLCFSConstants.BP_NAME_CHUNK_MAGIC)
        for bodypart in header.bodypart_names:
            body_bytes = bodypart.encode("utf-8")
            self._out_file.write(to_bytes(len(body_bytes), luint16))
            self._out_file.write(body_bytes)

        # Finish by writing the begining of the frame data chunk:
        self._out_file.write(DLCFSConstants.FRAME_DATA_CHUNK_MAGIC)


    def write_data(self, data: TrackingData):
        """
        Write the following frames to the file.

        :param data: A TrackingData object, which contains frame data
        """
        # Some checks to make sure tracking data parameters match those set in the header:
        self._current_frame += data.get_frame_count()
        if(self._current_frame >= self._header.number_of_frames):
            raise ValueError(f"Data Overflow! '{self._header.number_of_frames}' frames expected, tried to write "
                             f"'{self._current_frame + 1}' frames.")

        if(data.get_bodypart_count() != len(self._header.bodypart_names)):
            raise ValueError(f"'{data.get_bodypart_count()}' body parts does not match the "
                             f"'{len(self._header.bodypart_names)}' body parts specified in the header.")

        for frm_idx in range(data.get_frame_count()):
            for bp in range(data.get_bodypart_count()):
                frame = data.get_prob_table(frm_idx, bp)
                if(self._threshold is not None):
                    # Sparsify the data by removing everything below the threshold...
                    sparse_y, sparse_x = np.nonzero(frame > self._threshold)
                    probs = frame[(sparse_y, sparse_x)]
                    # Check if we managed to strip out at least 2/3rds of the data, and if so write the frame using the
                    # sparse format. Otherwise it is actually more memory efficient to just store the entire frame...
                    if(len(frame.flat) >= (len(sparse_y) * DLCFSConstants.MIN_SPARSE_SAVING_FACTOR)):
                        self._out_file.write(to_bytes(1, luint8))  # Sparse indicator flag
                        # COMPRESSED DATA:
                        buffer = BytesIO()
                        buffer.write(to_bytes(len(sparse_y), luint64))  # The length of the sparse data entries.
                        buffer.write(sparse_y.astype(luint32).tobytes('C'))  # Y coord data
                        buffer.write(sparse_x.astype(luint32).tobytes('C'))  # X coord data
                        buffer.write(probs.astype(lfloat).tobytes('C'))  # Probabilities
                        # Compress the sparse data and write it's length, followed by itself....
                        comp_data = zlib.compress(buffer.getvalue(), self._compression_level)
                        self._out_file.write(to_bytes(len(comp_data), luint64))
                        self._out_file.write(comp_data)

                        continue
                # If sparse optimization mode is off or the sparse format wasted more space, just write the entire
                # frame...
                self._out_file.write(to_bytes(0, luint8))

                comp_data = zlib.compress(frame.astype(lfloat).tobytes('C'))
                self._out_file.write(to_bytes(len(comp_data), luint64))
                self._out_file.write(comp_data)

