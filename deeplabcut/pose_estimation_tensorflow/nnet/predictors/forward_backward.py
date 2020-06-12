# For types in methods
from typing import Union, List, Tuple, Any, Dict, Callable, Optional, Sequence
from numpy import ndarray
import tqdm

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose

# For computations
import numpy as np
from collections import deque

# TODO: Add more test methods, disable numpy warnings....
# TODO: Add Concept of being "In the Ground..."
# TODO: Forward Pass, Backward Pass, Multiply the two to produce postieror probs.
# Represents an valid numpy indexing type.
Indexer = Union[slice, int, List[int], Tuple[int], None]

class SparseTrackingData:
    """
    Represents sparse tracking data. Includes probabilities, offsets, and x/y coordinates in the probability map.
    """

    class NumpyAccessor:
        """
        A numpy accessor class, allows for controlled access to a numpy array (Disables direct assignment).
        """
        def __init__(self, src_arr: ndarray, pre_index: Tuple[Indexer] = None):
            self._src_arr = src_arr
            self._pre_idx = pre_index if(pre_index is not None) else tuple()

        def __getitem__(self, *index: Indexer):
            if (self._pre_idx is None):
                return self._src_arr[index]
            return self._src_arr[self._pre_idx + index]

        def __setitem__(self, *index: Indexer, value: ndarray):
            self._src_arr[self._pre_idx + index] = value

    def __init__(self):
        """
        Makes a new tracking data with all empty fields.
        """
        self._coords = None
        self._offsets_probs = None
        self._coords_access = None
        self._off_access = None
        self._prob_access = None

    @property
    def coords(self):
        """
        The coordinates of this SparceTrackingData in [y, x] order.
        """
        return self._coords_access

    @property
    def probs(self):
        """
        The probabilities of this SparceTrackingData.
        """
        return self._prob_access

    @property
    def offsets(self):
        """
        The offsets of this SparceTrackingData.
        """
        return self._off_access

    def _set_data(self, coords: ndarray, offsets: ndarray, probs: ndarray):
        # If no coordinates are passed, set everything to None as there is no data.
        if(len(coords) == 0):
            self._coords = None
            self._coords_access = None
            self._offsets_probs = None
            self._off_access = None
            self._prob_access = None
            return
        # Combine offsets and probabilities into one numpy array...
        offsets_plus_probs = np.concatenate((np.transpose(np.expand_dims(probs, axis=0)), offsets), axis=1)

        if((len(offsets_plus_probs.shape) != 2) and offsets_plus_probs.shape[1] == 3):
            raise ValueError("Offset/Probability array must be n by 3...")

        if(len(coords.shape) != 2 or coords.shape[0] != offsets_plus_probs.shape[0] or coords.shape[1] != 2):
            raise ValueError("Coordinate array must be n by 2...")

        # Set the arrays to the data an make new numpy accessors which point to the new numpy arrays
        self._coords = coords
        self._coords_access = self.NumpyAccessor(self._coords)
        self._offsets_probs = offsets_plus_probs
        self._off_access = self.NumpyAccessor(self._offsets_probs, (slice(1, 3),))
        self._prob_access = self.NumpyAccessor(self._offsets_probs, (0,))

    def pack(self, y_coords: Optional[ndarray], x_coords: Optional[ndarray], probs: Optional[ndarray],
             x_offset: Optional[ndarray], y_offset: Optional[ndarray]):
        """
        Pack the passed data into this SparceTrackingData object.

        :param y_coords: Y coordinate value of each probability within the probability map.
        :param x_coords: X coordinate value of each probability within the probability map.
        :param probs: The probabilities of the probability frame.
        :param x_offset: The x offset coordinate values within the original video.
        :param y_offset: The y offset coordinate values within the original video.
        """
        self._set_data(np.transpose((y_coords, x_coords)), np.transpose((x_offset, y_offset)), probs)

    # noinspection PyTypeChecker
    def unpack(self) -> Tuple[Optional[ndarray], Optional[ndarray], Optional[ndarray], Optional[ndarray], Optional[ndarray]]:
        """
        Return all fields of this SparseTrackingData

        :return: A tuple of 1 dimensional numpy arrays, being: (y_coord, x_coord, probs, x_offset, y_offset)
        """
        if(self._coords is None):
            return (None, None, None, None, None)

        return tuple(np.transpose(self._coords)) + tuple(np.transpose(self._offsets_probs))

    def duplicate(self) -> "SparseTrackingData":
        """
        Creates a copy this SparseTrackingData, returning it.

        :return: A new SparseTrackingData, which is identical to the current SparseTrackingData.
        """
        new_sparse_data = SparseTrackingData()
        new_sparse_data.pack(*(np.copy(arr) for arr in self.unpack()))
        return new_sparse_data


    def desparsify(self, orig_width: int, orig_height: int, orig_stride: int) -> Optional[TrackingData]:
        """
        Desparsifies this SparseTrackingData object.

        :param orig_width: The original width of the tracking data object, will be the frame width of the newly
                           constructed TrackingData object.
        :param orig_height: The original height of the tracking data object, will be the frame height of the newly
                           constructed TrackingData object.
        :param orig_stride: The stride of the newly constructed tracking object (upscaling factor to scale to the size
                            of the original video).

        :return: A new TrackingData object, with data matching the original SparseTrackingData object. Returns None
                 if this object has no data stored yet...
        """
        y, x, probs, x_off, y_off = self.unpack()

        if(y is None):
            return None

        new_td = TrackingData.empty_tracking_data(1, 1, orig_width, orig_height, orig_stride)
        new_td.set_offset_map(np.zeros((1, orig_height, orig_width, 1, 2), dtype=np.float32))

        new_td.get_source_map()[0, y, x, 0] = probs
        new_td.get_offset_map()[0, y, x, 0, 0] = x_off
        new_td.get_offset_map()[0, y, x, 0, 1] = y_off

        return new_td


    @classmethod
    def sparsify(cls, track_data: TrackingData, frame: int, bodypart: int, threshold: float) -> "SparseTrackingData":
        """
        Sparcify the TrackingData.

        :param track_data: The TrackingData to sparcify
        :param frame: The frame of the TrackingData to sparcify
        :param bodypart: The bodypart of the TrackingData to sparcify
        :param threshold: The threshold to use when scarifying. All values below the threshold are removed.
        :return: A new SparceTrackingData object containing the data of the TrackingData object.
        """
        new_sparse_data = cls()

        y, x = np.nonzero(track_data.get_prob_table(frame, bodypart) > threshold)

        if(track_data.get_offset_map() is None):
            x_off, y_off = np.zeros(x.shape, dtype=np.float32), np.zeros(y.shape, dtype=np.float32)
        else:
            x_off, y_off = np.transpose(track_data.get_offset_map()[frame, y, x, bodypart])

        probs = track_data.get_prob_table(frame, bodypart)[y, x]

        new_sparse_data.pack(y, x, probs, x_off, y_off)

        return new_sparse_data


class ForwardBackward(Predictor):
    """
    A predictor that applies the Forward Backward algorithm to frames in order to predict poses.
    The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but
    is also more memory intensive and computationally expensive. This specific implementation
    uses sparse matrix multiplication for massive speedup over a normal
    Forward Backward implementation...)
    """
    # The amount of side block increase for the normal distribution to increase by 1...
    ND_UNIT_PER_SIDE_COUNT = 10

    def __init__(self, bodyparts: Union[List[str]], num_outputs: int, num_frames: int,
                 settings: Union[Dict[str, Any], None], video_metadata: Optional[Dict[str, Any]]):
        """ Initialized a ForwardBackward plugin for analyzing a video """
        super().__init__(bodyparts, num_outputs, num_frames, settings, video_metadata)

        # Store bodyparts and num_frames for later use, they become useful
        self._num_frames = num_frames
        # Indexing if there are multiple outputs per body part
        self._num_outputs = num_outputs
        self._total_bp_count = len(bodyparts) * num_outputs

        # Holds the original TrackingData in a sparse format using the SparseTrackingData specified above.
        self._sparse_data: List[List[Optional[SparseTrackingData]]] = [None] * num_frames
        # Will hold the viterbi frames on the forward compute...
        # Dimension are: Frame -> Bodypart -> (Forward/Backward probability).
        self._frame_probs: List[List[Optional[ndarray]]] = [None] * num_frames

        # Stashes edge to edge probabilities, in the form delta-edge-index -> gaussian...
        self._edge_edge_table: ndarray = None
        # Stores actual edge values
        self._edge_vals: ndarray = None
        self._edge_coords: ndarray = None

        # Represents precomputed gaussian values... Stored in the form (delta_y, delta_x) -> gaussian value.
        self._gaussian_table: ndarray = None
        # Stores stride used in this video...
        self._down_scaling = None
        # Keeps track of current frame...
        self._current_frame = 0

        # Precomputed table for computing negative impacts of body parts
        self._neg_gaussian_table = None

        # Values for the gaussian formula, can be adjusted in dlc_config for differing results...
        self.NORM_DIST_UNSCALED = settings["norm_dist"]  # The normal distribution
        self.NORM_DIST = None # To be computed below...
        self.AMPLITUDE = settings["amplitude"]  # The amplitude, or height of the gaussian curve
        self.LOWEST_VAL = settings["lowest_gaussian_value"]  # Changes the lowest value that the gaussian curve can produce
        self.THRESHOLD = settings["threshold"]  # The threshold for the matrix... Everything below this value is ignored.
        self.EDGE_PROB = settings["edge_probability"]  # DLC "Predicted" going off screen probability.
        self.BLOCKS_PER_EDGE = settings["edge_blocks_per_side"]  # Number of blocks to allocate per edge....
        self._edge_block_value = self.EDGE_PROB / (self.BLOCKS_PER_EDGE * 4)  # Probability value for every block...

        # More global variables, can also be set in dlc_config...
        self.NEGATE_ON = bool(settings["negate_overlapping_predictions"])  # Enables prior body part negation...
        self.NEG_NORM_DIST_UNSCALED = settings["negative_impact_distance"]  # Normal distribution of negative 2D gaussian curve
        self.NEG_NORM_DIST = None  # To be computed below....
        self.NEG_AMPLITUDE = settings["negative_impact_factor"]  # Negative amplitude to use for 2D negation gaussian
        self.NEG_CURVE_TYPE = settings["negation_curve"] == "Gaussian"
        

    def _gaussian_formula(self, prior_x: float, x: float, prior_y: float, y: float) -> float:
        """
        Private method, computes location of point (x, y) on a gaussian curve given a prior point (x_prior, y_prior)
        to use as the center.

        :param prior_x: The x point in the prior frame
        :param x: The current frame's x point
        :param prior_y: The y point in the prior frame
        :param y: The current frame's y point
        :return: The location of x, y given gaussian curve centered at x_prior, y_prior
        """
        # Formula for 2D gaussian curve (or bump)
        inner_x_delta = ((prior_x - x) ** 2) / (2 * self.NORM_DIST ** 2)
        inner_y_delta = ((prior_y - y) ** 2) / (2 * self.NORM_DIST ** 2)
        return self.AMPLITUDE * np.exp(-(inner_x_delta + inner_y_delta)) + self.LOWEST_VAL


    def _neg_gaussian_formula(self, prior_x: float, x: float, prior_y: float, y: float) -> float:
        """
        Private method, computes location of point (x, y) on a inverted gaussian (or quartic) curve given a prior point
        (x_prior, y_prior) to use as the center.

        :param prior_x: The x point in the prior frame
        :param x: The current frame's x point
        :param prior_y: The y point in the prior frame
        :param y: The current frame's y point
        :return: The location of x, y given inverted gaussian curve (1 - curve) centered at x_prior, y_prior
        """
        if(self.NEG_CURVE_TYPE):
            # Formula for 2D inverted gaussian curve (or dip)
            inner_x_delta = ((prior_x - x) ** 2) / (2 * self.NEG_NORM_DIST ** 2)
            inner_y_delta = ((prior_y - y) ** 2) / (2 * self.NEG_NORM_DIST ** 2)
            return 1 - (self.NEG_AMPLITUDE * np.exp(-(inner_x_delta + inner_y_delta)))
        else:
            # Formula for quartic negation curve, gives much steeper sides and longer sustained peek...
            quad_function = max(0.0, (0.5 - ((x / self.NEG_NORM_DIST) ** 4)) + (0.5 - ((y / self.NEG_NORM_DIST) ** 4)))
            return 1 - (self.NEG_AMPLITUDE * quad_function)


    def _compute_gaussian_table(self, width: int, height: int) -> None:
        """
        Compute the gaussian table given the width and height of each probability frame. Results are stored in
        self._gaussian_table
        """
        # Compute the normal distribution based on how many blocks per frame there are...
        self.NORM_DIST = (np.sqrt(width * height) / self.ND_UNIT_PER_SIDE_COUNT) * self.NORM_DIST_UNSCALED
        # Allocate gaussian table of width x height...
        self._gaussian_table = np.zeros((height + 2, width + 2), dtype="float32")

        # Iterate through filling in the values, using (0, 0) as the prior coordinate
        for y in range(height + 2):
            for x in range(width + 2):
                self._gaussian_table[y, x] = self._gaussian_formula(0, x, 0, y)


    def _compute_neg_gaussian_table(self, width: int, height: int) -> None:
        """
        Computes the precomputed inverted 2D gaussian curve used for providing negative impacts at prior predicted
        bodyparts. Stored in self._neg_gaussian_table.
        """
        # Scale normal distribution based on size of frames...
        self.NEG_NORM_DIST = (np.sqrt(width * height) / self.ND_UNIT_PER_SIDE_COUNT) * self.NEG_NORM_DIST_UNSCALED
        # Allocate...
        self._neg_gaussian_table = np.zeros((height, width), dtype="float32")

        # Iterate and fill values
        for y in range(height):
            for x in range(width):
                self._neg_gaussian_table[y, x] = self._neg_gaussian_formula(0, x, 0, y)


    def _compute_edge_coordinates(self, width: int, height: int, num_of_blocks: int):
        """ Computes centered coordinates for edge blocks, which are used for gaussian computations... """
        self._edge_coords = np.zeros((2, num_of_blocks * 4), dtype=int)

        # Used to get the midpoints of the blocks on a given side.....
        def get_midpoints(num_blocks, side_length):
            #      __________Block's starting point in list__________ + _____midway of a subsection_______
            return (np.arange(num_blocks) / num_blocks) * side_length + ((side_length / num_blocks) * 0.5)

        # Side 1 (Left...)
        self._edge_coords[0, 0:num_of_blocks] = -1 # X values
        self._edge_coords[1, 0:num_of_blocks] = get_midpoints(num_of_blocks, height) # Y Values

        # Side 2 (Bottom...)
        self._edge_coords[0, num_of_blocks:num_of_blocks * 2] = get_midpoints(num_of_blocks, width) # X values
        self._edge_coords[1, num_of_blocks:num_of_blocks * 2] = height # Y Values

        # Side 3 (Right...)
        self._edge_coords[0, num_of_blocks * 2:num_of_blocks * 3] = width # X values
        self._edge_coords[1, num_of_blocks * 2:num_of_blocks * 3] = get_midpoints(num_of_blocks, height) # Y Values

        # Side 4 (Top...)
        self._edge_coords[0, num_of_blocks * 3:num_of_blocks * 4] = get_midpoints(num_of_blocks, width) # X values
        self._edge_coords[1, num_of_blocks * 3:num_of_blocks * 4] = -1 # Y Values

        # Transpose so it's edge index -> x, y coordinate...
        self._edge_coords = self._edge_coords.transpose()

        # Done, return...
        return


    def _gaussian_values_at(self, current_xs: ndarray, current_ys: ndarray, prior_xs: ndarray, prior_ys: ndarray) -> ndarray:
        """
        Get the gaussian values for a collection of points provided as arrays....

        :param current_xs: The array storing the current frame's x values
        :param current_ys: The array storing the current frame's y values
        :param prior_xs: The array storing the prior frame's x values
        :param prior_ys: The array storing the prior frame's y values
        :return: A 2D array representing all gaussian combinations that need to be computed for this frame...
        """
        # Compute delta x and y values. Use broadcasting to compute this for all current frames.
        # We take the absolute value since differences can be negative and gaussian mirrors over the axis...
        delta_x = np.abs(np.expand_dims(current_xs, axis=1) - np.expand_dims(prior_xs, axis=0))
        delta_y = np.abs(np.expand_dims(current_ys, axis=1) - np.expand_dims(prior_ys, axis=0))

        # Return the current_frame by prior_frame gaussian array by just plugging in y and x indexes...
        # This is the easiest way I have found to do it but I am sure there is a better way...
        return self._gaussian_table[delta_y.flatten(), delta_x.flatten()].reshape(delta_y.shape)


    def _init_data_structures(self, scmap: TrackingData):
        """ Initializes all data structures and computational tables needed by the algorithm """
        # Pre-compute the gaussian...
        self._compute_gaussian_table(scmap.get_frame_width(), scmap.get_frame_height())
        # Set down scaling.
        self._down_scaling = scmap.get_down_scaling()

        # Compute negative gaussian if negative impact setting is switched on...
        if (self.NEGATE_ON):
            self._compute_neg_gaussian_table(scmap.get_frame_width(), scmap.get_frame_height())

        # Adjust the blocks-per edge to avoid having it greater then the length of one of the sides of the frame.
        self.BLOCKS_PER_EDGE = min(scmap.get_frame_height(), scmap.get_frame_width(), self.BLOCKS_PER_EDGE)
        self._edge_block_value = self.EDGE_PROB / (self.BLOCKS_PER_EDGE * 4)  # Must be recomputed...

        # Create off edge point table of gaussian values for off-edge/on-edge transitions...
        self._compute_edge_coordinates(scmap.get_frame_width(), scmap.get_frame_height(), self.BLOCKS_PER_EDGE)
        self._edge_vals = np.zeros((self._num_frames, self._total_bp_count, self.BLOCKS_PER_EDGE * 4), dtype="float32")


    def _sparcify_and_store(self, out_bp: int, out_frame: int, srcmp_bp: int, srcmp_frame: int, scmap: TrackingData):
        """ Sparsifies the tracking data and then stores it in self._sparse_data so that it can be used by the Viterbi
            algorithm later and repeatedly... """
        self._sparse_data[out_frame][out_bp] = SparseTrackingData.sparsify(scmap, srcmp_frame, srcmp_bp, self.THRESHOLD)


    def _sparcify_source_map(self, scmap: TrackingData):
        """
        Sparcifies the all of the frames in scmap, the provided tracking data. Uses self._current_frame to figure out
        where to put these frames, but does not increment or change the value of it.
        """
        out_frame = self._current_frame

        for srcmp_frame in range(scmap.get_frame_count()):
            # Initialize all the body parts list to a list of None.
            self._sparse_data[out_frame] = [None] * self._total_bp_count

            for out_bp in range(self._total_bp_count):
                # Compute the source map body part, and then sparcify this frame.
                srcmp_bp = int(out_bp // self._num_outputs)
                self._sparcify_and_store(out_bp, out_frame, srcmp_bp, srcmp_frame, scmap)

            out_frame += 1


    def _compute_bp_neg_frame(self, out_bp: int, frame: int, bp_queue: deque):
        """
        Runs body part negation on a frame.
        """
        y, x, __, __, __ = self._sparse_data[frame][out_bp].unpack()
        probs = self._frame_probs[frame][out_bp]

        if(y is None):
            self._frame_probs[frame][out_bp] = None
            bp_queue.append((None, None, None))

        for px, py, pprob in bp_queue:
            if(px is None):
                continue
            probs = probs * self._neg_gaussian_table[np.abs(y - py), np.abs(x - px)]
            probs = probs / (np.sum(probs) + np.sum(self._edge_vals[frame, out_bp]))

        max_edge_val = np.max(self._edge_vals[frame, out_bp])
        max_loc = np.argmax(probs)

        if(probs[max_loc] >= max_edge_val):
            bp_queue.append((x[max_loc], y[max_loc], probs[max_loc]))
        else:
            bp_queue.append((None, None, None))

        self._frame_probs[frame][out_bp] = probs


    def _compute_init_frame(self, out_bp: int, frame: int):
        """ Inserts the initial frame, or first frame to have actual points that are above the threshold. """
        # Get coordinates for all above threshold probabilities in this frame...
        y, x, probs, x_off, y_off = self._sparse_data[frame][out_bp].unpack()

        # Set initial value for edge values...
        self._edge_vals[frame, out_bp, :] = self._edge_block_value

        # If no points are found above the threshold, frame for this bodypart is a dud, set it to none...
        if (y is None):
            self._frame_probs[frame][out_bp] = None
            return

        # Set first attribute for this bodypart to the y, x coordinates element wise.
        self._frame_probs[frame][out_bp] = probs * (1 - self.EDGE_PROB)


    def _compute_normal_frame(self, out_bp: int, frame_idx: int, prior_frame_idx: int,
                              trans_func: Callable[[ndarray, ndarray, ndarray, ndarray], ndarray],
                              is_secondary_pass: bool = False, use_sum: bool = True):
        """
        Computes and inserts a frame that has occurred after the first data-full frame.

        :param out_bp: Body part index to work on, an integer.
        :param frame_idx: The index of the frame to work on, integer.
        :param prior_frame_idx: The prior frame index. Considered the prior frame when forward is done.
        :param trans_func: The functions which computes transfer probabilities between the two frames. It accepts 4
                           numpy arrays (current x indexes, current y indexes, prior x indexes, prior y indexes) and
                           expects a 2 dimensional array of (current location) -> (prior location) -> (transition prob)
        :param is_secondary_pass: If set to True, don't use original data for current probabilities, but rather use
                                  data already stored in _frame_probs and _edge_vals for this current frame, which is
                                  empty on the first pass. Defaults to False.
        :param use_sum: Boolean, set to true to use sum while merging (Forward/Backward), set to false to use max while
                        merging probabilities (Viterbi).
        """
        # Get coordinates for all above threshold probabilities in this frame...
        cy, cx, current_prob, cx_off, cy_off = self._sparse_data[frame_idx][out_bp].unpack()

        # SPECIAL CASE: NO IN-FRAME FORWARD/BACKWARD/VITERBI VALUES THAT MAKE IT ABOVE THRESHOLD...
        if (cy is None):
            # In this special case, we just copy the prior frame data...
            self._sparse_data[frame_idx][out_bp] = self._sparse_data[prior_frame_idx][out_bp].duplicate()
            self._frame_probs[frame_idx][out_bp] = np.copy(self._frame_probs[prior_frame_idx][out_bp])
            self._edge_vals[frame_idx, out_bp] = self._edge_vals[prior_frame_idx, out_bp]
            return

        # NORMAL CASE:

        # Get data for the prior frame...
        py, px, pprobs, px_off, py_off = self._sparse_data[prior_frame_idx][out_bp].unpack()
        prior_vit_probs = self._frame_probs[prior_frame_idx][out_bp]

        # Scale current probabilities to include edges... (If this is not a secondary pass)
        if(is_secondary_pass):
            current_prob = self._frame_probs[frame_idx][out_bp]
        else:
            current_prob = current_prob * (1 - self.EDGE_PROB)

        # Get all of the same data for the edge values...
        edge_x, edge_y = self._edge_coords[:, 0], self._edge_coords[:, 1]
        prior_edge_probs = self._edge_vals[prior_frame_idx, out_bp, :]
        if(is_secondary_pass):
            current_edge_probs = self._edge_vals[frame_idx, out_bp, :]
        else:
            current_edge_probs = np.array([self._edge_block_value] * (self.BLOCKS_PER_EDGE * 4))

        # COMPUTE IN-FRAME FORWARD/BACKWARD/VITERBI VALUES:
        # Compute probabilities of transferring from the prior frame to this frame...
        frame_to_frame = (np.expand_dims(current_prob, axis=1) * np.expand_dims(prior_vit_probs, axis=0)
                          * trans_func(cx, cy, px, py))
        # Compute the probabilities of transferring from the prior edge to this frame.
        edge_to_frame = (np.expand_dims(current_prob, axis=1) * trans_func(cx, cy, edge_x, edge_y)
                         * np.expand_dims(prior_edge_probs, axis=0))

        # Merge probabilities of going from the edge to the frame or frame to frame, using different operation depending
        # on the algorithm selected...
        if(use_sum):
            # Forward/Backward: We sum the probabilities for every path to each current location.
            viterbi_vals = np.add(np.sum(frame_to_frame, axis=1), np.sum(edge_to_frame, axis=1))
        else:
            # Viterbi: We only keep the max path for every path to each current location.
            viterbi_vals = np.maximum(np.max(frame_to_frame, axis=1), np.max(edge_to_frame, axis=1))

        # COMPUTE OFF-SCREEN FORWARD/BACKWARD/VITERBI VALUES:
        # Compute the probability of transitioning from the prior frame to the current edge.....
        frame_to_edge = (np.expand_dims(current_edge_probs, axis=1) * trans_func(edge_x, edge_y, px, py)
                         * np.expand_dims((prior_vit_probs), axis=0))
        # Compute the probability of transitioning from the prior edge to the current edge...
        edge_to_edge = (np.expand_dims(current_edge_probs, axis=1) * np.expand_dims(prior_edge_probs, axis=0)
                        * trans_func(edge_x, edge_y, edge_x, edge_y))
        # Merge probabilities to produce final edge transitioning forward/backward/viterbi values...
        if(use_sum):
            edge_vit_vals = np.add(np.sum(frame_to_edge, axis=1), np.sum(edge_to_edge, axis=1))
        else:
            edge_vit_vals = np.maximum(np.max(frame_to_edge, axis=1), np.max(edge_to_edge, axis=1))

        # NORMALIZE PROBABILITIES:
        # If we were to leave the probabilities as-is then we would experience floating point underflow after enough
        # frames, so we normalize the probabilities before storing them to avoid this issue. Since relations and scaling
        # between the probabilities is still the same, normalizing has no effect on the results...
        total_sum = np.sum(edge_vit_vals) + np.sum(viterbi_vals)
        viterbi_vals = viterbi_vals / total_sum
        edge_vit_vals = edge_vit_vals / total_sum

        # POST FILTER PHASE:

        # Check and make sure not all values are nan or zero
        post_filter = (~np.isnan(viterbi_vals)) | (viterbi_vals > self.THRESHOLD)

        if (sum(post_filter) == 0):
            # In the case where all values are nan or zero, copy the prior frame
            self._sparse_data[frame_idx][out_bp] = self._sparse_data[prior_frame_idx][out_bp].duplicate()
            self._frame_probs[frame_idx][out_bp] = np.copy(self._frame_probs[prior_frame_idx][out_bp])
            self._edge_vals[frame_idx, out_bp] = self._edge_vals[prior_frame_idx, out_bp]
            return


        # SAVE NEW VITERBI FRAMES:
        self._edge_vals[frame_idx, out_bp] = edge_vit_vals
        self._frame_probs[frame_idx][out_bp] = viterbi_vals


    def _forward_step(self, frame_count: int):
        """ Performs a single phase of Forward. It assumes the frame has already been sparsified and stored in
            self._sparse_data. """
        # Continue on to main loop...
        for __ in range(frame_count):

            # Create a frame...
            self._frame_probs[self._current_frame] = [None] * (self._total_bp_count)

            for out_bp in range(self._total_bp_count):
                # If the prior frame was a dud frame and all frames before it where dud frames, try initializing
                # on this frame... Note in a dud frame all points are below threshold...(Also init on first frame)
                if((self._current_frame == 0) or (self._frame_probs[self._current_frame - 1][out_bp] is None)):
                    self._compute_init_frame(out_bp, self._current_frame)
                # Otherwise we can do full viterbi on this frame...
                else:
                    self._compute_normal_frame(out_bp, self._current_frame, self._current_frame - 1,
                                               self._gaussian_values_at)

            # Increment frame counter
            self._current_frame += 1


    def _backward_step(self, frame_count: int):
        """ Performs a single phase of Backward. Makes the same assumptions as _forward_step """
        # No point in commenting, identical to _forward_step, but incrementing is backwards and we start at the end.
        for __ in range(frame_count):

            self._frame_probs[self._current_frame] = [None] * (self._total_bp_count)

            for out_bp in range(self._total_bp_count):
                if(((self._current_frame + 1) == self._num_frames) or
                    (self._frame_probs[self._current_frame + 1][out_bp] is None)):
                    self._compute_init_frame(out_bp, self._current_frame)
                else:
                    self._compute_normal_frame(out_bp, self._current_frame, self._current_frame + 1,
                                               self._gaussian_values_at)

            self._current_frame -= 1


    def _post_forward_backward(self, progress_bar: Optional[tqdm.tqdm],
                               trans_func: Callable[[ndarray, ndarray, ndarray, ndarray], ndarray],
                               pre_func: Optional[Callable[[int, int], None]] = None):
        orig_frames = self._frame_probs
        orig_edges = np.copy(self._edge_vals)

        # Run forward on data currently stored in "_frame_probs" and "_edge_vals"
        for f_idx in range(1, self._num_frames):
            for bp_idx in range(self._total_bp_count):
                if(pre_func is not None):
                    pre_func(f_idx, bp_idx)
                if(self._frame_probs[f_idx - 1][bp_idx] is not None):
                    self._compute_normal_frame(bp_idx, f_idx, f_idx - 1, trans_func, True)
            if(progress_bar is not None):
                progress_bar.update(1)

        # Copy in original data again
        f_frames = self._frame_probs
        f_edges = np.copy(self._edge_vals)
        self._frame_probs = orig_frames
        self._edge_vals = orig_edges

        for f_idx in range(self._num_frames - 2, -1, -1):
            for bp_idx in range(self._total_bp_count):
                if(pre_func is not None):
                    pre_func(f_idx, bp_idx)
                if(self._frame_probs[f_idx + 1][bp_idx] is not None):
                    self._compute_normal_frame(bp_idx, f_idx, f_idx + 1, trans_func, True)
            if(progress_bar is not None):
                progress_bar.update(1)

        self._merge_probabilities(f_frames, f_edges, progress_bar)


    def on_frames(self, scmap: TrackingData) -> None:
        """ Handles Forward part of the Forward/Backward algorithm, allowing for faster post processing. """
        # Perform a step of forward, and return None to indicate we need to post process frames...
        # If gaussian table has not been computed, initialize all data structures.
        if(self._gaussian_table is None):
            self._init_data_structures(scmap)

        # Sparsify and store the source map, and then compute the forward of it....
        self._sparcify_source_map(scmap)
        self._forward_step(scmap.get_frame_count())
        return None


    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        """ Handles backward part of viterbi, and then returns the poses """
        progress_bar.reset(total=self._num_frames * (7 if(self.NEGATE_ON) else 3) - (self.NEGATE_ON * 2))
        self._complete_posterior_probs(progress_bar)

        if(self.NEGATE_ON):
            self._bp_negation_pass(progress_bar)
            self._post_forward_backward(progress_bar, self._gaussian_values_at)

        return self._get_maximums(progress_bar)


    def _bp_negation_pass(self, progress_bar: Optional[tqdm.tqdm]):
        bp_queue = deque(maxlen=self._total_bp_count - 1)

        for f_idx in range(self._num_frames):
            for bp_idx in range(self._total_bp_count):
                self._compute_bp_neg_frame(bp_idx, f_idx, bp_queue)
            if(progress_bar is not None):
                progress_bar.update(1)


    def _get_maximums(self, progress_bar: Optional[tqdm.tqdm]) -> Pose:
        # Our final pose object:
        poses = Pose.empty_pose(self._num_frames, self._total_bp_count)

        # We now do maximum selection, same as doing back tracing without transitions.
        for f_idx in range(self._num_frames):
            for bp_idx in range(self._total_bp_count):
                if(self._frame_probs[f_idx][bp_idx] is None):
                    poses.set_at(f_idx, bp_idx, (0, 0), (-4, -4), 0, 8)
                else:
                    # Get the max location in the frame....
                    y_coords, x_coords, __, x_offsets, y_offsets = self._sparse_data[f_idx][bp_idx].unpack()
                    max_loc = np.argmax(self._frame_probs[f_idx][bp_idx])
                    m_y, m_x, m_p = y_coords[max_loc], x_coords[max_loc], self._frame_probs[f_idx][bp_idx][max_loc]
                    m_offx, m_offy = x_offsets[max_loc], y_offsets[max_loc]
                    # Get the max location on the edges....
                    max_edge_loc = np.argmax(self._edge_vals[f_idx, bp_idx])
                    m_edge_prob = self._edge_vals[f_idx, bp_idx, max_edge_loc]

                    if(m_edge_prob > m_p):
                        poses.set_at(f_idx, bp_idx, (0, 0), (-4, -4), 0, 8)
                    else:
                        poses.set_at(f_idx, bp_idx, (m_x, m_y), (m_offx, m_offy), m_p, self._down_scaling)

            if(progress_bar is not None):
                progress_bar.update(1)

        return poses

    def _complete_posterior_probs(self, progress_bar: Optional[tqdm.tqdm]):
        # Make variable pointing to Forward results so we don't lose them.
        forward_frames = self._frame_probs
        forward_edges = np.copy(self._edge_vals)
        # Currently equals num_frames, need to move it down one before we enter the loop below which does array access.
        self._current_frame -= 1

        # Run Backward
        for i in range(self._num_frames):
            self._backward_step(1)
            if(progress_bar is not None):
                progress_bar.update(1)

        # Take both Forward and Backward results and merge them via multiplication, then normalize the probabilities.
        # If one of the two produces a None frame, we skip these frames.
        self._merge_probabilities(forward_frames, forward_edges, progress_bar)


    def _merge_probabilities(self, to_merge_frame: List[List[ndarray]], to_merge_edges: ndarray,
                             progress_bar: Optional[tqdm.tqdm]):
        """ Merges probabilities into _frame_probs and _edge_vals via multiplication. """
        for f_idx in range(self._num_frames):
            for bp_idx in range(self._total_bp_count):
                if((self._frame_probs[f_idx][bp_idx] is not None) and (to_merge_frame[f_idx][bp_idx] is not None)):
                    frame_result = self._frame_probs[f_idx][bp_idx] * to_merge_frame[f_idx][bp_idx]
                    edge_result = self._edge_vals[f_idx, bp_idx] * to_merge_edges[f_idx, bp_idx]
                    total_sum = np.sum(frame_result) + np.sum(edge_result)
                    self._frame_probs[f_idx][bp_idx] = frame_result / total_sum
                    self._edge_vals[f_idx, bp_idx] = edge_result / total_sum
                else:
                    self._frame_probs[f_idx][bp_idx] = None

            if(progress_bar is not None):
                progress_bar.update(1)


    @classmethod
    def get_settings(cls) -> Union[List[Tuple[str, str, Any]], None]:
        return [
            ("norm_dist", "The normal distribution of the 2D gaussian curve used"
                          "for transition probabilities by the viterbi algorithm.", 1),
            ("amplitude", "The amplitude of the gaussian curve used by the viterbi algorithm.", 1),
            ("lowest_gaussian_value", "The lowest value of the gaussian curve used by the viterbi algorithm."
                                      "Really a constant that is added on the the 2D gaussian to give all points"
                                      "a minimum probability.", 0),
            ("threshold", "The minimum floating point value a pixel within the probability frame must have"
                          "in order to be kept and added to the sparse matrix.", 0.001),
            ("edge_probability", "A constant float between 0 and 1 that determines the probability that a point goes"
                                 "off the screen. This probability is divided among edge blocks", 0.2),
            ("edge_blocks_per_side", "Number of edge blocks to have per side.", 4),
            ("negate_overlapping_predictions", "If enabled, predictor will discourage a bodypart from being in the same"
                                               "location as prior predicted body parts.", True),
            ("negative_impact_factor", "The height of the upside down 2D gaussian curve used for negating locations"
                                       "of prior predicted body parts.", 1),
            ("negative_impact_distance", "The normal distribution of the 2D gaussian curve used for negating locations"
                                         "of prior predicted body parts.", 5),
            ("negation_curve", "Curve to be used for body part negation. Options are 'Gaussian' and 'Quartic'.",
             "Quartic")
        ]

    @classmethod
    def get_name(cls) -> str:
        return "forward_backward"

    @classmethod
    def get_tests(cls) -> Union[List[Callable[[], Tuple[bool, str, str]]], None]:
        return [cls.test_plotting, cls.test_sparsification, cls.test_desparsification]

    @classmethod
    def get_test_data(cls) -> TrackingData:
        # Make tracking data...
        track_data = TrackingData.empty_tracking_data(4, 1, 3, 3, 2)

        track_data.set_prob_table(0, 0, np.array([[0, 0, 0],
                                                  [0, 1, 0],
                                                  [0, 0, 0]]))
        track_data.set_prob_table(1, 0, np.array([[0, 1.0, 0],
                                                  [0, 0.5, 0],
                                                  [0, 0.0, 0]]))
        track_data.set_prob_table(2, 0, np.array([[1, 0.5, 0],
                                                  [0, 0.0, 0],
                                                  [0, 0.0, 0]]))
        track_data.set_prob_table(3, 0, np.array([[0.5, 0, 0],
                                                  [1.0, 0, 0],
                                                  [0.0, 0, 0]]))

        return track_data

    @classmethod
    def get_test_instance(cls, track_data: TrackingData) -> "ForwardBackward":
        return cls(
            [f"part{i + 1}" for i in range(track_data.get_bodypart_count())],
            1,
            track_data.get_frame_count(),
            {name: val for name, desc, val in cls.get_settings()},
            None
        )

    @classmethod
    def test_plotting(cls) -> Tuple[bool, str, str]:
        track_data = cls.get_test_data()
        predictor = cls.get_test_instance(track_data)

        # Probabilities can change quite easily by even very minute changes to the algorithm, so we don't care about
        # them, just the predicted locations of things...
        expected_result = np.array([[3, 3], [3, 1], [1, 1], [1, 1]])

        # Pass it data...
        predictor.on_frames(track_data)

        # Check output
        poses = predictor.on_end(tqdm.tqdm(total=4)).get_all()

        if (np.allclose(poses[:, :2], expected_result)):
            return (True, "\n" + str(expected_result), "\n" + str(np.array(poses[:, :2])))
        else:
            return (False, "\n" + str(expected_result), "\n" + str(np.array(poses[:, :2])))


    @classmethod
    def test_sparsification(cls) -> Tuple[bool, str, str]:
        # Make tracking data...
        track_data = cls.get_test_data()
        predictor = cls.get_test_instance(track_data)

        # Pass it data...
        predictor.on_frames(track_data)

        # Check output
        predictor.on_end(tqdm.tqdm(total=4))

        if(any([(data[0] is None) for data in predictor._frame_probs]) or
           any([(data[0] is None) for data in predictor._sparse_data])):
            return (False, str((predictor._frame_probs, predictor._sparse_data)), "No None Entries...")
        else:
            return (True, str((predictor._frame_probs, predictor._sparse_data)), "No None Entries...")

    @classmethod
    def test_desparsification(cls):
        # Make tracking data...
        track_data = cls.get_test_data()
        orig_frame = track_data.get_prob_table(0, 0)
        result_frame = (
                SparseTrackingData.sparsify(track_data, 0, 0, 0.001)
                .desparsify(orig_frame.shape[1], orig_frame.shape[0], 8).get_prob_table(0, 0)
        )

        return (np.allclose(result_frame, orig_frame), str(orig_frame) + "\n", str(result_frame) + "\n")

    @classmethod
    def supports_multi_output(cls) -> bool:
        return True


