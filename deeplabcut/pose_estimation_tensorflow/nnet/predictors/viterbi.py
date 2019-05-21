
# For types in methods
from typing import Union, List, Tuple
from numpy import ndarray
import tqdm

# Plugin base class
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Predictor
from deeplabcut.pose_estimation_tensorflow.nnet.processing import TrackingData
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose

# For computations
import numpy as np

# TODO: Might change so that a 2D array of gaussian values is precomputed for speed...(In the constructor...)

class Viterbi(Predictor):
    """
    A predictor that applies the Viterbi algorithm to frames in order to predict poses.
    The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but
    is also more memory intensive and computationally expensive.
    """
    # Global values for the gaussian formula, can be adjusted for differing results...
    NORM_DIST = 1  # The normal distribution
    AMPLITUDE = 1  # The amplitude, or height of the gaussian curve


    def __init__(self, bodyparts: List[str], num_frames: int):
        super().__init__(bodyparts, num_frames)

        # Store bodyparts and num_frames for later use, they become useful
        self._bodyparts = bodyparts
        self._num_frames = num_frames
        # Used to store viterbi frames
        self._viterbi_frames: TrackingData= None
        self._current_frame = 0



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
        return self.AMPLITUDE * np.exp(-(inner_x_delta + inner_y_delta))


    def _compute_frame(self, prior_frame: ndarray, current_frame: ndarray) -> ndarray:
        """
        Computes the viterbi frame given a current frame and the prior viterbi frame...

        :param prior_frame: The prior 2D(y, x) viterbi frame, for a given frame and body part
        :param current_frame: The 2D(y, x) non-viterbi current frame, of the next frame but same body part.
        :return: The viterbi 2D frame for the current frame.
        """
        viterbi_frame = np.zeros(current_frame.shape, dtype=current_frame.dtype)

        # Iterating all x and y in current frame
        for y in current_frame.shape[0]:
            for x in current_frame.shape[1]:
                # Create variable to store the max value
                best_val = None

                # Iterate the prior frame computing viterbi values at the point, and also find the maximum...
                for py in prior_frame.shape[0]:
                    for px in prior_frame.shape[1]:
                        # Viterbi computation for single point...on the log scale...
                        temp = (prior_frame[py, px] + np.log(self._gaussian_formula(px, x, py, y)) +
                                np.log(current_frame[y, x]))

                        if(best_val is None):
                            best_val = temp
                        elif(best_val < temp):
                            best_val = temp
                # Set the value for this point in the viterbi frame to the max...
                viterbi_frame[y, x] = best_val

        # Return the max...
        return viterbi_frame


    def _back_compute(self, current_frame: ndarray, prior_point: Tuple[int, int, float]) -> ndarray:
        """
        Performs a back computation, computing viterbi frame of current frame using a point in front of it

        :param current_frame: The current frame to compute a viterbi for...
        :param prior_point: The prior point, or the predicted value for the frame in front of the current frame.
                            A tuple containing the y, x and probability of the prior point...
        :return: A numpy array being the full viterbi frame for the current frame...
        """
        py, px, prob = prior_point
        # Create the new frame
        new_frame = np.zeros(current_frame.shape, dtype=current_frame.dtype)

        for cy in current_frame.shape[0]:
            for cx in current_frame.shape[1]:
                # Applying same viterbi backwards, but since we know the max of the frame in front we don't need to
                # compute it for all points of the prior frame, but rather one point, as seen here...
                new_frame[cy, cx] = [np.log(current_frame[cy, cx]) + np.log(self._gaussian_formula(px, cx, py, cx)) +
                                     prob]

        return new_frame


    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        """ Handles Forward part of the viterbi algorithm, allowing for faster post processing. """
        # Check if this is the first frame
        if(self._viterbi_frames is None):
            # Create an empty tracking data object
            self._viterbi_frames = TrackingData.empty_tracking_data(self._num_frames, len(self._bodyparts),
                                   scmap.get_frame_width(), scmap.get_frame_height(), scmap.get_down_scaling())
            # Set offset map to empty array, we will eventually just copy all frame offsets over...
            self._viterbi_frames.set_offset_map(np.zeros((self._num_frames, scmap.get_frame_height(),
                                                scmap.get_frame_width(), len(self._bodyparts), 2), dtype="float32"))

            # Add the first frame...
            self._viterbi_frames.get_source_map()[0] = np.log(scmap.get_source_map()[0])
            scmap.set_source_map(scmap.set_source_map()[1:])

            self._current_frame += 1

        for frame in range(scmap.get_frame_count()):
            # Copy over offset map for this frame...
            self._viterbi_frames.get_offset_map()[self._current_frame] = scmap.get_offset_map()[frame]

            for bodypart in range(scmap.get_bodypart_count()):
                # Compute viterbi frame for this frame and body part...
                frame_data = self._viterbi_frames.get_prob_table(self._current_frame, bodypart)
                prior_frame = self._viterbi_frames.get_prob_table(self._current_frame - 1, bodypart)
                self._viterbi_frames.set_prob_table(self._current_frame, bodypart, self._compute_frame(prior_frame, frame_data))
            # Increment global frame counter...
            self._current_frame += 1

        # Return None since we are storing frames
        return None



    def on_end(self, progress_bar: tqdm.tqdm) -> Union[None, Pose]:
        """ Handles backward part of viterbi, and then returns the poses """
        r_counter = self._num_frames - 1
        # To store final poses....
        poses = Pose.empty_pose(self._num_frames, len(self._bodyparts))

        prior_points = []

        width = self._viterbi_frames.get_frame_width()
        height = self._viterbi_frames.get_frame_height()

        # Just get max of first frame....
        for bp in range(self._viterbi_frames.get_bodypart_count()):
            # Compute the max...
            table = self._viterbi_frames.get_prob_table(r_counter, bp)
            y, x = np.unravel_index(np.argmax(table, shape=(height, width)))
            prob = table[y, x]
            # Set the pose at this point....
            self._viterbi_frames.set_pose_at(r_counter, bp, x, y, poses)
            # Add to prior_point
            prior_points.append((y, x, prob))

            r_counter -= 1
            progress_bar.update()

        # Now begin main loop...
        while(r_counter >= 0):
            # The current body part points...
            current_points = []
            # For every body part...
            for bp in range(self._viterbi_frames.get_bodypart_count()):
                # Compute the max of the viterbi probability table
                table = self._back_compute(self._viterbi_frames.get_prob_table(r_counter, bp), prior_points[bp])
                y, x = np.unravel_index(np.argmax(table, shape=(height, width)))
                prob = table[y, x]
                # Set the point in the pose object and append it to current points
                self._viterbi_frames.set_pose_at(r_counter, bp, x, y, poses)
                current_points.append((y, x, prob))

            # Decrement the counter and set the prior points to the current points
            r_counter -= 1
            progress_bar.update()
            prior_points = current_points

        # Done, return the predicted poses...
        return poses


    @staticmethod
    def get_name() -> str:
        return "viterbi"

    @staticmethod
    def get_description() -> str:
        return ("A predictor that applies the Viterbi algorithm to frames in order to predict poses.\n"
                "The algorithm is frame-aware, unlike the default algorithm used by DeepLabCut, but\n"
                "is also more memory intensive and computationally expensive.")