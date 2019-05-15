"""
Author: Isaac Robinson

Contains Abstract Base Class for all predictor plugins, which when provided probability frames from the neural net,
figure out where the points should be in the image. They are executed when deeplabcut.analyze_videos is run...
"""
# Abstract class stuff
from abc import ABC
from abc import abstractmethod

# Used for type hints
from numpy import ndarray
from typing import List, Union, Type, Tuple, Iterable, Sequence

# Used by get_predictor for loading plugins
import deeplabcut.pose_estimation_tensorflow.util.pluginloader as loader

# Used by TrackData class
import numpy as np


class TrackingData:
    """
    Represents tracking data recieved from the DeepLabCut neural network. Includes the source map of probabilities,
    the location offset, scaling info and ect. Also provides many convienence methods for working with getting info
    from this data.
    """

    # The default image down scaling used by DeepLabCut
    DEFAULT_SCALE: int= 8


    def __init__(self, scmap: ndarray, locref: ndarray = None, scaling: int= DEFAULT_SCALE):
        """
        Create an new track data object to store tracking data for one frame or a batch of frames

        :param scmap: The probability maps produced by the neural network, a 4-dimensional array containing the
                      dimensions: [frame, y location, x location, body part].
        :param locref: The "offsets" produced by DeepLabCut neural network, stored in a 5-dimensional array containing
                       the dimensions: [frame, y location, x location, bodypart, 0 for x offset or 1 for y offset]
        :param scaling: The down scaling done to probability map to decrease it's size relative to the original video,
                        stored as an integer.
        """
        # If scmap recieved is only 3-dimension, it is of only 1 frame, but add the dimension so it works better.
        if(len(scmap.size) == 3):
            self._scmap = np.expand_dims(scmap, axis=0)
        else:
            self._scmap = scmap

        if(len(locref.size) == 3):
            self._locref = np.expand_dims(locref, axis=0)
        else:
            self._locref = locref

        self._scaling = scaling


    def get_source_map(self) -> ndarray:
        """
        Gets the raw probability source map of this tracking data.

        :return: A numpy array representing the source probability map of this tracking data. It is a 4-dimensional
                array containing the dimensions: [frame, y location, x location, body part]
        """
        return self._scmap

    def get_offset_map(self) -> ndarray:
        """
        Gets the offset map for precision offsets for each point.

        :return: A numpy array representing the offsets within the scmap.
        """
        return self._locref

    def get_down_scaling(self) -> int:
        """
        Gets the down scaling performed on the source map, as an integer

        :return: An integer representing the downscaling of the source map compared to the original video file
        """
        return self._scaling

    def get_max_scmap_points(self) -> Tuple[ndarray, ndarray]:
        """
        Gets the maximum points for each frame in the array

        :return: A tuple of numpy arrays, the first numpy array being the y coordinate max for each frame, the second
                 being the x coordinate max for each frame
        """
        y_dim, x_dim = self._scmap.size[1], self._scmap.shape[2]
        flat_max = np.argmax(self._scmap.reshape((self._scmap.size[0], y_dim * x_dim, self._scmap.size[3])), axis=1)
        return np.unravel_index(flat_max, dims=(y_dim, x_dim))

    def get_poses_for(self, points: Tuple[ndarray, ndarray]):
        """
        Get the pose object for the specified numpy array of points.

        :param points: A tuple of 2 numpy arrays, one representing the y values for each frame and body part in the
                       frame, the other being the x values represented the same way.
        :return: The Pose object representing all poses for selected points...
        """
        y, x = points
        # Create new numpy array to store probabilities, x offsets, and y offsets...
        probs = np.zeros(x.shape)
        x_offsets = np.zeros(x.shape)
        y_offsets = np.zeros(y.shape)

        # Iterate the frame and body part indexes in x and y, we just use x since both are the same size
        for frame in range(x.shape[0]):
            for bp in range(x.shape[1]):
                probs[frame, bp] = self._scmap[frame, y[frame, bp], x[frame, bp], bp]
                # Locref is frame -> y -> x -> bodypart -> relative coordinate pair offset
                x_offsets[frame, bp], y_offsets[frame, bp] = self._locref[frame, y[frame, bp], x[frame, bp], bp]

        # Now apply offsets to x and y to get actual x and y coordinates...
        # Done by multiplying by scale, centering in the middle of the "scale square" and then adding extra offset
        x = x.astype("float") * self._scaling + (0.5 * self._scaling) + x_offsets
        y = y.astype("float") * self._scaling + (0.5 * self._scaling) + y_offsets

        # Create and return a new pose object now....
        return Pose(x, y, probs)

    def get_prob_table_for(self, frame: Union[int, slice, Sequence[int]], bodypart: Union[int, slice, Sequence[int]]) -> ndarray:
        """
        Get the probability table for a selection of frames and body parts or a single frame and body part.

        :param frame: The frame index, as an integer or slice.
        :param bodypart: The body part index, as an integer or slice.
        :return: The probability map for a single frame or selection of frames based on indexes, as a numpy array...
        """
        # Utility method to get length of an index selection(as in how many indexes it selects...)
        def get_count_of(val: Union[int, slice, bodypart], length: int) -> int:
            if (isinstance(val, Sequence)):
                return len(val)
            elif (isinstance(val, slice)):
                start, stop, step = val.indices(length)
                return len(range(start, stop - 1, step))
            elif(isinstance(val, int)):
                return 1
            else:
                raise ValueError("Value is not a slice, integer, or list...")

        # Compute amount of frames and body parts selected....
        frame_count = get_count_of(frame, self.get_frame_count())
        part_count = get_count_of(bodypart, self.get_bodypart_count())

        # Return the frames, reshaped to be more "frame like"...
        return (self._scmap[frame, :, :, bodypart]
                   .reshape((frame_count, self.get_frame_height(), self.get_frame_width(), part_count))
                   .squeeze())

    def get_frame_count(self) -> int:
        """
        Get the number of frames stored in this TrackingData object.

        :return: The number of frames stored in this tracking data object.
        """
        return self._scmap.shape[0]

    def get_bodypart_count(self) -> int:
        """
        Get the number of body parts stored in this TrackingData object per frame.

        :return: The number of body parts per frame as an integer.
        """
        return self._scmap.shape[3]

    def get_frame_width(self) -> int:
        """
        Return the width of each probability map in this TrackingData object.

        :return: The width of each map as an integer.
        """
        return self._scmap.shape[2]

    def get_frame_height(self) -> int:
        """
        Return the height of each probability map in this TrackingData object.

        :return: The height of each map as an integer.
        """
        return self._scmap.shape[1]


class Pose:
    """
    Class defines the poses for given amount of frames
    """
    def __init__(self, x: ndarray, y: ndarray, prob: ndarray):
        """
        Create a new Pose object, or batch of poses for frames

        :param x: All x-values for these poses, in ndarray indexing format frame->body part->x-value
        :param y: All y-values for these poses, in ndarray indexing format frame->body part->y-value
        :param prob: All probabilities for these poses, in ndarray indexing format frame->body part->p-value
        """
        self._data = np.empty((x.size[0], x.size[1] * 3), dtype=x.dtype)
        self.set_all_x(x)
        self.set_all_y(y)
        self.set_all_prob(prob)

    # Helper Methods

    def _fix_index(self, index: Union[int, slice], value_offset: int) -> Union[int, slice]:
        """
        Fixes slice or integer indexing received by user for body part to fit the actual way it is stored.
        PRIVATE METHOD! Should not be used outside this class, for internal index correction!

        :param index: An integer or slice representing indexing
        :param value_offset: An integer representing the offset of the desired values in stored data
        :return: Slice or integer, being the fixed indexing to actually get the body parts
        """
        if(isinstance(index, int)):
            # Since all data is all stored together, multiply by 3 and add the offset...
            return (index * 3) + value_offset
        elif(isinstance(index, slice)):
            # Normalize the slice and adjust the indexes.
            start, end, step = index.indices(self._data.shape[1] // 3)
            return slice((start * 3) + value_offset, (end * 3) + value_offset, step * 3)
        else:
            raise ValueError("Index is not of type slice or integer!")

    # Setter Methods

    def set_all_x(self, x: ndarray):
        """
        Sets the x values of this batch of Poses

        :param x: An ndarray with same dimensions as this pose object, providing all x-values
        """
        self._data[:, 0::3] = x

    def set_all_y(self, y: ndarray):
        """
        Sets the y values of this batch of Poses

        :param y: An ndarray with same dimensions as this pose object, providing all y-values
        """
        self._data[:, 1::3] = y

    def set_all_prob(self, probs: ndarray):
        """
        Sets the probability values of this batch of Poses

        :param probs: An ndarray with same dimensions as this pose object, providing all probability values for given x, y
                  points...
        """
        self._data[:, 2::3] = probs

    def set_x_at(self, frame: Union[int, slice], bodypart: Union[int, slice], values: ndarray):
        """
        Set the x-values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :param values: The x-values to set this pose's x-values to, as a numpy array
        """
        self._data[frame, self._fix_index(bodypart, 0)] = values

    def set_y_at(self, frame: Union[int, slice], bodypart: Union[int, slice], values: ndarray):
        """
        Set the y-values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :param values: The y-values to set this pose's x-values to, as a numpy array
        """
        self._data[frame, self._fix_index(bodypart, 1)] = values

    def set_prob_at(self, frame: Union[int, slice], bodypart: Union[int, slice], values: ndarray):
        """
        Set the probability values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :param values: The x-values to set this pose's probability values to, as a numpy array
        """
        self._data[frame, self._fix_index(bodypart, 2)] = values

    # Getter Methods

    def get_all(self) -> ndarray:
        """
        Returns all data combined together into a numpy array. Note method is mostly useful to DLC, not Predictor
        plugins.

        :return: A numpy array with indexing of the style frame -> x, y or prob every 3-slots.
        """
        return self._data

    def get_all_x(self) -> ndarray:
        """
        Returns x-data for all frames and body parts.

        :return: The x-data for all frames and body parts
        """
        return self._data[:, 0::3]

    def get_all_y(self) -> ndarray:
        """
        Returns y-data for all frames and body parts.

        :return: The y-data for all frames and body parts
        """
        return self._data[:, 1::3]

    def get_all_prob(self) -> ndarray:
        """
        Returns probability data for all frames and body parts.

        :return: The probability data for all frames and body parts
        """
        return self._data[:, 2::3]

    def get_x_at(self, frame: Union[int, slice], bodypart: Union[int, slice]) -> ndarray:
        """
        Get the x-values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :returns: The x-values for the given frames, in the form of a numpy array...
        """
        return self._data[frame, self._fix_index(bodypart, 0)]

    def get_y_at(self, frame: Union[int, slice], bodypart: Union[int, slice]) -> ndarray:
        """
        Get the y-values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :returns: The y-values for the given frames, in the form of a numpy array...
        """
        return self._data[frame, self._fix_index(bodypart, 1)]

    def get_prob_at(self, frame: Union[int, slice], bodypart: Union[int, slice]) -> ndarray:
        """
        Get the probability values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :returns: The probability values for the given frames, in the form of a numpy array...
        """
        return self._data[frame, self._fix_index(bodypart, 2)]


class Predictor(ABC):
    """
    Base plugin class for all predictor plugins.

    Predictors accept a source map of data received.
    """
    # TODO: Maybe add generator for on_end so DLC can display a progress bar for post processing if it exist
    @abstractmethod
    def __init__(self, bodyparts: List[str]):
        """
        Constructor for the predictor.

        :param bodyparts: The bodyparts for the dataset, a list of the string friendly names in order.
        """
        pass

    @abstractmethod
    def on_frames(self, scmap: TrackingData) -> Union[None, Pose]:
        """
        Executed on every frame on the video, processing or stores source maps data and returns the guessed max
        locations.

        :param scmap: A TrackingData object, containing probability maps, offset maps, and all data and methods needed
                      to generate poses.

        :return: A Pose object representing a collection of poses for frames and body parts, or None if TrackingData
                 objects need to be stored since algorithm requires post-processing.
        """
        pass

    @abstractmethod
    def on_end(self) -> Union[None, Pose]:
        """
        Executed once all frames have been run through. Should be used for post-processing, if it needs to store all
        of the frames in order to do it's prediction algorithm.
        :return: A Pose object representing a collection of poses for frames and body parts, or None if TrackingData
                 objects were already converted to poses in on_frame method and returned.
        """
        pass


    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """
        Get the name of this predictor plugin, name is used when selecting a predictor in deeplabcut.analyze_videos

        :return: The name of this plugin to be used to select it, as a string.
        """
        pass


    @staticmethod
    @abstractmethod
    def get_description() -> str:
        """
        Get the description of this plugin, the equivalent of a doc-string for this plugin, is displayed when
        user lists available plugins

        :return: The description/summary of this plugin.
        """
        pass


def get_predictor(name: str) -> Type[Predictor]:
    """
    Gets the predictor plugin by the specified given name.

    :param name: The name of this plugin, should be a string
    :return: The plugin class that has a name that matches the specified name
    """
    # Load the plugins
    plugins = loader.load_plugin_classes("deeplabcut.pose_estimation_tensorflow.nnet.predictors", Predictor)

    # Iterate the plugins until we find one with a matching name, otherwise throw a ValueError if we don't find one.
    for plugin in plugins:
        if(plugin.get_name() == name):
            return plugin
    else:
        raise ValueError(f"Predictor plugin {name} does not exist, try another plugin name...")

