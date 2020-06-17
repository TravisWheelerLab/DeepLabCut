import wx
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.video_player import VideoPlayer
import cv2
import matplotlib.pyplot as plt

def _bounded_float(low: float, high: float):
    def convert(value: float):
        value = float(value)
        if(not (low <= value <= high)):
            raise ValueError(f"{value} is not between {low} and {high}!")
        return value

    return convert


class Initialisable:

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        cls.init_class()

    @classmethod
    def init_class(cls):
        raise NotImplementedError("Abstract method that must be initialized...")


class BasicDataFields(Initialisable):
    METHOD_LIST = []

    @classmethod
    def init_class(cls):
        for name, private_name, cast_method in cls.METHOD_LIST:
            cls._add_methods(name, private_name, cast_method)

    @classmethod
    def _add_methods(cls, name, private_name, cast_method):
        setattr(cls, f"get_{name}", lambda self: getattr(self, private_name))
        setattr(cls, f"set_{name}", lambda self, value: setattr(self, private_name, cast_method(value)))

    def __init__(self, *args, **kwargs):
        names = {public_name for public_name, __, __ in self.METHOD_LIST}

        for i in range(min(len(args), len(self.METHOD_LIST))):
            getattr(self, f"set_{self.METHOD_LIST[i][0]}")(args[i])

        # Keyword arguments will override positional arguments...
        for name, value in kwargs.items():
            if(name in names):
                getattr(self, f"set_{name}")(value)


class PointViewNEdit(VideoPlayer, BasicDataFields):

    DEF_MAP = None
    METHOD_LIST = [
        ("colormap", "_colormap", plt.get_cmap),
        ("plot_threshold", "_plot_threshold", _bounded_float(0, 1)),
        ("point_radius", "_point_radius", int),
        ("point_alpha", "_point_alpha", _bounded_float(0, 1))
    ]

    def __init__(
        self,
        parent,
        video_hdl: cv2.VideoCapture,
        poses: Pose,
        colormap: str = DEF_MAP,
        plot_threshold: float = 0.1,
        point_radius: int = 5,
        point_alpha: float = 0.7,
        w_id=wx.ID_ANY,
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=wx.BORDER_DEFAULT,
        validator=wx.DefaultValidator,
        name="VideoPlayer"
    ):
        VideoPlayer.__init__(self, parent, w_id, video_hdl, pos, size, style, validator, name)

        self._poses = poses
        self._colormap = None
        self._plot_threshold = None
        self._point_radius = None
        self._point_alpha = None

        BasicDataFields.__init__(self, colormap, plot_threshold, point_radius, point_alpha)

        self._edit_point = None
        self._new_location = None
        self._old_location = None
        # Handle point changing events....
        # self.Bind(wx.EVT_LEFT_DOWN, self.on)

    def on_draw(self, dc: wx.BufferedPaintDC):
        super().on_draw(dc)

        width, height = self.GetClientSize()
        if((not width) or (not height)):
            return

        ov_h, ov_w = self._current_frame.shape[:2]
        nv_w, nv_h = self._get_resize_dims(self._current_frame, width, height)
        x_off, y_off = (width - nv_w) // 2, (height - nv_h) // 2

        num_out = self._poses.get_bodypart_count()
        colormap = plt.get_cmap(self._colormap)
        frame = self.get_offset_count()

        for bp_idx in range(num_out):
            x = self._poses.get_x_at(frame, bp_idx)
            y = self._poses.get_y_at(frame, bp_idx)
            prob = self._poses.get_prob_at(frame, bp_idx)

            if(prob < self._plot_threshold):
                continue

            color = colormap(bp_idx / num_out, bytes=True)
            wx_color = wx.Colour(*color)
            dc.SetPen(wx.Pen(wx_color, 2, wx.PENSTYLE_SOLID))
            dc.SetBrush(wx.Brush(wx_color, wx.BRUSHSTYLE_SOLID))

            dc.DrawCircle((x * (nv_w / ov_w)) + x_off, (y * (nv_h / ov_h)) + y_off, self._point_radius)

    def _get_selected_bodypart(self):
        pass

    def on_press(self, event: wx.MouseEvent):
        if(not self.is_playing() and self._edit_point is not None):
            self._old_location = ()



print(dir(PointViewNEdit))
# class EditPointsControl()