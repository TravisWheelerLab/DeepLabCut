from typing import Tuple, List

import wx
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.video_player import VideoPlayer
import cv2
import matplotlib.pyplot as plt
from wx.lib.newevent import NewCommandEvent

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

    PointChangeEvent, EVT_POINT_CHANGE = NewCommandEvent()
    PointInitEvent, EVT_POINT_INIT = NewCommandEvent()

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
        self._pressed = False
        # Handle point changing events....
        self.Bind(wx.EVT_LEFT_DOWN, self.on_press)
        self.Bind(wx.EVT_MOTION, self.on_move)
        self.Bind(wx.EVT_LEFT_UP, self.on_release)

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

    def _get_selected_bodypart(self) -> Tuple[float, float, float]:
        x = self._poses.get_x_at(self.get_offset_count(), self._edit_point)
        y = self._poses.get_y_at(self.get_offset_count(), self._edit_point)
        prob = self._poses.get_prob_at(self.get_offset_count(), self._edit_point)
        return x, y, prob

    def _set_selected_bodypart(self, x: float, y: float, probability: float):
        self._poses.set_x_at(self.get_offset_count(), self._edit_point, x)
        self._poses.set_y_at(self.get_offset_count(), self._edit_point, y)
        self._poses.set_prob_at(self.get_offset_count(), self._edit_point, probability)

    def _get_mouse_loc_video(self, evt: wx.MouseEvent):
        total_w, total_h = self.GetClientSize()
        if((not total_w) or (not total_h) or (self._current_frame is None)):
            return (None, None)

        # orig_x, orig_y = self.GetScreenPosition().Get()
        x = evt.GetX() # - orig_x
        y = evt.GetY() # - orig_y
        # Now we need to translate into video coordinates...
        x_off, y_off, w, h = self._get_video_bbox(self._current_frame, total_w, total_h)
        v_h, v_w = self._current_frame.shape[:2]

        final_x, final_y = (x - x_off) * (v_w / w), (y - y_off) * (v_h / h)
        final_x, final_y = max(0, min(final_x, v_w)), max(0, min(final_y, v_h))

        return (final_x, final_y)

    def _push_point_change_event(self, new_point: Tuple[float, float, float], old_point: Tuple[float, float, float]):
        new_evt = self.PointChangeEvent(id=self.Id, frame=self.get_offset_count(), part=self._edit_point,
                                        new_location=new_point, old_location=old_point)
        wx.PostEvent(self, new_evt)

    def _push_point_init_event(self, old_point: Tuple[float, float, float]):
        new_evt = self.PointInitEvent(id=self.Id, frame=self.get_offset_count(), part=self._edit_point,
                                      current_location = old_point)
        wx.PostEvent(self, new_evt)

    def on_press(self, event: wx.MouseEvent):
        if(not self.is_playing() and self._edit_point is not None):
            self.freeze()
            self._old_location = self._get_selected_bodypart()
            self._pressed = True
            self._push_point_init_event(self._old_location)

    def on_move(self, event: wx.MouseEvent):
        if(self.is_playing() or self._edit_point is None):
            self._pressed = False
            return

        if(self._pressed and event.LeftIsDown()):
            x, y = self._get_mouse_loc_video(event)
            print(self._old_location)
            print(x, y)
            if(x is None):
                return
            self._set_selected_bodypart(x, y, 1)
            self.Refresh()

    def on_release(self, event: wx.MouseEvent):
        if((self.is_playing()) or (self._edit_point is None)):
            self._pressed = False
            return

        if(self._pressed and event.LeftUp()):
            x, y = self._get_mouse_loc_video(event)
            if(x is None):
                return
            self._set_selected_bodypart(x, y, 1)
            self._push_point_change_event((x, y, 1), self._old_location)
            self._old_location = None
            self._pressed = False
            self.unfreeze()
            self.Refresh()

    def get_selected_body_part(self) -> int:
        return self._edit_point

    def set_selected_bodypart(self, value: int):
        if(not (0 <= value <= self._poses.get_bodypart_count())):
            raise ValueError("Selected Body part not within range!")
        self._edit_point = value


class PointEditor(wx.Panel):

    def __init__(self,
        parent,
        video_hdl: cv2.VideoCapture,
        poses: Pose,
        bp_names: List[str],
        colormap: str = PointViewNEdit.DEF_MAP,
        plot_threshold: float = 0.1,
        point_radius: int = 5,
        point_alpha: float = 0.7,
        w_id = wx.ID_ANY,
        pos = wx.DefaultPosition,
        size = wx.DefaultSize,
        style = wx.TAB_TRAVERSAL,
        name = "PointEditor"
    ):
        super().__init__(parent, w_id, pos, size, style, name)

        if(poses.get_bodypart_count() != len(bp_names)):
            raise ValueError("Length of the body part names provided does not match body part count of poses object!")

        self._main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.video_viewer = PointViewNEdit(self, video_hdl, poses, colormap, plot_threshold, point_radius, point_alpha)
        self.select_box = wx.RadioBox(self, label="Body Part", choices=bp_names, style=wx.RA_SPECIFY_ROWS)

        self._main_sizer.Add(self.video_viewer, 1, wx.EXPAND)
        self._main_sizer.Add(self.select_box, 0, wx.EXPAND)

        self.SetSizerAndFit(self._main_sizer)

        self.select_box.Bind(wx.EVT_RADIOBOX, self.on_radio_change)
        self.on_radio_change(None)


    def on_radio_change(self, event):
        idx = self.select_box.GetSelection()

        if(idx != wx.NOT_FOUND):
            self.video_viewer.set_selected_bodypart(idx)

# class EditPointsControl()