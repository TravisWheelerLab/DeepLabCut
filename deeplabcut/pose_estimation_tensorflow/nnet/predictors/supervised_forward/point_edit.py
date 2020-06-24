from typing import Tuple, List, Optional, Union

import wx
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.video_player import VideoPlayer
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from wx.lib.newevent import NewCommandEvent
from wx.lib.scrolledpanel import ScrolledPanel

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
        colormap: Union[str, Colormap] = DEF_MAP,
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
        return float(x), float(y), float(prob)

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

    def get_selected_body_part(self) -> Optional[int]:
        return self._edit_point

    def set_selected_bodypart(self, value: Optional[int]):
        if(value is None):
            self._edit_point = None
            return
        if(not (0 <= value <= self._poses.get_bodypart_count())):
            raise ValueError("Selected Body part not within range!")
        self._edit_point = value


class ColoredCircle(wx.Control):
    def __init__(self, parent, color: wx.Colour, w_id = wx.ID_ANY, pos = wx.DefaultPosition,
                 size = wx.DefaultSize, style=wx.BORDER_NONE, validator=wx.DefaultValidator, name = "ColoredCircle"):
        super().__init__(parent, w_id, pos, size, style, validator, name)

        self._color = color
        self.SetInitialSize(size)
        self.SetSize(size)

        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_paint(self, event):
        self.on_draw(wx.GCDC(wx.BufferedPaintDC(self)))

    def on_draw(self, dc: wx.DC):
        width, height = self.GetClientSize()

        if((not width) or (not height)):
            return

        dc.SetBackground(wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID))
        dc.Clear()

        dc.SetBrush(wx.Brush(self._color, wx.BRUSHSTYLE_SOLID))
        dc.SetPen(wx.Pen(self._color, 1, wx.PENSTYLE_TRANSPARENT))

        circle_radius = min(width, height) // 2
        dc.DrawCircle(width // 2, height // 2, circle_radius)

    def get_circle_color(self) -> wx.Colour:
        return self._color

    def set_circle_color(self, value: wx.Colour):
        self._color = wx.Colour(value)


class ColoredRadioButton(wx.Panel):

    ColoredRadioEvent, EVT_COLORED_RADIO = NewCommandEvent()
    PADDING = 10

    def __init__(self, parent, button_idx: int, color: wx.Colour, label: str, w_id = wx.ID_ANY, pos = wx.DefaultPosition,
                 size = wx.DefaultSize, style = wx.TAB_TRAVERSAL, name = "ColoredRadioButton"):
        super().__init__(parent, w_id, pos, size, style, name)

        self._index = button_idx

        self._sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.radio_button = wx.CheckBox(self, label=label, style=wx.CHK_2STATE)
        height = self.radio_button.GetBestSize().GetHeight()
        self.circle = ColoredCircle(self, color, size=wx.Size(height, height))

        self.radio_button.SetValue(False)

        self._sizer.Add(self.circle, 0, wx.EXPAND, self.PADDING)
        self._sizer.Add(self.radio_button, 1, wx.EXPAND, self.PADDING)

        self.SetSizerAndFit(self._sizer)

        self.SetInitialSize(size)

        self.radio_button.Bind(wx.EVT_CHECKBOX, self._send_event)

    def _send_event(self, event):
        evt = self.ColoredRadioEvent(id=self.Id, button_id=self._index, label=self.radio_button.GetLabelText())
        wx.PostEvent(self, evt)


class ColoredRadioBox(wx.Panel):

    ColoredRadioEvent, EVT_COLORED_RADIO = ColoredRadioButton.ColoredRadioEvent, ColoredRadioButton.EVT_COLORED_RADIO
    PADDING = 20

    def __init__(self, parent, colormap: Union[str, Colormap], labels: List[str], w_id = wx.ID_ANY,
                 pos = wx.DefaultPosition, size = wx.DefaultSize, style = wx.TAB_TRAVERSAL | wx.BORDER_DEFAULT,
                 name = "ColoredRadioBox"):
        super().__init__(parent, w_id, pos, size, style, name)

        self._scroller = ScrolledPanel(self, style=wx.VSCROLL)
        self._main_sizer = wx.BoxSizer(wx.VERTICAL)

        self._inner_sizer = wx.BoxSizer(wx.VERTICAL)
        self._buttons = []
        self._selected = None

        self._colormap = plt.get_cmap(colormap)

        for i, label in enumerate(labels):
            color = self._colormap(i / len(labels), bytes=True)
            wx_color = wx.Colour(*color)

            radio_button =  ColoredRadioButton(self._scroller, i, wx_color, label)
            radio_button.Bind(ColoredRadioButton.EVT_COLORED_RADIO, self._enforce_single_select)
            self._inner_sizer.Add(radio_button, 0, wx.EXPAND, self.PADDING)
            self._buttons.append(radio_button)

        self._scroller.SetSizerAndFit(self._inner_sizer)
        self._scroller.SetMinSize(wx.Size(self._scroller.GetMinSize().GetWidth() + self.PADDING, -1))
        self._scroller.SetAutoLayout(True)
        self._scroller.SetupScrolling()
        self._scroller.SendSizeEvent()

        self._main_sizer.Add(self._scroller, 1, wx.EXPAND)
        self.SetSizerAndFit(self._main_sizer)

    def _correct_sidebar_size(self, forward_now = True):
        self._scroller.Fit()
        self._scroller.SetMinSize(wx.Size(self._scroller.GetMinSize().GetWidth() + self.PADDING, -1))
        if(forward_now):
            self.SendSizeEvent()

    def _enforce_single_select(self, event: ColoredRadioButton.ColoredRadioEvent, post: bool = True):
        # If we clicked on the already selected widget, toggle it off...
        if(self._selected == event.button_id):
            event.button_id = None

        # Disable all radio buttons except for the one that was just toggled on.
        for i, radio_button in enumerate(self._buttons):
            radio_button.radio_button.SetValue(i == event.button_id)

        self._selected = event.button_id

        # Repost the event on this widget...
        if(post):
            wx.PostEvent(self, event)

    def get_selected(self) -> Optional[int]:
        return self._selected

    def set_selected(self, value: int):
        if(not (0 <= value < len(self._buttons)) and (value is not None)):
            raise ValueError("Not a valid selection!!!!")
        if(value is not None):
            value = int(value)

        new_evt = ColoredRadioButton.ColoredRadioEvent(button_id=value)
        self._enforce_single_select(new_evt, False)

    def get_labels(self) -> List[str]:
        return [button.radio_button.GetLabel() for button in self._buttons]

    def set_labels(self, value: List[str]):
        if(len(self._buttons) != len(value)):
            raise ValueError("Length of labels does not match the number of radio buttons!")

        for button, label in zip(self._buttons, value):
            button.SetLabel(label)

        self._correct_sidebar_size()

    def get_colormap(self) -> Colormap:
        return self._colormap

    def set_colormap(self, value: Union[str, Colormap]):
        self._colormap = plt.get_cmap(value)

        for i, button in enumerate(self._buttons):
            color = self._colormap(i / len(self._buttons), bytes=True)
            wx_color = wx.Colour(*color)
            button.circle.set_circle_color(wx_color)


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
        self.select_box = ColoredRadioBox(self, colormap, bp_names)

        self._main_sizer.Add(self.video_viewer, 1, wx.EXPAND)
        self._main_sizer.Add(self.select_box, 0, wx.EXPAND)

        self.SetSizerAndFit(self._main_sizer)

        self.select_box.Bind(ColoredRadioBox.EVT_COLORED_RADIO, self.on_radio_change)


    def on_radio_change(self, event):
        idx = event.button_id
        self.video_viewer.set_selected_bodypart(idx)

# class EditPointsControl()