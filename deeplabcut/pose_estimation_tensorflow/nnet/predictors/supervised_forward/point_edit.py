import wx
from deeplabcut.pose_estimation_tensorflow.nnet.processing import Pose
from deeplabcut.pose_estimation_tensorflow.nnet.predictors.supervised_forward.video_player import VideoPlayer
import cv2
import matplotlib.pyplot as plt

class PointViewNEdit(VideoPlayer):

    DEF_MAP = None

    def __init__(
        self,
        parent,
        video_hdl: cv2.VideoCapture,
        poses: Pose,
        colormap: str = DEF_MAP,
        plot_threshold: float = 0.1,
        point_radius: int = 5,
        w_id=wx.ID_ANY,
        pos=wx.DefaultPosition,
        size=wx.DefaultSize,
        style=wx.BORDER_DEFAULT,
        validator=wx.DefaultValidator,
        name="VideoPlayer"
    ):
        super().__init__(parent, w_id, video_hdl, pos, size, style, validator, name)
        self._poses = poses
        self._colormap = colormap
        self._plot_threshold = plot_threshold
        self._point_radius = point_radius


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