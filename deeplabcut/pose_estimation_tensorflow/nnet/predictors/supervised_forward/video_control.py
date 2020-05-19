import wx
import cv2
import queue
import threading

class VideoControl(wx.Control):

    # The number of frames to store in the forward and backward buffer.
    BUFFER_SIZE = 50

    def __init__(self, parent, w_id=wx.ID_ANY, video_hdl: cv2.VideoCapture = None, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.BORDER_DEFAULT, validator=wx.DefaultValidator, name="VideoControl"):
        super().__init__(parent, w_id, pos, size, style, validator, name)

        self._video_hdl = video_hdl
        self._width = video_hdl.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._height = video_hdl.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._fps = video_hdl.get(cv2.CAP_PROP_FPS)
        self._num_frames = video_hdl.get(cv2.CAP_PROP_FRAME_COUNT)

        # Useful indicator variables...
        self._playing = False

        # The queue which stores frames...
        self._frames_future = queue.Queue(maxsize=self.BUFFER_SIZE)
        self._frames_past = queue.Queue(maxsize=self.BUFFER_SIZE)

        size = wx.Size(self._width, self._height)
        self.SetMinSize(size)
        self.SetInitialSize(size)

        self._core_timer = wx.Timer(self)

        self.Bind(wx.EVT_TIMER, self.on_timer)


    def on_timer(self, event):
        print("timer")

    def load_frames(self, end_time_frames: int):


    def play(self):
        pass

    def stop(self):
        pass

    def pause(self):
        pass

    def is_playing(self):
        pass

    def offset(self):
        pass

    def _frame_index_to_millis(self, frame_idx):
        return (1000 / self._fps) * frame_idx


