from typing import Callable, Any, Optional

import wx
import cv2
import queue
import threading
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from collections import deque
import numpy as np

class ControlDeque:
    """
    A control deque. This deque is used to store video frames while they are coming in
    """
    def __init__(self, maxsize: int):
        """
        Create a new ControlDeque.

        :param maxsize: The maximum allowed size of the deque.
        """
        self._deque = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

        self._cancel_all_ops = threading.Event()

    def clear(self):
        """
        Clear the deque. This will also clear any push operations currently waiting.

        :return:
        """
        with self._lock:
            with self._cancel_all_ops:
                # Clear all push operations currently waiting
                self._cancel_all_ops.set()
                self._not_full.notify()
                # Clear the deque
                self._deque.clear()

    @property
    def maxsize(self):
        """
        Get the max size of this deque.

        :return: The max size of this deque.
        """
        return self._deque.maxlen

    def __len__(self):
        """
        Get the current length of the deque. NOT RELIABLE!

        :return: The current size of this deque.
        """
        with self._lock:
            return len(self._deque)

    def _do_relaxed_op(self, lock: threading.Condition, check: Callable[[], bool], action: Callable[[], Any],
                       notify: threading.Condition, push_mode: bool = False):
        """
        Internal method, perform a relaxed operation.

        :param lock: The lock to acquire during the process.
        :param check: The condition to check for. We continue to wait while the condition is true, a callable.
        :param action: The action to perform once the condition is achieved, another callable
        :param notify: The condition/lock to notify once the action is performed.
        :param push_mode: A boolean, set to true if this is a push operation. Used to determine if this event can be
                          canceled. (Push events can be canceled if clear is called)
        :return: Whatever is returned by action, if anything. Otherwise returns None.
        """
        with lock:
            if(self.maxsize >= 0):
                while(check() and ((not push_mode) or (not self._cancel_all_ops.is_set()))):
                    lock.wait()
                if(push_mode and self._cancel_all_ops.is_set()):
                    with self._cancel_all_ops:
                        self._cancel_all_ops.clear()
                        return
                result = action()
                notify.notify()
                return result


    def push_left_relaxed(self, value):
        """
        Push object onto left side of the deque. If there is no space available, we block until there is space.

        :param value: The value to push onto the stack.
        """
        check = lambda: len(self._deque) >= self.maxsize
        action = lambda: self._deque.appendleft(value)
        self._do_relaxed_op(self._not_full, check, action, self._not_empty, True)

    def push_right_relaxed(self, value):
        """
        Push object onto right side of the deque. If there is no space available, we block until there is space.

        :param value: The value to push onto the stack.
        """
        check = lambda: len(self._deque) >= self.maxsize
        action = lambda: self._deque.append(value)
        self._do_relaxed_op(self._not_full, check, action, self._not_empty, True)

    def pop_left_relaxed(self) -> Any:
        """
        Pop an object from the left side of the deque. If there are no items, we block until there is space.

        :return: The item on the left side of the stack.
        """
        check = lambda: len(self._deque) == 0
        action = lambda: self._deque.popleft()
        return self._do_relaxed_op(self._not_empty, check, action, self._not_full)

    def pop_right_relaxed(self) -> Any:
        """
        Pop an object from the right side of the deque. If there are no items, we block until there is space.

        :return: The item on the right side of the stack.
        """
        check = lambda: len(self._deque) == 0
        action = lambda: self._deque.pop()
        return self._do_relaxed_op(self._not_empty, check, action, self._not_full)

    def push_left_force(self, value) -> Any:
        """
        Push object onto left side of the deque. If there is no space available, we forcefully push the object onto
        the deque, causing the value on the other side of the deque to be deleted.

        :param value: The value to forcefully push onto the deque.
        """
        with self._lock:
            self._deque.appendleft(value)
            self._not_empty.notify()

    def push_right_force(self, value) -> Any:
        """
        Push object onto right side of the deque. If there is no space available, we forcefully push the object onto
        the deque, causing the value on the other side of the deque to be deleted.

        :param value: The value to forcefully push onto the deque.
        """
        with self._lock:
            self._deque.append(value)
            self._not_empty.notify()


def time_check(time_controller: Connection) -> Optional[int]:
    """
    Waits for a new time from the connection.

    :param time_controller: The connection being used for sending updated times.
    :return: An integer being a new time to move to, or None otherwise.
    """
    value = -1

    while(value < 0):
        value = time_controller.recv()

    return value

def video_loader(frame_queue: ControlDeque, time_loc: Connection):
    # Begin by waiting for the location in the video to be set.
    video_file = time_loc.recv()
    if(video_file is None):
        return
    video_hdl = cv2.VideoCapture(video_file)

    while(video_hdl.isOpened()):
        # If a new time was sent through the pipe, set our time to that. Sending none through the pipe stops this
        # process.
        if(time_loc.poll()):
            new_loc = time_check(time_loc)
            if(new_loc is None):
                video_hdl.release()
                return
            video_hdl.set(cv2.CAP_PROP_POS_MSEC, new_loc)

        valid_frame, frame = video_hdl.read()

        # If we are at the end of the video we pause execution waiting for a response from the user (to change the time)
        if(not valid_frame):
            new_loc = time_check(time_loc)
            if(new_loc is None):
                video_hdl.release()
                return
            video_hdl.set(cv2.CAP_PROP_POS_MSEC, new_loc)
        else:
            # Otherwise we push the frame we just read...
            frame_queue.push_right_relaxed(frame)


class VideoControl(wx.Control):

    # The number of frames to store in the forward and backward buffer.
    BUFFER_SIZE = 50
    BACK_LOAD_AMT = 20

    def __init__(self, parent, w_id=wx.ID_ANY, video_path: str = None, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.BORDER_DEFAULT, validator=wx.DefaultValidator, name="VideoControl"):
        super().__init__(parent, w_id, pos, size, style, validator, name)

        video_hdl = cv2.VideoCapture(video_path)
        self._width = video_hdl.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._height = video_hdl.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._fps = video_hdl.get(cv2.CAP_PROP_FPS)
        self._num_frames = video_hdl.get(cv2.CAP_PROP_FRAME_COUNT)
        video_hdl.release()
        del video_hdl

        # Useful indicator variables...
        self._playing = False
        self._current_frame = None

        self._front_queue = ControlDeque(self.BUFFER_SIZE)
        self._back_queue = deque(maxlen=self.BACK_LOAD_AMT)
        self._current_loc = 0

        # The queue which stores frames...
        self._frames_future = queue.Queue(maxsize=self.BUFFER_SIZE)
        self._frames_past = queue.Queue(maxsize=self.BUFFER_SIZE)

        size = wx.Size(self._width, self._height)
        self.SetMinSize(size)
        self.SetInitialSize(size)
        self.SetSize(size)

        self._core_timer = wx.Timer(self)

        # Create the video loader to start loading frames:
        receiver, self._sender = Pipe(False)
        self._video_loader = threading.Thread(target=video_loader, args=(self._front_queue, receiver))
        self._video_loader.start()
        self._sender.send(video_path)

        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None)


    def on_paint(self, event):
        painter = wx.BufferedPaintDC(self)

        self.on_draw(painter)


    def on_draw(self, dc: wx.BufferedPaintDC):

        width, height = self.GetClientSize()

        if((not width) or (not height)):
            return

        dc.SetBackground(wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID))
        dc.Clear()

        # Draw the video background
        b_h, b_w = self._current_frame.shape[:2]
        bitmap = wx.Bitmap.FromBuffer(b_w, b_h, self._current_frame[:, :, ::-1].astype(dtype=np.uint8))

        loc_x = (width - b_w) // 2
        loc_y = (height - b_h) // 2

        dc.DrawBitmap(bitmap, loc_x, loc_y)

    def on_timer(self, event):
        if(self._playing):
            if(self._current_loc >= (self._num_frames - 1)):
                self.pause()
                return
            self._current_frame = self._front_queue.pop_left_relaxed()
            self._current_loc += 1
            self.Refresh()
            self.Update() # Force a redraw....
            self._core_timer.StartOnce(1000 / self._fps)

    def play(self):
        self._playing = True
        self.on_timer(None)

    def stop(self):
        self._playing = False
        self.set_offset_frames(0)

    def pause(self):
        self._playing = False

    def is_playing(self):
        return self._playing

    def get_offset_millis(self):
        return self._frame_index_to_millis(self._current_loc)

    def get_offset_count(self):
        return self._current_loc

    def get_total_frames(self):
        return self._num_frames

    def set_offset_millis(self, value: int):
        self.set_offset_frames(int(value / (1000 / self._fps)))

    def set_offset_frames(self, value: int):
        # Determine how many more frames we can go back.
        go_back_frames = min(value, self.BACK_LOAD_AMT)
        self._current_loc = value

        self._sender.send(-1) # Tell the video loader to stop.

        self._front_queue.clear() # Completely wipe the queue
        # Tell the video loader to go to the new track location, and pop the extra frames we load on the back...
        self._sender.send(self._frame_index_to_millis(value - go_back_frames))
        for i in range(go_back_frames):
            self._back_queue.append(self._front_queue.pop_left_relaxed())

    def _frame_index_to_millis(self, frame_idx):
        return (1000 / self._fps) * frame_idx

    def __del__(self):
        self._sender.send(None)
        self._sender.close()

if(__name__ == "__main__"):
    vid_path = "/home/isaac/Code/MultiTrackTest7-IsaacRobinson-2020-03-04/videos/TestVideos/V2cut.mp4"

    app = wx.App()
    wid_frame = wx.Frame(None, title="Test...")
    wid = VideoControl(wid_frame, video_path=vid_path)
    # frame.AddChild(wid)
    wid_frame.Show(True)
    wid.play()
    app.MainLoop()