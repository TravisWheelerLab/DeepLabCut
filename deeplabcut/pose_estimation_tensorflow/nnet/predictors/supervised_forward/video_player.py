from typing import Callable, Any, Optional

import wx
from wx.lib.newevent import NewCommandEvent
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

        self._num_push = 0
        self._cancel_all_ops = threading.Event()

    def clear(self):
        """
        Clear the deque. This will also clear any push operations currently waiting.
        """
        with self._lock:
            # Clear all push operations currently waiting
            self._cancel_all_ops.set()
            self._not_full.notify()
            # Clear the deque
            self._deque.clear()

    def flush(self):
        """
        Flush any current push events, not actually adding there results to the deque.
        """
        with self._lock:
            # Set the cancel all ops method, and notify all currently waiting events...
            self._cancel_all_ops.set()
            self._not_full.notify()

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
        clear_before_enter = push_mode and self._cancel_all_ops.is_set()

        with lock:
            if(self.maxsize >= 0):
                self._num_push += 1
                while(check() and ((not push_mode) or (not self._cancel_all_ops.is_set()))):
                    lock.wait()
                if(push_mode and self._cancel_all_ops.is_set() and (not clear_before_enter)):
                    self._num_push -= 1
                    if(self._num_push == 0):
                        self._cancel_all_ops.clear()
                    lock.notify()
                    return
                result = action()
                notify.notify()
                self._num_push -= 1
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

    def push_left_force(self, value):
        """
        Push object onto left side of the deque. If there is no space available, we forcefully push the object onto
        the deque, causing the value on the other side of the deque to be deleted.

        :param value: The value to forcefully push onto the deque.
        """
        with self._lock:
            self._deque.appendleft(value)
            self._not_empty.notify()

    def push_right_force(self, value):
        """
        Push object onto right side of the deque. If there is no space available, we forcefully push the object onto
        the deque, causing the value on the other side of the deque to be deleted.

        :param value: The value to forcefully push onto the deque.
        """
        with self._lock:
            self._deque.append(value)
            self._not_empty.notify()

    def pop_left_force(self) -> Any:
        """
        Forcefully pop a value from the left side of the deque, throwing an exception if the deque is empty.

        :return: The value left most on the deque.
        """
        with self._lock:
            result = self._deque.popleft()
            self._not_full.notify()
            return result

    def pop_right_force(self) -> Any:
        """
        Forcefully pop a value from the right side of the deque, throwing an exception if the deque is empty.

        :return: The value right most on the deque.
        """
        with self._lock:
            result = self._deque.pop()
            self._not_full.notify()
            return result


def time_check(time_controller: Connection) -> Optional[int]:
    """
    Waits for a new time from the connection.

    :param time_controller: The connection being used for sending updated times.
    :return: An integer being a new time to move to, or None otherwise.
    """
    value = -1

    while((value is not None) and (value < 0)):
        value = time_controller.recv()

    return value

def video_loader(video_hdl: cv2.VideoCapture, frame_queue: ControlDeque, time_loc: Connection):
    """
    The core video loading function. Loads the video on a separate thread in the background for smooth performance.

    :param video_hdl: The cv2 VideoCapture object to read frames from.
    :param frame_queue: The ControlDeque to append frames to.
    :param time_loc: A multiprocessing Connection object, used to control this video loader. Sending a -1 through the
                    pipe pauses the loader, sending a positive integer sets the offset of this video loader to the
                    passed integer in milliseconds, and sending None closes this video loader and its associated thread.
    """
    # Begin by waiting for a simple message to give the go ahead to run
    video_file = time_loc.recv()
    if(video_file is None):
        return

    while(video_hdl.isOpened()):
        # If a new time was sent through the pipe, set our time to that. Sending none through the pipe stops this
        # process.
        if(time_loc.poll()):
            new_loc = time_check(time_loc)
            if(new_loc is None):
                return
            video_hdl.set(cv2.CAP_PROP_POS_MSEC, new_loc)

        valid_frame, frame = video_hdl.read()

        # If we are at the end of the video we pause execution waiting for a response from the user (to change the time)
        if(not valid_frame):
            new_loc = time_check(time_loc)
            if(new_loc is None):
                return
            video_hdl.set(cv2.CAP_PROP_POS_MSEC, new_loc)
        else:
            # Otherwise we push the frame we just read...
            frame_queue.push_right_relaxed(frame)


class VideoPlayer(wx.Control):
    """
    A video player for wx Widgets, Using cv2 for solid cross-platform video support. Can play video, but no audio.
    """

    # The number of frames to store in the forward and backward buffer.
    BUFFER_SIZE = 50
    BACK_LOAD_AMT = 20

    FrameChangeEvent, EVT_FRAME_CHANGE = NewCommandEvent()
    PlayStateChangeEvent, EVT_PLAY_STATE_CHANGE = NewCommandEvent()

    def __init__(self, parent, w_id=wx.ID_ANY, video_hdl: cv2.VideoCapture = None, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.BORDER_DEFAULT, validator=wx.DefaultValidator, name="VideoPlayer"):
        """
        Create a new VideoPlayer

        :param parent: The wx Control Parent.
        :param w_id: The wx ID.
        :param video_hdl: The cv2 VideoCapture to play video from. One should avoid never manipulate the video capture
                          once passed to this constructor, as the handle will be passed to another thread for fast
                          video loading.
        :param pos: The position of the widget.
        :param size: The size of the widget.
        :param style: The style of the widget.
        :param validator: The widgets validator.
        :param name: The name of the widget.
        """
        super().__init__(parent, w_id, pos, size, style, validator, name)

        self._width = video_hdl.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._height = video_hdl.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._fps = video_hdl.get(cv2.CAP_PROP_FPS)
        self._num_frames = video_hdl.get(cv2.CAP_PROP_FRAME_COUNT)

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
        self._video_loader = threading.Thread(target=video_loader, args=(video_hdl, self._front_queue, receiver))
        self._video_loader.start()
        self._sender.send(0)

        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None)

    @staticmethod
    def _resize_video(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Private method. Resizes the passed frame to optimally fit into the specified width and height, while maintaining
        aspect ratio.

        :param frame: The frame (cv2 image which is really a numpy array) to resize.
        :param width: The desired width of the resized frame.
        :param height: The desired height of the resized frame.
        :return: A new numpy array, being the resized version of the frame.
        """
        frame_aspect = frame.shape[0] / frame.shape[1]  # <-- Height / Width
        passed_aspect = height / width

        if(passed_aspect <= frame_aspect):
            # Passed aspect has less height per unit width, so height is the limiting dimension
            return cv2.resize(frame, (int(height / frame_aspect), height), interpolation=cv2.INTER_LINEAR)
        else:
            # Otherwise the width is the limiting dimension
            return cv2.resize(frame, (width, int(width * frame_aspect)), interpolation=cv2.INTER_LINEAR)

    def on_paint(self, event):
        """ Run on a paint event, redraws the widget. """
        painter = wx.BufferedPaintDC(self)
        self.on_draw(painter)

    def on_draw(self, dc: wx.BufferedPaintDC):
        """
        Draws the widget.

        :param dc: The wx DC to use for drawing.
        """
        width, height = self.GetClientSize()

        if((not width) or (not height)):
            return

        dc.SetBackground(wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID))
        dc.Clear()

        resized_frame = self._resize_video(self._current_frame, width, height)

        # Draw the video background
        b_h, b_w = resized_frame.shape[:2]
        bitmap = wx.Bitmap.FromBuffer(b_w, b_h, resized_frame[:, :, ::-1].astype(dtype=np.uint8))

        loc_x = (width - b_w) // 2
        loc_y = (height - b_h) // 2

        dc.DrawBitmap(bitmap, loc_x, loc_y)

    def _push_time_change_event(self):
        """ Private, used to specify how long the event should  """
        new_event = self.FrameChangeEvent(id=self.Id, frame=self.get_offset_count(), time=self.get_offset_millis())
        wx.PostEvent(self, new_event)

    def on_timer(self, event):
        if(self._playing):
            # If we have reached the end of the video, pause the video and don't perform a frame update as
            # we will deadlock the system by waiting for a frame forever...
            if(self._current_loc >= (self._num_frames - 1)):
                self.pause()
                return
            # Get the next frame and set it as the current frame
            self._current_frame = self._front_queue.pop_left_relaxed()
            self._current_loc += 1
            # Post a frame change event.
            self._push_time_change_event()
            # Trigger a redraw on the next pass through the loop and start the timer to play the next frame...
            self.Refresh()
            self.Update() # Force a redraw....
            self._core_timer.StartOnce(1000 / self._fps)

    def play(self):
        if(not self.is_playing()):
            self._playing = True
            wx.PostEvent(self, self.PlayStateChangeEvent(id=self.Id, playing=True, stop_triggered = False))
            self.on_timer(None)

    def stop(self):
        print("Stopping...")
        self._playing = False
        wx.PostEvent(self, self.PlayStateChangeEvent(id=self.Id, playing=False, stop_triggered = True))
        self.set_offset_frames(0)

    def pause(self):
        self._playing = False
        wx.PostEvent(self, self.PlayStateChangeEvent(id=self.Id, playing=False, stop_triggered = False))

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

    def _full_jump(self, value: int):
        current_state = self.is_playing()
        self._playing = False

        # Determine how many more frames we can go back.
        go_back_frames = min(value, self.BACK_LOAD_AMT)
        self._current_loc = value

        self._sender.send(-1) # Tell the video loader to stop.

        self._front_queue.clear() # Completely wipe the queue
        # Tell the video loader to go to the new track location, and pop the extra frames we load on the back...
        self._sender.send(self._frame_index_to_millis(value - go_back_frames))
        for i in range(go_back_frames):
            self._current_frame = self._front_queue.pop_left_relaxed()
            self._back_queue.append(self._current_frame)

        self._push_time_change_event()
        # Restore play state prior to frame change...
        self._playing = current_state
        self._core_timer.StartOnce(1000 / self._fps)


    def _fast_back(self, amount: int):
        current_state = self.is_playing()
        self._playing = False

        # Pause the video player, flush any push events it is trying to do.
        self._sender.send(-1)
        self._front_queue.flush()

        # Move back the passed amount of frames.
        for i in range(amount):
            self._current_frame = self._back_queue.pop()
            self._front_queue.push_left_force(self._current_frame)
        self._current_loc = self._current_loc - amount

        # Move the video play to how far ahead it would be after moving back this many frames...
        self._sender.send(self._frame_index_to_millis(max(self._num_frames - 1, self._current_loc + self.BUFFER_SIZE)))

        self._push_time_change_event()

        self._playing = current_state
        self._core_timer.StartOnce(1000 / self._fps)

    def _fast_forward(self, amount: int):
        current_state = self.is_playing()
        self._playing = False

        # Move the passed amount of frames forward. Video reader will automatically move forward with us...
        for i in range(amount):
            self._back_queue.append(self._front_queue.pop_left_force())
        self._current_loc = self._current_loc + amount

        self._push_time_change_event()

        self._playing = current_state
        self._core_timer.StartOnce(1000 / self._fps)

    def move_back(self, amount: int = 1):
        # Check if movement is valid...
        if(amount <= 0):
            raise ValueError("Offset must be positive!")
        if(self._current_loc - amount < 0):
            raise ValueError(f"Can't go back {amount} frames when at frame {self._current_loc}.")
        # Check if we can perform a 'fast' backtrack, where we have all of the frames in the queue. If not perform
        # a more computationally expensive full jump.
        if(amount > len(self._back_queue)):
            self._full_jump(self._current_loc - amount)
        else:
            self._fast_back(amount)

    def move_forward(self, amount: int = 1):
        # Check if movement is valid...
        if(amount <= 0):
            raise ValueError("Offset must be positive!")
        if(self._current_loc + amount >= self._num_frames):
            raise ValueError(f"Can't go forward {amount} frames when at frame {self._current_loc}.")
        # Check if we can do a fast forward, which is basically the same as moving through frames normally...
        # Otherwise we perform a more expensive full jump.
        if(amount > len(self._front_queue)):
            self._full_jump(self._current_loc + amount)
        else:
            self._fast_forward(amount)


    def set_offset_frames(self, value: int):
        # Is this a valid frame value?
        if(not (0 <= value < self._num_frames)):
            raise ValueError(f"Can't set frame index to {value}, there is only {self._num_frames} frames.")
        # Determine which way the value is moving the current video location, and move backward/forward based on that.
        if(value < self._current_loc):
            self.move_back(self._current_loc - value)
        elif(value > self._current_loc):
            self.move_forward(value - self._current_loc)

    def _frame_index_to_millis(self, frame_idx):
        return (1000 / self._fps) * frame_idx

    def __del__(self):
        self._sender.send(None)
        self._front_queue.clear()
        self._sender.close()


class VideoController(wx.Panel):

    PLAY_SYMBOL = "\u25B6"
    PAUSE_SYMBOL = "\u23F8"
    STOP_SYMBOL = "\u23F9"
    FRAME_BACK_SYMBOL = "\u21b6"
    FRAME_FORWARD_SYMBOL = "\u21b7"

    def __init__(self,parent, w_id = wx.ID_ANY, video_player: VideoPlayer = None, pos = wx.DefaultPosition,
                 size = wx.DefaultSize, style = wx.TAB_TRAVERSAL, name = "VideoController"):
        super().__init__(parent, w_id, pos, size, style, name)

        if(video_player is None):
            raise ValueError("Have to pass a VideoPlayer!!!")

        self._video_player = video_player

        self._sizer = wx.BoxSizer(wx.HORIZONTAL)

        self._back_btn = wx.Button(self, label=self.FRAME_BACK_SYMBOL)
        self._play_pause_btn = wx.Button(self, label=self.PLAY_SYMBOL)
        self._stop_btn = wx.Button(self, label=self.STOP_SYMBOL)
        self._forward_btn = wx.Button(self, label=self.FRAME_FORWARD_SYMBOL)

        self._slider_control = wx.Slider(self, value=0, minValue=0, maxValue=video_player.get_total_frames() - 1,
                                         style=wx.SL_HORIZONTAL | wx.SL_LABELS)

        self._sizer.Add(self._back_btn, 0, wx.ALIGN_CENTER)
        self._sizer.Add(self._play_pause_btn, 0, wx.ALIGN_CENTER)
        self._sizer.Add(self._stop_btn, 0, wx.ALIGN_CENTER)
        self._sizer.Add(self._forward_btn, 0, wx.ALIGN_CENTER)
        self._sizer.Add(self._slider_control, 1, wx.EXPAND)

        self._sizer.SetSizeHints(self)
        self.SetSizer(self._sizer)

        self._video_player.Bind(VideoPlayer.EVT_FRAME_CHANGE, self.frame_change)
        self._video_player.Bind(VideoPlayer.EVT_PLAY_STATE_CHANGE, self.on_play_switch)
        self._slider_control.Bind(wx.EVT_SLIDER, self.on_slide)
        self._play_pause_btn.Bind(wx.EVT_BUTTON, self.on_play_pause_press)
        self._back_btn.Bind(wx.EVT_BUTTON, self.on_back_press)
        self._forward_btn.Bind(wx.EVT_BUTTON, self.on_forward_press)
        self._stop_btn.Bind(wx.EVT_BUTTON, lambda evt: self._video_player.stop())

    def frame_change(self, event):
        print(f"Frame Change to: {event.frame}")
        frame, time = event.frame, event.time
        self._slider_control.SetValue(frame)
        self._back_btn.Enable(frame > 0)
        self._forward_btn.Enable(frame < (self._video_player.get_total_frames() - 1))

    def on_play_switch(self, event):
        self._play_pause_btn.SetLabel(self.PAUSE_SYMBOL if(event.playing) else self.PLAY_SYMBOL)

    def on_slide(self, event):
        self._video_player.set_offset_frames(self._slider_control.GetValue())

    def on_play_pause_press(self, event):
        if(self._video_player.is_playing()):
            self._video_player.pause()
        else:
            self._video_player.play()

    def on_back_press(self, event):
        if(self._video_player.get_offset_count() > 0):
            self._video_player.move_back()

    def on_forward_press(self, event):
        if(self._video_player.get_offset_count() < (self._video_player.get_total_frames() - 1)):
            self._video_player.move_forward()


if(__name__ == "__main__"):
    vid_path = "/home/isaac/Code/MultiTrackTest7-IsaacRobinson-2020-03-04/videos/TestVideos/V2cut.mp4"

    app = wx.App()
    wid_frame = wx.Frame(None, title="Test...")
    panel = wx.Panel(parent=wid_frame)

    sizer = wx.BoxSizer(wx.VERTICAL)

    wid = VideoPlayer(panel, video_hdl=cv2.VideoCapture(vid_path))
    obj2 = VideoController(panel, video_player=wid)

    sizer.Add(wid, 0, wx.EXPAND)
    sizer.Add(obj2, 0, wx.EXPAND)

    sizer.SetSizeHints(panel)
    panel.SetSizer(sizer)

    wid_frame.Fit()

    wid_frame.Show(True)


    def destroy(evt):
        global wid
        del wid
        wid_frame.Destroy()

    wid_frame.Bind(wx.EVT_CLOSE, destroy)
    wid.play()
    app.MainLoop()