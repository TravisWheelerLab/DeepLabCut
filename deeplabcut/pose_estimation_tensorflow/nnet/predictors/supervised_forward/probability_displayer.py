from typing import Tuple

import wx
import numpy as np

class ProbabilityDisplayer(wx.Control):
    """
    A probability displayer...
    """
    MIN_PROB_STEP = 10
    VISIBLE_PROBS = 100
    DEF_HEIGHT = 50
    TRIANGLE_SIZE = 7
    TOP_PADDING = 3

    def __init__(self, parent, data: np.ndarray = None, text: str = None, w_id=wx.ID_ANY, height: int = DEF_HEIGHT,
                 visible_probs: int = VISIBLE_PROBS, pos=wx.DefaultPosition, size=wx.DefaultSize,
                 style=wx.BORDER_DEFAULT, validator=wx.DefaultValidator, name="ProbabilityDisplayer"):
        super().__init__(parent, w_id, pos, size, style, validator, name)

        if((len(data.shape) != 1)):
            raise ValueError("Invalid data! Must be a numpy array of 1 dimension...")

        self._data = data / np.max(data)
        self._ticks_visible = visible_probs

        size = wx.Size(self.MIN_PROB_STEP * 5, max(height, (self.TRIANGLE_SIZE * 4) + self.TOP_PADDING))
        self.SetMinSize(size)
        self.SetInitialSize(size)

        self._current_index = 0

        self._text = text

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda evt: None)

    def on_paint(self, event):
        painter = wx.BufferedPaintDC(self)
        # Using a GCDC allows for much prettier aliased painting, making plot look nicer.
        painter = wx.GCDC(painter)
        self.on_draw(painter)

    def _compute_points(self, height: int, width: int, data: np.ndarray) -> Tuple[int, int, np.ndarray]:
        tick_step = max(self.MIN_PROB_STEP, int(width / self._ticks_visible))

        center = (width // 2)
        values_per_side = (center - 1) // tick_step

        low_val = max(self._current_index - values_per_side, 0)
        high_val = min(self._current_index + values_per_side + 1, len(data))

        offset = center - ((self._current_index - low_val) * tick_step)

        x = np.arange(0, high_val - low_val) * tick_step + offset
        y = data[low_val:high_val]
        y = (1 - y) * (height - ((self.TRIANGLE_SIZE * 2) + self.TOP_PADDING)) + self.TOP_PADDING

        final_arr = np.zeros((len(x), 2), dtype=np.int32)
        final_arr[:, 0] = x
        final_arr[:, 1] = y

        return (center, self._current_index - low_val, final_arr)

    def on_draw(self, dc: wx.DC):
        width, height = self.GetClientSize()

        if((not width) or (not height)):
            return

        # Clear the background with the default color...
        dc.SetBackground(wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID))
        dc.Clear()

        # Colors used in pens and brushes below...
        fg_color = self.GetForegroundColour()
        fg_color2 = wx.Colour(fg_color.Red(), fg_color.Green(), fg_color.Blue(), fg_color.Alpha() * 0.3)
        red = wx.Colour(255, 0, 0, 200)

        # All of the pens and brushes we will need...
        foreground_pen = wx.Pen(fg_color, 2, wx.PENSTYLE_SOLID)
        transparent_pen = wx.Pen(fg_color, 2, wx.PENSTYLE_TRANSPARENT)
        foreground_pen2 = wx.Pen(fg_color, 5, wx.PENSTYLE_SOLID)
        foreground_brush = wx.Brush(fg_color2, wx.BRUSHSTYLE_SOLID)
        foreground_brush2 = wx.Brush(fg_color, wx.BRUSHSTYLE_SOLID)
        red_pen = wx.Pen(red, 5, wx.PENSTYLE_SOLID)
        red_pen2 = wx.Pen(red, 1, wx.PENSTYLE_SOLID)

        # Compute the center and points to place on the line...
        center, current_idx, points = self._compute_points(height, width, self._data)

        # Plot all of the points the filled-in polygon underneath, and the line connecting the points...
        wrap_polygon_points = np.array([[points[-1, 0], height], [points[0, 0], height]])
        dc.DrawPolygonList([np.concatenate((points, wrap_polygon_points))], transparent_pen, foreground_brush)
        dc.DrawLineList(np.transpose([points[:-1, 0], points[:-1, 1], points[1:, 0], points[1:, 1]]), foreground_pen)
        dc.DrawPointList(points, foreground_pen2)

        # Draw the current location indicating line, point and arrow, indicates which data point we are currently on.
        dc.DrawLineList([[center, 0, center, height]], red_pen2)
        dc.DrawPolygonList([[[center - self.TRIANGLE_SIZE, height], [center + self.TRIANGLE_SIZE, height],
                        [center, height - int(self.TRIANGLE_SIZE * 1.5)]]], foreground_pen, foreground_brush2)
        dc.DrawPointList([points[current_idx]], red_pen)

        if(self._text is not None):
            back_pen = wx.Pen(self.GetBackgroundColour(), 3, wx.PENSTYLE_SOLID)
            back_brush = wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID)
            dc.SetTextBackground(self.GetBackgroundColour())
            dc.SetTextForeground(self.GetForegroundColour())

            dc.SetFont(self.GetFont())
            size: wx.Size = dc.GetTextExtent(self._text)
            width, height = size.GetWidth(), size.GetHeight()
            dc.DrawRectangleList([(0, 0, width, height)], back_pen, back_brush)
            dc.DrawText(self._text, 0, 0)

    def set_location(self, location: int):
        if(not (0 <= location < self._data.shape[0])):
            raise ValueError(f"Location {location} is not within the range: 0 through {self._data.shape[0]}.")
        self._current_index = location
        self.Refresh()

    def get_location(self) -> int:
        return self._current_index

    def set_data(self, data: np.ndarray):
        self._data[:] = data / np.max(data)
        self.Refresh()

    def get_text(self) -> str:
        return self._text

    def set_text(self, value: str):
        self._text = value
