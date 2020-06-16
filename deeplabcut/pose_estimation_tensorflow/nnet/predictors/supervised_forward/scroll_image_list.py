from typing import List, Optional

import wx


class ScrollImageList(wx.ScrolledCanvas):

    SCROLL_RATE = 20

    def __init__(self, parent, img_list: Optional[List[wx.Bitmap]], orientation = wx.VERTICAL, padding = 20,
                 id=wx.ID_ANY, pos=wx.DefaultPosition, size=wx.DefaultSize, style=wx.HSCROLL | wx.VSCROLL,
                 name="ScrollImageList"):
        super().__init__(parent, id, pos, size, style, name)
        if(img_list is None):
            img_list = []

        self._bitmaps = []
        self._mode = wx.VERTICAL
        self._padding = 20
        self._dims = None

        self.set_bitmaps(img_list)
        self.set_orientation(orientation)
        self.set_padding(padding)

        if(size == wx.DefaultSize):
            self.SetInitialSize(wx.Size(*self._dims))
        else:
            self.SetInitialSize(size)
        self.SetSize(wx.Size(*self._dims))
        self.EnableScrolling(True, True)
        self.ShowScrollbars(True, True)

    def SetScrollPageSize(self, orient, pageSize):
        super(ScrollImageList, self).SetScrollPageSize()

    def _compute_dimensions(self):
        if(self._dims is not None):
            return self._dims

        if(len(self._bitmaps) == 0):
            width, height =  100, 100
        elif(self._mode == wx.VERTICAL):
            width = max((bitmap.GetWidth() for bitmap in self._bitmaps))
            height = sum(bitmap.GetHeight() for bitmap in self._bitmaps) + self._padding * len(self._bitmaps)
        else:
            width = sum(bitmap.GetWidth() for bitmap in self._bitmaps) + self._padding * len(self._bitmaps)
            height = max(bitmap.GetHeight() for bitmap in self._bitmaps)

        self._dims = width, height
        self.SetVirtualSize(width, height)
        self.SetScrollRate(self.SCROLL_RATE, self.SCROLL_RATE)
        self.AdjustScrollbars()
        self.SendSizeEvent()
        self.Refresh()
        return width, height

    def OnDraw(self, dc: wx.DC):
        width, height = self.GetClientSize()

        if((not width) or (not height)):
            return

        # dc.SetBackground(wx.Brush(self.GetBackgroundColour(), wx.BRUSHSTYLE_SOLID))
        # dc.Clear()

        offset = 0

        if(self._mode == wx.VERTICAL):
            for bitmap in self._bitmaps:
                dc.DrawBitmap(bitmap, 0, offset)
                offset += bitmap.GetHeight() + self._padding
        else:
            for bitmap in self._bitmaps:
                dc.DrawBitmap(bitmap, offset, 0)
                offset += bitmap.GetWidth() + self._padding

    def get_padding(self) -> int:
        return self._padding

    def set_padding(self, value: int):
        self._padding = int(value)
        self._dims = None
        self._compute_dimensions()

    def get_orientation(self) -> int:
        return self._mode

    def set_orientation(self, value: int):
        if((value != wx.VERTICAL) and (value != wx.HORIZONTAL)):
            raise ValueError("Orientation must be wx.VERTICAL or wx.HORIZONTAL!!!")
        self._mode = value
        self._dims = None
        self._compute_dimensions()

    def get_bitmaps(self) -> List[wx.Bitmap]:
        return self._bitmaps

    def set_bitmaps(self, bitmaps: List[wx.Bitmap]):
        if(bitmaps is None):
            bitmaps = []
        self._bitmaps = bitmaps
        self._dims = None
        self._compute_dimensions()