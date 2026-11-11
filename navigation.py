"""Mouse navigation shared by the two plots in the analysis tab."""

from __future__ import annotations


class PlotNavigation:
    """Use buttons 2/3 and the wheel, leaving the left button to measurement."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.map_ax = None
        self.profile_ax = None
        self.size_um = None
        self.pitch_um = None
        self.drag = None
        canvas.mpl_connect("button_press_event", self.press)
        canvas.mpl_connect("motion_notify_event", self.pan)
        canvas.mpl_connect("button_release_event", self.release)
        canvas.mpl_connect("scroll_event", self.scroll)

    def set_axes(self, map_ax, profile_ax, scan):
        self.map_ax, self.profile_ax = map_ax, profile_ax
        self.size_um = scan.size_um if scan is not None else None
        self.pitch_um = scan.pitch_um if scan is not None else None
        self.drag = None

    def handles(self, ax):
        return self.size_um is not None and ax in (self.map_ax, self.profile_ax)

    def press(self, event):
        ax = event.inaxes
        if self.handles(ax) and event.button in (2, 3):
            self.drag = ax, event.button, event.x, event.y, ax.get_xlim(), ax.get_ylim()

    def pan(self, event):
        if self.drag is None:
            return
        ax, _, x, y, xlim, ylim = self.drag
        dx = (event.x - x) * (xlim[1] - xlim[0]) / ax.bbox.width
        dy = (event.y - y) * (ylim[1] - ylim[0]) / ax.bbox.height
        self.set_view(ax, (xlim[0] - dx, xlim[1] - dx), (ylim[0] - dy, ylim[1] - dy))

    def release(self, event):
        if self.drag is not None and event.button == self.drag[1]:
            self.drag = None

    def scroll(self, event):
        ax = event.inaxes
        if not self.handles(ax) or event.xdata is None or event.ydata is None or not event.step:
            return
        factor = 1.2 ** -max(-8, min(8, event.step))

        def zoom(limits, center):
            return tuple(center + (edge - center) * factor for edge in limits)

        self.set_view(ax, zoom(ax.get_xlim(), event.xdata), zoom(ax.get_ylim(), event.ydata))

    def set_view(self, ax, xlim, ylim):
        if ax is self.map_ax:
            def within(limits, extent, pitch):
                span = min(extent, max(2 * pitch, limits[1] - limits[0]))
                start = max(0, min(extent - span, (sum(limits) - span) / 2))
                return start, start + span

            xlim = within(xlim, self.size_um[0], self.pitch_um[0])
            ylim = within(ylim, self.size_um[1], self.pitch_um[1])
        ax.set(xlim=xlim, ylim=ylim)
        self.canvas.draw_idle()
