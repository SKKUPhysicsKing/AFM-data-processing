"""Interactive 3D display of a calibrated AFM raster."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from afm import Scan


class SurfaceView:
    """Orbit using elevation and azimuth while keeping data coordinates intact."""

    def __init__(self, figure: Figure, camera_changed):
        self.figure = figure
        self.camera_changed = camera_changed
        self.ax = None
        self.scan = None
        self.aspect = (1, 1, 0.55)
        self.drag = None
        self.elevation = 30.0
        self.azimuth = -60.0
        self.zoom = 1.0
        canvas = figure.canvas
        canvas.mpl_connect("button_press_event", self.press)
        canvas.mpl_connect("motion_notify_event", self.rotate)
        canvas.mpl_connect("button_release_event", self.release)
        canvas.mpl_connect("scroll_event", self.scroll)

    def show(self, scan: Scan | None, values: np.ndarray | None, level: str):
        self.figure.clear()
        self.scan = scan
        self.drag = None
        if scan is None or min(scan.values.shape) < 2:
            self.ax = None
            message = ("Open a scan to view its 3D surface" if scan is None
                       else "A 3D surface needs at least two rows and two columns.")
            self.figure.text(0.5, 0.5, message,
                             ha="center", va="center", color="0.45")
            self.figure.canvas.draw_idle()
            return

        self.ax = ax = self.figure.add_subplot(projection="3d")
        ax.disable_mouse_rotation()  # Use the explicit azimuth/elevation drag below.
        x, y = scan.centers_um
        # Draw a bounded mesh for smooth rotation, retaining the first and last samples.
        i = np.linspace(0, x.size - 1, min(x.size, 72), dtype=int)
        j = np.linspace(0, y.size - 1, min(y.size, 72), dtype=int)
        xx, yy = np.meshgrid(x[i], y[j])
        low, high = float(values.min()), float(values.max())
        if low == high:
            padding = max(abs(low) * 1e-5, 1e-6)
            low, high = low - padding, high + padding
        raster = ax.plot_surface(xx, yy, values[np.ix_(j, i)], cmap="afmhot",
                                 vmin=low, vmax=high, rstride=1, cstride=1,
                                 linewidth=0, antialiased=False)
        sx, sy = scan.size_um
        ax.set(xlim=(0, sx), ylim=(0, sy), zlim=(low, high),
               xlabel="Scan X (µm)", ylabel="Scan Y (µm)",
               zlabel=f"{'Height' if scan.is_height else scan.channel} ({scan.unit})",
               title=f"{scan.channel} · {scan.direction}  |  {level}")
        ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        self.figure.colorbar(raster, ax=ax, label=scan.unit, shrink=0.72, pad=0.09)
        self.aspect = (sx, sy, max(sx, sy) * 0.55)
        self.update_camera()

    def update_camera(self):
        if self.ax is None:
            return
        self.ax.view_init(elev=self.elevation, azim=self.azimuth)
        self.ax.set_box_aspect(self.aspect, zoom=self.zoom)
        self.camera_changed(self.azimuth, self.elevation, self.zoom)
        self.figure.canvas.draw_idle()

    def reset(self):
        self.elevation = 30.0
        self.azimuth = -60.0
        self.zoom = 1.0
        self.drag = None
        self.update_camera()

    def press(self, event):
        if self.ax is not None and event.inaxes is self.ax and event.button == 1:
            self.drag = event.x, event.y

    def rotate(self, event):
        if self.drag is None:
            return
        x, y = self.drag
        self.drag = event.x, event.y
        self.azimuth = (self.azimuth - 0.45 * (event.x - x) + 180) % 360 - 180
        self.elevation = float(np.clip(self.elevation + 0.45 * (event.y - y), -90, 90))
        self.update_camera()

    def release(self, event):
        if event.button == 1:
            self.drag = None

    def scroll(self, event):
        if self.ax is not None and event.inaxes is self.ax and event.step:
            self.zoom = float(np.clip(self.zoom * 1.13 ** np.clip(event.step, -8, 8), 0.45, 1.85))
            self.update_camera()
