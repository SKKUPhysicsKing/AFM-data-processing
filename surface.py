"""Interactive 3D display of a calibrated AFM raster."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, degrees, radians, sin, sqrt
from time import monotonic

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from afm import Scan

PREVIEW_SAMPLES = 29
DETAIL_SAMPLES = 100
FRAME_INTERVAL = 1 / 30


@dataclass(frozen=True)
class Camera:
    azimuth: float = -60.0
    elevation: float = 30.0
    zoom: float = 1.0


def surface_axes(figure: Figure, scan: Scan, values: np.ndarray, level: str):
    """Set calibrated axes and color limits before adding a display mesh."""
    ax = figure.add_subplot(projection="3d")
    ax.disable_mouse_rotation()
    low, high = float(values.min()), float(values.max())
    if low == high:
        padding = max(abs(low) * 1e-5, 1e-6)
        low, high = low - padding, high + padding
    width, height = scan.size_um
    ax.set(xlim=(0, width), ylim=(0, height), zlim=(low, high),
           xlabel="Scan X (µm)", ylabel="Scan Y (µm)",
           zlabel=f"{'Height' if scan.is_height else scan.channel} ({scan.unit})",
           title=f"{scan.channel} · {scan.direction}\n{level}")
    ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    return ax, (width, height, max(width, height) * 0.55), Normalize(low, high)


def add_mesh(ax, scan: Scan, values: np.ndarray, norm: Normalize, samples: int):
    x, y = scan.centers_um
    i = np.linspace(0, x.size - 1, min(x.size, samples), dtype=int)
    j = np.linspace(0, y.size - 1, min(y.size, samples), dtype=int)
    xx, yy = np.meshgrid(x[i], y[j])
    return ax.plot_surface(xx, yy, values[np.ix_(j, i)], cmap="afmhot",
                           norm=norm, rstride=1, cstride=1, linewidth=0, antialiased=False)


def apply_camera(ax, aspect, camera: Camera):
    ax.view_init(elev=camera.elevation, azim=camera.azimuth)
    ax.set_box_aspect(aspect, zoom=camera.zoom)


def save_surface(path, scan: Scan, values: np.ndarray, level: str, camera: Camera):
    """Render the current view with a detailed mesh, independent of preview LOD."""
    fig = Figure(figsize=(9, 7), layout="constrained")
    ax, aspect, norm = surface_axes(fig, scan, values, level)
    mesh = add_mesh(ax, scan, values, norm, samples=150)
    fig.colorbar(mesh, ax=ax, label=scan.unit, shrink=0.72, pad=0.09)
    apply_camera(ax, aspect, camera)
    fig.savefig(path, dpi=180)


def orbit_delta(dx, dy, size, camera: Camera):
    """Move the surface with the cursor, using screen-right and screen-up axes."""
    # At steep elevations, a given yaw projects less horizontal displacement.
    # Cap the gain near the pole so a small move cannot produce a large spin.
    tilt = radians(camera.elevation)
    sensitivity = 165 / max(size, 1)
    azimuth = camera.azimuth - dx * sensitivity / max(0.55, abs(cos(tilt)))
    elevation = camera.elevation - dy * sensitivity
    return Camera((azimuth + 180) % 360 - 180,
                  max(-85.0, min(85.0, elevation)), camera.zoom)


class SurfaceView:
    """Draw an inexpensive preview while the camera moves."""

    def __init__(self, figure: Figure, camera_changed):
        self.figure = figure
        self.camera_changed = camera_changed
        self.ax = None
        self.scan = None
        self.drag = None
        self.camera = Camera()
        self.last_frame = 0.0
        self.finish_timer = None
        self.in_preview = False
        canvas = figure.canvas
        canvas.mpl_connect("button_press_event", self.press)
        canvas.mpl_connect("motion_notify_event", self.rotate)
        canvas.mpl_connect("button_release_event", self.release)
        canvas.mpl_connect("scroll_event", self.scroll)

    def show(self, scan: Scan | None, values: np.ndarray | None, level: str):
        if self.finish_timer is not None:
            self.finish_timer.stop()
            self.finish_timer = None
        self.figure.set_layout_engine("constrained")
        self.figure.clear()
        self.scan = scan
        self.drag = None
        self.in_preview = False
        if scan is None or min(scan.values.shape) < 2:
            self.ax = None
            message = ("Open a scan to view its 3D surface" if scan is None
                       else "A 3D surface needs at least two rows and two columns.")
            self.figure.text(0.5, 0.5, message, ha="center", va="center", color="0.45")
            self.figure.canvas.draw_idle()
            return
        self.ax, self.aspect, norm = surface_axes(self.figure, scan, values, level)
        self.detail = add_mesh(self.ax, scan, values, norm, DETAIL_SAMPLES)
        self.preview = add_mesh(self.ax, scan, values, norm, PREVIEW_SAMPLES)
        self.preview.set_visible(False)
        self.colorbar = self.figure.colorbar(self.detail, ax=self.ax, label=scan.unit, shrink=0.72, pad=0.09)
        apply_camera(self.ax, self.aspect, self.camera)
        self.draw_camera()

    def draw_camera(self):
        self.last_frame = monotonic()
        self.camera_changed(self.camera.azimuth, self.camera.elevation, self.camera.zoom)
        self.figure.canvas.draw_idle()

    def preview_mode(self, active: bool):
        if active == self.in_preview:
            return
        self.in_preview = active
        self.preview.set_visible(active)
        self.detail.set_visible(not active)
        # Layout and 3D tick projection are much more expensive than the
        # preview mesh itself. Restore both as soon as interaction ends.
        self.figure.set_layout_engine("none" if active else "constrained")
        if active:
            self.ax.set_axis_off()
            self.colorbar.ax.set_visible(False)
        else:
            self.ax.set_axis_on()
            self.colorbar.ax.set_visible(True)

    def reset(self):
        if self.ax is None:
            return
        self.camera = Camera()
        self.drag = None
        self.finish_interaction()

    def press(self, event):
        if self.ax is None or event.inaxes is not self.ax or event.button != 1:
            return
        if self.finish_timer is not None:
            self.finish_timer.stop()
            self.finish_timer = None
        self.drag = event.x, event.y, self.camera

    def rotate(self, event):
        if self.drag is None or event.x is None or event.y is None:
            return
        x, y, initial = self.drag
        if (event.x, event.y) == (x, y) and self.camera == initial:
            return
        size = min(self.ax.bbox.width, self.ax.bbox.height)
        self.camera = orbit_delta(event.x - x, event.y - y, size, initial)
        if monotonic() - self.last_frame >= FRAME_INTERVAL or not self.in_preview:
            self.preview_mode(True)
            self.ax.view_init(elev=self.camera.elevation, azim=self.camera.azimuth)
            self.draw_camera()

    def release(self, event):
        if event.button == 1 and self.drag is not None:
            self.rotate(event)
            self.drag = None
            self.finish_interaction()

    def scroll(self, event):
        if self.ax is None or event.inaxes is not self.ax or not event.step or self.drag is not None:
            return
        zoom = self.camera.zoom * 1.13 ** max(-8, min(8, event.step))
        self.camera = Camera(self.camera.azimuth, self.camera.elevation, max(0.45, min(1.85, zoom)))
        self.preview_mode(True)
        self.ax.set_box_aspect(self.aspect, zoom=self.camera.zoom)
        self.draw_camera()
        if self.finish_timer is None:
            self.finish_timer = self.figure.canvas.new_timer(interval=160)
            self.finish_timer.single_shot = True
            self.finish_timer.add_callback(self.finish_interaction)
        self.finish_timer.stop()
        self.finish_timer.start()

    def finish_interaction(self):
        if self.finish_timer is not None:
            self.finish_timer.stop()
            self.finish_timer = None
        if self.ax is not None:
            self.preview_mode(False)
            apply_camera(self.ax, self.aspect, self.camera)
            self.draw_camera()
