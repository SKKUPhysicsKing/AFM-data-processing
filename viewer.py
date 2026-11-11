"""Desktop viewer for Park Systems AFM scan files."""

from __future__ import annotations

import os
import tempfile
import tkinter as tk
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

from afm import Scan, line_profile, load_inputs, pair_metrics, roughness, scan_pairs, step_height
from batch import describe, draw_map, save_json, save_map, save_pixels
from corrections import BACKGROUNDS, ROW_FLATTEN, ZERO_REFERENCES, Corrections
from navigation import PlotNavigation
from profiles import PROFILE_MODES, axis_endpoints, axis_profile, near_line, translated_line
from surface import SurfaceView


class Viewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.scans: list[Scan] = []
        self.scan: Scan | None = None
        self.pair: tuple[Scan, Scan] | None = None
        self.regions = {}
        self.points = []
        self.tool = None
        self.selector = None
        self.profiles = []
        self.profile_axis = None
        self.profile_drag = None
        self.correction_cache = {}
        self.correction_key = None
        self.view_scan = None
        root.title("AFM XE7 ANALYZER")
        root.geometry("1200x780")
        root.minsize(850, 620)

        panel = ttk.Frame(root, padding=(12, 10))
        panel.pack(side="top", fill="x")
        ttk.Button(panel, text="Open ZIP / TIFF", command=self.open_files).pack(side="left")
        ttk.Button(panel, text="Open folder", command=self.open_folder).pack(side="left", padx=5)
        self.channel = ttk.Combobox(panel, state="readonly", width=44)
        self.channel.pack(side="left", fill="x", expand=True, padx=6)
        self.channel.bind("<<ComboboxSelected>>", self.select_scan)
        revision = os.environ.get("AFM_XE7_REVISION", "")
        if revision:
            ttk.Label(panel, text=f"Build {revision[:7]}").pack(side="right", padx=4)
        self.setup_corrections()
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.analysis_tab = ttk.Frame(self.tabs)
        self.surface_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.analysis_tab, text="2D Analysis")
        self.tabs.add(self.surface_tab, text="3D Surface")

        tools = ttk.Frame(self.analysis_tab, padding=(12, 8))
        tools.pack(side="top", fill="x")
        ttk.Label(tools, text="Line profile").pack(side="left", padx=(0, 5))
        self.profile_mode = ttk.Combobox(tools, values=tuple(PROFILE_MODES), state="readonly", width=15)
        self.profile_mode.current(0)
        self.profile_mode.pack(side="left", padx=(0, 5))
        self.profile_mode.bind("<<ComboboxSelected>>", self.change_profile_mode)
        ttk.Button(tools, text="Place line", command=lambda: self.start_tool("line")).pack(side="left", padx=(0, 8))
        self.position_label = tk.StringVar(value="Position (µm)")
        ttk.Label(tools, textvariable=self.position_label).pack(side="left", padx=(0, 4))
        self.position = ttk.Spinbox(tools, width=9, state="disabled", command=self.set_profile_position)
        self.position.pack(side="left", padx=(0, 8))
        self.position.bind("<Return>", self.set_profile_position)
        region_tools = ttk.Frame(self.analysis_tab, padding=(12, 0, 12, 6))
        region_tools.pack(fill="x")
        for label, mode in (("Region A", "A"), ("Region B", "B")):
            ttk.Button(region_tools, text=label, command=lambda m=mode: self.start_tool(m)).pack(side="left", padx=(0, 5))
        ttk.Button(region_tools, text="Clear", command=self.clear_selection).pack(side="left", padx=(0, 5))
        ttk.Button(region_tools, text="Fit map", command=self.fit_map).pack(side="left")
        ttk.Button(region_tools, text="Export", command=self.export).pack(side="right")
        ttk.Label(self.analysis_tab, text="Left drag a profile line to move it · Right/middle drag to pan · Scroll to zoom",
                  padding=(12, 0)).pack(anchor="w")
        self.status = tk.StringVar(value="Open a Park AFM TIFF, folder, or ZIP.")
        ttk.Label(self.analysis_tab, textvariable=self.status, padding=(12, 3)).pack(anchor="w")

        self.details = tk.StringVar()
        ttk.Label(self.analysis_tab, textvariable=self.details, padding=(12, 8),
                  justify="left").pack(side="bottom", fill="x")
        figure_frame = ttk.Frame(self.analysis_tab)
        figure_frame.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(10, 5), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.navigation = PlotNavigation(self.canvas)
        self.canvas.mpl_connect("button_press_event", self.click_map)
        self.canvas.mpl_connect("motion_notify_event", self.drag_profile)
        self.canvas.mpl_connect("button_release_event", self.release_profile)
        root.bind("<Escape>", self.cancel_tool)
        self.setup_surface()
        self.tabs.bind("<<NotebookTabChanged>>", self.refresh_surface)
        self.render()

    def setup_corrections(self):
        panel = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        panel.pack(side="top", fill="x")

        def control(label, choices, width):
            ttk.Label(panel, text=label).pack(side="left", padx=(0, 4))
            box = ttk.Combobox(panel, values=choices, state="readonly", width=width)
            box.current(0)
            box.pack(side="left", padx=(0, 10))
            box.bind("<<ComboboxSelected>>", self.render)
            return box

        self.level = control("Background", BACKGROUNDS, 17)
        self.flatten = control("Row flatten", ROW_FLATTEN, 13)
        self.zero = control("Zero Z", ZERO_REFERENCES, 16)
        self.correct_measurements = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Measure A/B after correction", variable=self.correct_measurements,
                        command=self.update_details).pack(side="left")

    def setup_surface(self):
        tools = ttk.Frame(self.surface_tab, padding=(12, 8))
        tools.pack(fill="x")
        ttk.Label(tools, text="Left drag to orbit · Scroll to zoom").pack(side="left")
        ttk.Button(tools, text="Reset camera", command=lambda: self.surface.reset()).pack(side="right")
        self.camera = tk.StringVar()
        ttk.Label(self.surface_tab, textvariable=self.camera, padding=(12, 2)).pack(anchor="w")
        figure = Figure(figsize=(10, 5), layout="constrained")
        canvas = FigureCanvasTkAgg(figure, master=self.surface_tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.surface = SurfaceView(figure, self.update_camera_label)
        self.surface_key = None
        self.surface.show(None, None, "")

    def update_camera_label(self, azimuth, elevation, zoom):
        low, high = self.surface.ax.get_zlim()
        self.camera.set(f"Azimuth {azimuth:.0f}°   Elevation {elevation:.0f}°   Zoom {zoom:.2f}×"
                        f"   |   Z limits {low:.4g} to {high:.4g} {self.surface.scan.unit}")

    def refresh_surface(self, event=None):
        if self.scan is None or self.tabs.select() != str(self.surface_tab):
            return
        settings = self.correction_settings()
        key = (id(self.scan), settings)
        if key != self.surface_key:
            self.camera.set("")
            self.surface.show(self.scan, self.displayed_values(self.scan), settings.label)
            self.surface_key = key

    def open_files(self):
        files = filedialog.askopenfilenames(
            title="Open Park AFM scans", filetypes=[("Park AFM scans", "*.zip *.tif *.tiff"), ("All files", "*.*")]
        )
        if files:
            self.open_inputs([Path(file) for file in files])

    def open_folder(self):
        folder = filedialog.askdirectory(title="Open folder of Park AFM scans")
        if folder:
            self.open_inputs([Path(folder)])

    def open_inputs(self, paths):
        scans, problems = load_inputs(paths)
        if not scans:
            messagebox.showerror("No scans loaded", "\n".join(problems[:12]))
            return
        self.scans = scans
        self.channel.configure(values=[
            f"{i + 1:02d}  {scan.channel} · {scan.direction}   |   {scan.name}"
            for i, scan in enumerate(scans)
        ])
        self.channel.current(next((i for i, scan in enumerate(scans)
                                   if scan.is_height and scan.direction == "Forward"), 0))
        self.select_scan()
        self.status.set(f"Loaded {len(scans)} channels.")
        if problems:
            messagebox.showwarning("Some files were skipped", "\n".join(problems[:12]))

    def select_scan(self, event=None):
        self.scan = self.scans[self.channel.current()]
        self.pair = next((p for p in scan_pairs(self.scans) if any(s is self.scan for s in p)), None)
        minimum = min(self.scan.values.shape)
        options = BACKGROUNDS if minimum >= 3 else BACKGROUNDS[:3] if minimum >= 2 else BACKGROUNDS[:1]
        self.level.configure(values=options if self.scan.is_height else BACKGROUNDS[:1])
        self.flatten.configure(values=ROW_FLATTEN if self.scan.is_height else ROW_FLATTEN[:1])
        self.zero.configure(values=ZERO_REFERENCES if self.scan.is_height else ZERO_REFERENCES[:1])
        for control in (self.level, self.flatten, self.zero):
            control.current(0)
        self.correction_cache.clear()
        self.clear_selection()

    def clear_selection(self):
        self.points.clear()
        self.profile_axis = None
        self.profile_drag = None
        self.regions.clear()
        self.tool = None
        if self.level.get() == "Region A plane":
            self.level.current(0)
        if self.zero.get() == "Region A median":
            self.zero.current(0)
        self.status.set("Choose a line profile or select a region.")
        self.render()

    def fit_map(self):
        if self.scan is not None:
            width, height = self.scan.size_um
            self.map_ax.set(xlim=(0, width), ylim=(0, height))
            self.canvas.draw_idle()

    def start_tool(self, mode: str):
        if self.scan is None:
            return
        if mode in ("A", "B") and not self.scan.is_height:
            self.status.set("Select a topography channel to measure height and roughness.")
            return
        self.profile_drag = None
        if mode == "line":
            axis = PROFILE_MODES[self.profile_mode.get()]
            if not axis or axis != self.profile_axis or not self.points:
                self.change_profile_mode()
            if self.profile_axis:
                self.tool = "axis"
                self.status.set("Click or drag on the map to place the profile line.")
                self.render()
            return
        else:
            self.tool = mode
            self.status.set(f"Drag a rectangle for region {mode}.")
        self.render()

    def change_profile_mode(self, event=None):
        if self.scan is None:
            return
        self.profile_axis = PROFILE_MODES[self.profile_mode.get()]
        self.profile_drag = None
        if self.profile_axis:
            fixed = 1 if self.profile_axis == "x" else 0
            limits = self.map_ax.get_ylim() if fixed else self.map_ax.get_xlim()
            self.points = axis_endpoints(self.scan, self.profile_axis, sum(limits) / 2)
            self.tool = None
            self.status.set("Drag the profile line, or enter its position and press Enter.")
        else:
            self.points.clear()
            self.tool = "line"
            self.status.set("Click the start and end of the line on the map.")
        self.render()

    def set_profile_position(self, event=None):
        if self.scan is None or not self.profile_axis or not self.points:
            return
        fixed = 1 if self.profile_axis == "x" else 0
        try:
            position = float(self.position.get())
            if not 0 <= position <= self.scan.size_um[fixed]:
                raise ValueError
        except ValueError:
            self.status.set(f"Enter a position from 0 to {self.scan.size_um[fixed]:g} µm.")
            self.update_profile()
            return
        self.points = axis_endpoints(self.scan, self.profile_axis, position)
        self.update_profile()

    def cancel_tool(self, event=None):
        self.profile_drag = None
        if self.tool is None:
            return
        if self.tool == "line" and len(self.points) == 1:
            self.points.clear()
        self.tool = None
        self.status.set("Selection canceled.")
        self.render()

    def click_map(self, event):
        if (self.scan is None or self.tool in ("A", "B") or event.button != 1
                or event.inaxes is not self.map_ax or event.xdata is None or event.ydata is None):
            return
        point = tuple(float(np.clip(v, 0, size)) for v, size in zip(
            (event.xdata, event.ydata), self.scan.size_um
        ))
        if self.tool == "line":
            if self.points and point == self.points[0]:
                return
            self.points.append(point)
            if len(self.points) == 2:
                self.tool = None
                self.status.set("Drag the line to move the profile.")
            self.update_profile()
        elif self.tool == "axis":
            position = point[1 if self.profile_axis == "x" else 0]
            self.points = axis_endpoints(self.scan, self.profile_axis, position)
            self.tool = None
            self.profile_drag = (point, self.points.copy())
            self.status.set("Drag the line to move the profile.")
            self.update_profile()
        elif len(self.points) == 2 and near_line(self.map_ax.transData.transform(self.points), event.x, event.y):
            self.profile_drag = (point, self.points.copy())

    def drag_profile(self, event):
        if self.profile_drag is None or event.x is None or event.y is None:
            return
        (x0, y0), points = self.profile_drag
        x, y = self.map_ax.transData.inverted().transform((event.x, event.y))
        dx = 0 if self.profile_axis == "x" else x - x0
        dy = 0 if self.profile_axis == "y" else y - y0
        self.points = translated_line(points, dx, dy, self.scan.size_um)
        self.update_profile()

    def release_profile(self, event):
        if event.button == 1:
            self.profile_drag = None

    def select_region(self, start, end):
        if any(value is None for value in (start.xdata, start.ydata, end.xdata, end.ydata)):
            return
        width, height = self.scan.size_um
        x0, x1 = sorted(np.clip([start.xdata, end.xdata], 0, width))
        y0, y1 = sorted(np.clip([start.ydata, end.ydata], 0, height))
        bounds = tuple(map(float, (x0, x1, y0, y1)))
        try:
            roughness(self.scan, bounds)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.regions[self.tool] = bounds
        self.status.set(f"Region {self.tool} measured.")
        self.tool = None
        self.root.after_idle(self.render)

    def correction_settings(self) -> Corrections:
        background, zero = self.level.get(), self.zero.get()
        reference = self.regions.get("A") if background == "Region A plane" or zero == "Region A median" else None
        return Corrections(background, self.flatten.get(), zero, reference)

    def displayed_values(self, scan: Scan) -> np.ndarray:
        settings = self.correction_settings()
        if self.correction_key != settings:
            self.correction_cache.clear()
            self.correction_key = settings
        key = id(scan)
        if key not in self.correction_cache:
            self.correction_cache[key] = settings.apply(scan)
        return self.correction_cache[key]

    def measurements(self) -> dict:
        result = {}

        def source(scan):
            if self.correct_measurements.get():
                return replace(scan, values=self.displayed_values(scan))
            return scan

        selected = source(self.scan)
        for label, bounds in self.regions.items():
            result[label] = {"region_um": bounds, **roughness(selected, bounds)}
        if {"A", "B"} <= self.regions.keys():
            result["step"] = [
                {"direction": scan.direction, **step_height(source(scan), self.regions["A"], self.regions["B"])}
                for scan in self.pair or (self.scan,)
            ]
        return result

    def render(self, event=None):
        self.profile_drag = None
        if self.selector is not None:
            self.selector.disconnect_events()
            self.selector = None
        limits = None
        if self.view_scan is not None and self.view_scan is self.scan:
            limits = self.map_ax.get_xlim(), self.map_ax.get_ylim()
        self.figure.clear()
        self.map_ax, self.profile_ax = self.figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
        self.navigation.set_axes(self.map_ax, self.profile_ax, self.scan)
        self.profiles = []
        if self.scan is None:
            self.map_ax.set_axis_off()
            self.profile_ax.set_axis_off()
            self.canvas.draw_idle()
            return
        if self.level.get() == "Region A plane" and "A" not in self.regions:
            self.level.current(0)
            self.status.set("Select region A before leveling to it.")
        if self.zero.get() == "Region A median" and "A" not in self.regions:
            self.zero.current(0)
            self.status.set("Select region A before setting its median to zero.")
        scan = self.scan
        data = self.displayed_values(scan)
        raster = draw_map(self.map_ax, scan, data)
        self.map_ax.set_title(f"{scan.channel} · {scan.direction}\n{self.correction_settings().label}")
        self.figure.colorbar(raster, ax=self.map_ax, label=scan.unit, shrink=0.85)
        for label, (x0, x1, y0, y1) in self.regions.items():
            color = "#38d9e6" if label == "A" else "#85f078"
            self.map_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                            edgecolor=color, linewidth=1.6))
            self.map_ax.text(x0, y1, label, color=color, fontweight="bold")
        self.map_line, = self.map_ax.plot([], [], "o-", color="#38d9e6", linewidth=1.8, markersize=4)
        if limits:
            self.map_ax.set(xlim=limits[0], ylim=limits[1])
        self.view_scan = scan
        self.profile_ax.set(ylabel=scan.unit)
        self.profile_curves = [
            (member, self.profile_ax.plot([], [], label=member.direction)[0])
            for member in self.pair or (scan,)
        ]
        self.profile_legend = self.profile_ax.legend()
        self.profile_hint = self.profile_ax.text(
            0.5, 0.5, "Place a line on the map", ha="center", va="center",
            transform=self.profile_ax.transAxes, color="0.45",
        )
        self.profile_ax.grid(alpha=0.2)
        if self.tool in ("A", "B"):
            self.selector = RectangleSelector(self.map_ax, self.select_region, button=[1],
                                              props={"facecolor": "#38d9e6", "alpha": 0.25},
                                              useblit=False)
        self.update_profile()
        self.update_details()
        self.refresh_surface()

    def update_profile(self):
        if self.points:
            self.map_line.set_data(*zip(*self.points))
        else:
            self.map_line.set_data([], [])
        has_profile = len(self.points) == 2
        axis = self.profile_axis
        title, xlabel, column = "Line profile", "Distance (µm)", 0
        fixed_position = None
        if axis and has_profile:
            fixed = 1 if axis == "x" else 0
            fixed_label = "Y" if fixed else "X"
            fixed_position = self.points[0][fixed]
            column = 1 if axis == "x" else 2
            xlabel = f"Scan {axis.upper()} (µm)"
            title = f"{axis.upper()} profile at {fixed_label} = {fixed_position:.5g} µm"
            self.position_label.set(f"{fixed_label} (µm)")
            self.position.configure(state="normal", from_=0, to=self.scan.size_um[fixed],
                                    increment=self.scan.pitch_um[fixed], format="%.6f")
            self.position.set(f"{fixed_position:.6g}")
        else:
            self.position_label.set("Position (µm)")
            self.position.set("")
            self.position.configure(state="disabled")
        self.profile_ax.set(title=title, xlabel=xlabel)
        self.profiles.clear()
        for member, curve in self.profile_curves:
            if not has_profile:
                curve.set_data([], [])
                continue
            values = self.displayed_values(member)
            profile = (axis_profile(member, values, axis, fixed_position) if axis
                       else line_profile(member, values, *self.points))
            self.profiles.append((member.direction, profile))
            curve.set_data(profile[:, column], profile[:, 3])
        self.profile_legend.set_visible(has_profile)
        self.profile_hint.set_visible(not has_profile)
        if has_profile:
            self.profile_ax.relim()
            self.profile_ax.autoscale()
        self.canvas.draw_idle()

    def update_details(self):
        scan = self.scan
        if scan is None:
            return
        ny, nx = scan.values.shape
        sx, sy = scan.size_um
        rate = scan.metadata.get("scan_rate_hz")
        text = f"{scan.mode}  |  {sx:g} × {sy:g} µm  |  {nx} × {ny} px  |  {scan.unit}"
        if rate is not None:
            text += f"  |  line rate {rate:g} Hz"
        if self.pair:
            metrics = pair_metrics(*self.pair)
            text += f"  |  raw F−B RMS {metrics['rms_forward_minus_backward']:.3g} {scan.unit}"
        try:
            measurements = self.measurements()
            if measurements:
                mode = "corrected heights" if self.correct_measurements.get() else "raw heights"
                text += f"  |  A/B measurements use {mode}"
            for label in ("A", "B"):
                if label in measurements:
                    result = measurements[label]
                    text += f"\nRegion {label}, plane corrected   Sa {result['Sa_nm']:.3g} nm   Sq {result['Sq_nm']:.3g} nm"
            if "step" in measurements:
                values = "  |  ".join(f"{s['direction']} {s['B_minus_A_nm']:.3g} nm" for s in measurements["step"])
                text += f"\nStep B − A, common slope   {values}"
        except ValueError as exc:
            text += f"\n{exc}"
        self.details.set(text)

    def export(self):
        if self.scan is None:
            return
        try:
            measurements = self.measurements()
        except ValueError as exc:
            messagebox.showerror("Check the selected regions", str(exc))
            return
        parent = filedialog.askdirectory(title="Choose where to save the analysis")
        if not parent:
            return
        try:
            folder = Path(tempfile.mkdtemp(prefix="afm-", dir=parent))
            scan = self.scan
            values = self.displayed_values(scan)
            settings = self.correction_settings()
            save_map(folder / "map.png", scan, values, f"{scan.channel} · {scan.direction}\n{settings.label}")
            save_pixels(folder / "pixels.csv", scan, values)
            for direction, data in self.profiles:
                np.savetxt(folder / f"profile_{direction.lower()}.csv", data,
                           fmt="%.10g", delimiter=",", comments="",
                           header="distance_um,x_um,y_um,value")
            save_json(folder / "summary.json", {
                **describe(scan),
                "map_transform": settings.label,
                "corrections": asdict(settings),
                "measurement_source": "corrected" if self.correct_measurements.get() else "raw",
                "line_endpoints_um": self.points if len(self.points) == 2 else None,
                "profile_axis": self.profile_axis,
                "regions": measurements,
                "raw_direction_comparison": pair_metrics(*self.pair) if self.pair else None,
            })
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not export", str(exc))
            return
        messagebox.showinfo("Analysis saved", str(folder))


if __name__ == "__main__":
    root = tk.Tk()
    Viewer(root)
    root.mainloop()
