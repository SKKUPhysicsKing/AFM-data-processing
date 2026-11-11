"""Desktop viewer for Park Systems AFM scan files."""

from __future__ import annotations

import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

from afm import Scan, level_height, line_profile, load_inputs, pair_metrics, roughness, scan_pairs, step_height
from batch import describe, draw_map, save_json, save_map, save_pixels


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
        self.view_scan = None
        root.title("Park AFM Data Processing")
        root.geometry("1200x780")
        root.minsize(850, 620)

        panel = ttk.Frame(root, padding=(12, 10))
        panel.pack(side="top", fill="x")
        ttk.Button(panel, text="Open ZIP / TIFF", command=self.open_files).pack(side="left")
        ttk.Button(panel, text="Open folder", command=self.open_folder).pack(side="left", padx=5)
        self.channel = ttk.Combobox(panel, state="readonly", width=44)
        self.channel.pack(side="left", fill="x", expand=True, padx=6)
        self.channel.bind("<<ComboboxSelected>>", self.select_scan)
        self.level = ttk.Combobox(panel, state="readonly", width=18)
        self.level.pack(side="left", padx=5)
        self.level.bind("<<ComboboxSelected>>", self.render)
        ttk.Button(panel, text="Export", command=self.export).pack(side="left")

        tools = ttk.Frame(root, padding=(12, 0))
        tools.pack(side="top", fill="x")
        for label, mode in (("Line profile", "line"), ("Region A", "A"), ("Region B", "B")):
            ttk.Button(tools, text=label, command=lambda m=mode: self.start_tool(m)).pack(side="left", padx=(0, 5))
        ttk.Button(tools, text="Clear selection", command=self.clear_selection).pack(side="left", padx=(0, 5))
        ttk.Button(tools, text="Fit map", command=self.fit_map).pack(side="left")
        self.status = tk.StringVar(value="Open a Park AFM TIFF, folder, or ZIP.")
        ttk.Label(tools, textvariable=self.status, padding=(10, 0)).pack(side="left")

        self.details = tk.StringVar()
        ttk.Label(root, textvariable=self.details, padding=(12, 8), justify="left").pack(side="bottom", fill="x")
        figure_frame = ttk.Frame(root)
        figure_frame.pack(fill="both", expand=True)
        self.figure = Figure(figsize=(10, 5), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=figure_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, figure_frame, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.click_map)
        self.render()

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
        self.channel.current(0)
        self.select_scan()
        self.status.set(f"Loaded {len(scans)} channels.")
        if problems:
            messagebox.showwarning("Some files were skipped", "\n".join(problems[:12]))

    def select_scan(self, event=None):
        self.scan = self.scans[self.channel.current()]
        self.pair = next((p for p in scan_pairs(self.scans) if any(s is self.scan for s in p)), None)
        self.level.configure(values=("Raw", "Full scan plane", "Region A plane") if self.scan.is_height else ("Raw",))
        self.level.current(0)
        self.clear_selection()

    def clear_selection(self):
        self.points.clear()
        self.regions.clear()
        self.tool = None
        if self.level.get() == "Region A plane":
            self.level.current(0)
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
        if self.toolbar.mode:
            self.status.set("Turn off pan or zoom before selecting.")
            return
        self.tool = mode
        if mode == "line":
            self.points.clear()
            self.status.set("Click the start and end of the line on the map.")
        else:
            self.status.set(f"Drag a rectangle for region {mode}.")
        self.render()

    def click_map(self, event):
        if (self.tool != "line" or self.toolbar.mode or event.button != 1
                or event.inaxes is not self.map_ax or event.xdata is None or event.ydata is None):
            return
        point = (event.xdata, event.ydata)
        if self.points and point == self.points[0]:
            return
        self.points.append(point)
        if len(self.points) == 2:
            self.tool = None
            self.status.set("Line profile measured.")
        self.render()

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

    def displayed_values(self, scan: Scan) -> np.ndarray:
        mode = self.level.get()
        if mode == "Full scan plane":
            return level_height(scan)
        if mode == "Region A plane":
            return level_height(scan, self.regions["A"])
        return scan.values

    def measurements(self) -> dict:
        result = {}
        for label, bounds in self.regions.items():
            result[label] = {"region_um": bounds, **roughness(self.scan, bounds)}
        if {"A", "B"} <= self.regions.keys():
            result["step"] = [
                {"direction": scan.direction, **step_height(scan, self.regions["A"], self.regions["B"])}
                for scan in self.pair or (self.scan,)
            ]
        return result

    def render(self, event=None):
        if self.selector is not None:
            self.selector.disconnect_events()
            self.selector = None
        limits = None
        if self.view_scan is not None and self.view_scan is self.scan:
            limits = self.map_ax.get_xlim(), self.map_ax.get_ylim()
        self.figure.clear()
        self.map_ax, profile_ax = self.figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
        self.profiles = []
        if self.scan is None:
            self.map_ax.set_axis_off()
            profile_ax.set_axis_off()
            self.canvas.draw_idle()
            return
        if self.level.get() == "Region A plane" and "A" not in self.regions:
            self.level.current(0)
            self.status.set("Select region A before leveling to it.")
        scan = self.scan
        data = self.displayed_values(scan)
        raster = draw_map(self.map_ax, scan, data)
        self.map_ax.set_title(f"{scan.channel} · {scan.direction}\n{self.level.get()}")
        self.figure.colorbar(raster, ax=self.map_ax, label=scan.unit, shrink=0.85)
        for label, (x0, x1, y0, y1) in self.regions.items():
            color = "#38d9e6" if label == "A" else "#85f078"
            self.map_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                            edgecolor=color, linewidth=1.6))
            self.map_ax.text(x0, y1, label, color=color, fontweight="bold")
        if self.points:
            x, y = zip(*self.points)
            self.map_ax.plot(x, y, "o-", color="#38d9e6", linewidth=1.6, markersize=4)
        if limits:
            self.map_ax.set(xlim=limits[0], ylim=limits[1])
        self.view_scan = scan
        profile_ax.set(title="Line profile", xlabel="Distance (µm)", ylabel=scan.unit)
        if len(self.points) == 2:
            for member in self.pair or (scan,):
                profile = line_profile(member, self.displayed_values(member), *self.points)
                self.profiles.append((member.direction, profile))
                profile_ax.plot(profile[:, 0], profile[:, 3], label=member.direction)
            profile_ax.legend()
            profile_ax.grid(alpha=0.2)
        else:
            profile_ax.text(0.5, 0.5, "Select two points on the map", ha="center",
                            va="center", transform=profile_ax.transAxes, color="0.45")
        if self.tool in ("A", "B"):
            self.selector = RectangleSelector(self.map_ax, self.select_region, button=[1],
                                              props={"facecolor": "#38d9e6", "alpha": 0.25},
                                              useblit=True)
        self.toolbar.update()
        self.canvas.draw_idle()
        self.update_details()

    def update_details(self):
        scan = self.scan
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
            save_map(folder / "map.png", scan, values, f"{scan.channel} · {scan.direction}\n{self.level.get()}")
            save_pixels(folder / "pixels.csv", scan, values)
            for direction, data in self.profiles:
                np.savetxt(folder / f"profile_{direction.lower()}.csv", data,
                           fmt="%.10g", delimiter=",", comments="",
                           header="distance_um,x_um,y_um,value")
            save_json(folder / "summary.json", {
                **describe(scan),
                "map_transform": self.level.get(),
                "line_endpoints_um": self.points if len(self.points) == 2 else None,
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
