"""Save the plots and numeric data from the current AFM view."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import count
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from textwrap import fill

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font, PatternFill

from afm import Point, Region, Scan
from batch import draw_map, save_json, save_pixels
from profiles import profile_column, profile_labels
from surface import Camera, save_surface

LINE_COLOR = "#38d9e6"
REGION_COLORS = {"A": "#38d9e6", "B": "#85f078"}


def draw_selection(ax, regions: dict[str, Region]):
    for label, (x0, x1, y0, y1) in regions.items():
        color = REGION_COLORS[label]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.6))
        ax.text(x0, y1, label, color=color, fontweight="bold", clip_on=True)


def save_map_view(path, scan, values, points, regions, limits, correction_label):
    fig = Figure(figsize=(7.5, 6), layout="constrained")
    ax = fig.subplots()
    raster = draw_map(ax, scan, values)
    draw_selection(ax, regions)
    if len(points) == 2:
        ax.plot(*zip(*points), "o-", color=LINE_COLOR, linewidth=1.8, markersize=4)
    ax.set(title=f"{scan.channel} · {scan.direction}\n{fill(correction_label, 48)}",
           xlim=limits[0], ylim=limits[1])
    fig.colorbar(raster, ax=ax, label=scan.unit, shrink=0.85)
    fig.savefig(path, dpi=180)


def save_profile_view(path, scan, points, axis, profiles, limits, correction_label):
    fig = Figure(figsize=(8, 5.2), layout="constrained")
    ax = fig.subplots()
    title, xlabel = profile_labels(points, axis)
    for direction, samples in profiles:
        ax.plot(samples[:, profile_column(axis)], samples[:, 3], label=direction)
    ax.set(title=f"{title}\n{fill(correction_label, 60)}", xlabel=xlabel,
           ylabel=f"{'Height' if scan.is_height else scan.channel} ({scan.unit})",
           xlim=limits[0], ylim=limits[1])
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)


def sheet_with_header(workbook, name, labels, widths):
    sheet = workbook.create_sheet(name)
    sheet.freeze_panes = "B2"
    for i, width in enumerate(widths):
        sheet.column_dimensions[chr(ord("A") + i)].width = width
    cells = []
    for label in labels:
        cell = WriteOnlyCell(sheet, value=label)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="25485C")
        cells.append(cell)
    sheet.append(cells)
    return sheet


def save_workbook(path, scan, values, profiles, axis, profile_limits, report):
    """Write the exact displayed samples and source information to Excel."""
    workbook = Workbook(write_only=True)
    unit = scan.unit
    for direction, samples in profiles:
        name = f"Profile {direction}" if direction != "Unknown" else "Line profile"
        sheet = sheet_with_header(workbook, name,
                                  ["Distance (µm)", "X (µm)", "Y (µm)", f"Value ({unit})"],
                                  [18, 18, 18, 20])
        low, high = sorted(profile_limits[0])
        coordinate = samples[:, profile_column(axis)]
        for row in samples[(low <= coordinate) & (coordinate <= high)]:
            sheet.append([float(value) for value in row])

    notes = sheet_with_header(workbook, "Settings", ["Setting", "Value"], [36, 88])
    for key, value in report.items():
        if value is None:
            continue
        cell = WriteOnlyCell(notes, value=json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (dict, tuple, list)) else value)
        # TIFF metadata is text, including filenames that begin with '='.
        if isinstance(cell.value, str):
            cell.data_type = "s"
        notes.append([key, cell])

    # Keep native pixel coordinates, splitting tall data at the Excel row limit.
    x, y = scan.centers_um
    sheet = None
    for number, yc in enumerate(y):
        for offset, xc in enumerate(x):
            index = number * len(x) + offset
            if index % 1_048_575 == 0:
                page = index // 1_048_575 + 1
                name = "Map data" if page == 1 else f"Map data {page}"
                sheet = sheet_with_header(workbook, name, ["X (µm)", "Y (µm)", f"Value ({unit})"],
                                          [18, 18, 20])
            sheet.append([float(xc), float(yc), float(values[number, offset])])
    workbook.save(path)


def save_view(folder: Path, scan: Scan, values: np.ndarray,
              points: tuple[Point, ...], axis: str | None, profiles: tuple,
              regions: dict[str, Region], map_limits, profile_limits,
              camera: Camera, report: dict) -> list[Path]:
    """Export a single view to separately named files in the chosen folder."""
    folder = folder.resolve()
    label = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(scan.name).stem).strip(". ")[:72]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    prefix = f"{label or 'AFM'}_{stamp}"
    existing = tuple(path.name.casefold() for path in folder.iterdir())
    for number in count(1):
        candidate = prefix if number == 1 else f"{prefix}_{number}"
        if not any(name.startswith(f"{candidate.casefold()}_") for name in existing):
            prefix = candidate
            break
    correction_label = report["map_transform"]
    with TemporaryDirectory(prefix=".afm-export-", dir=folder) as temporary:
        staging = Path(temporary)
        save_map_view(staging / "map_2d.png", scan, values, points, regions, map_limits, correction_label)
        if min(values.shape) >= 2:
            save_surface(staging / "surface_3d.png", scan, values, fill(correction_label, 55), camera)
        if profiles:
            save_profile_view(staging / "line_profile.png", scan, points, axis, profiles,
                              profile_limits, correction_label)
        save_workbook(staging / "data.xlsx", scan, values, profiles, axis, profile_limits, report)
        save_pixels(staging / "pixels.csv", scan, values)
        for direction, samples in profiles:
            np.savetxt(staging / f"profile_{direction.lower()}.csv", samples,
                       fmt="%.10g", delimiter=",", comments="", header="distance_um,x_um,y_um,value")
        save_json(staging / "summary.json", report)
        files = []
        try:
            for file in sorted(staging.iterdir()):
                target = folder / f"{prefix}_{file.name}"
                file.rename(target)
                files.append(target)
        except OSError:
            for path in files:
                path.unlink(missing_ok=True)
            raise
    return files
