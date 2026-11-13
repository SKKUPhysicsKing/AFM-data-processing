"""Profiles along the native X or Y pixel centers and draggable line geometry."""

from __future__ import annotations

import numpy as np

from afm import Point, Scan, line_profile

PROFILE_MODES = {"Free line": None, "Horizontal (X)": "x", "Vertical (Y)": "y"}


def axis_endpoints(scan: Scan, axis: str, position: float) -> list[Point]:
    width, height = scan.size_um
    if axis == "x":
        return [(0.0, position), (width, position)]
    return [(position, 0.0), (position, height)]


def axis_profile(scan: Scan, values: np.ndarray, axis: str, position: float) -> np.ndarray:
    """Sample all native X/Y centers, interpolating across the fixed axis."""
    x, y = scan.centers_um
    coordinates, normal, samples = (x, y, values) if axis == "x" else (y, x, values.T)
    extent = scan.size_um[1] if axis == "x" else scan.size_um[0]
    if not 0 <= position <= extent:
        raise ValueError("Profile position is outside the scan.")
    fractional_index = np.interp(position, normal, np.arange(normal.size))
    index = int(fractional_index)
    weight = fractional_index - index
    profile = (1 - weight) * samples[index] + weight * samples[min(index + 1, normal.size - 1)]
    fixed = np.full(coordinates.shape, position)
    xy = (coordinates, fixed) if axis == "x" else (fixed, coordinates)
    return np.column_stack((coordinates - coordinates[0], *xy, profile))


def sample_profile(scan: Scan, values: np.ndarray, points: tuple[Point, Point], axis: str | None) -> np.ndarray:
    if axis:
        return axis_profile(scan, values, axis, points[0][1 if axis == "x" else 0])
    return line_profile(scan, values, *points)


def profile_column(axis: str | None) -> int:
    return {None: 0, "x": 1, "y": 2}[axis]


def profile_labels(points: tuple[Point, ...], axis: str | None) -> tuple[str, str]:
    if axis and len(points) == 2:
        fixed = 1 if axis == "x" else 0
        fixed_label = "Y" if fixed else "X"
        return (f"{axis.upper()} profile at {fixed_label} = {points[0][fixed]:.5g} µm",
                f"Scan {axis.upper()} (µm)")
    return "Line profile", "Distance (µm)"


def translated_line(points: list[Point], dx: float, dy: float, size: Point) -> list[Point]:
    """Move a line as a unit and stop at the scan edges."""
    x, y = zip(*points)
    dx = max(-min(x), min(dx, size[0] - max(x)))
    dy = max(-min(y), min(dy, size[1] - max(y)))
    return [(float(np.clip(px + dx, 0, size[0])), float(np.clip(py + dy, 0, size[1]))) for px, py in points]


def near_line(points_px: np.ndarray, x_px: float, y_px: float, tolerance_px: float = 9) -> bool:
    start, end = points_px
    direction = end - start
    offset = np.array([x_px, y_px]) - start
    length_squared = direction @ direction
    fraction = float(np.clip((offset @ direction) / length_squared, 0, 1)) if length_squared else 0
    return bool(np.linalg.norm(offset - fraction * direction) <= tolerance_px)
