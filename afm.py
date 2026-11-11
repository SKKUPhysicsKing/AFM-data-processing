"""Read calibrated Park Systems TIFF rasters and measure their topography."""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

TIFF_EXTENSIONS = {".tif", ".tiff"}
MAX_TIFF_BYTES = 128 * 1024**2
PARK_MAGIC = 0x0E031301
PARK_V1, PARK_V2 = 0x01000001, 0x01000002
DIRECTION_TOKEN = re.compile(r"(?<=[ _-])(forward|backward)(?=[ _.-]|$)", re.I)
LENGTH_TO_NM = {"m": 1e9, "mm": 1e6, "um": 1e3, "nm": 1.0, "pm": 1e-3, "å": 0.1}

# Regions are (x_min, x_max, y_min, y_max), in micrometres from the scan corner.
Region = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class Scan:
    path: str
    channel: str
    mode: str
    unit: str
    size_um: tuple[float, float]
    values: np.ndarray
    metadata: dict

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def direction(self) -> str:
        token = DIRECTION_TOKEN.search(Path(self.path).stem)
        return token[0].capitalize() if token else "Unknown"

    @property
    def is_height(self) -> bool:
        return self.channel.casefold() == "topography" and self.unit == "nm"

    @property
    def pitch_um(self) -> tuple[float, float]:
        ny, nx = self.values.shape
        return self.size_um[0] / nx, self.size_um[1] / ny

    @property
    def centers_um(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.values.shape
        dx, dy = self.pitch_um
        return (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy


def read_park_tiff(data: bytes, path: str = "scan.tiff") -> Scan:
    """Read the Park private tags in the first IFD of a classic TIFF.

    The raster and header payloads are little endian. TIFF directory entries
    themselves follow the byte order declared in the TIFF preamble.
    """
    if not 8 <= len(data) <= MAX_TIFF_BYTES:
        raise ValueError("TIFF size is invalid or exceeds the supported size")
    order = {b"II": "<", b"MM": ">"}.get(data[:2])
    if order is None or struct.unpack_from(order + "H", data, 2)[0] != 42:
        raise ValueError("Expected a classic TIFF with Park Systems private tags")
    ifd = struct.unpack_from(order + "I", data, 4)[0]
    if not 8 <= ifd <= len(data) - 2:
        raise ValueError("Invalid TIFF directory offset")
    count = struct.unpack_from(order + "H", data, ifd)[0]
    if ifd + 2 + count * 12 + 4 > len(data):
        raise ValueError("Truncated TIFF directory")
    tags = {
        tag: (kind, length, value)
        for tag, kind, length, value in struct.iter_unpack(
            order + "HHII", data[ifd + 2 : ifd + 2 + 12 * count]
        )
    }
    if tags.get(50432) != (4, 1, PARK_MAGIC):
        raise ValueError("Park Systems measurement tags are missing")
    version_tag = tags.get(50433)
    if version_tag is None or version_tag[:2] != (4, 1):
        raise ValueError("Park TIFF version tag is missing or invalid")
    version = version_tag[2]
    if version not in (PARK_V1, PARK_V2):
        raise ValueError(f"Unsupported Park TIFF version 0x{version:08x}")

    def payload(tag: int) -> memoryview:
        kind, length, offset = tags.get(tag, (0, 0, 0))
        if kind not in (1, 7) or length <= 4 or not 8 <= offset <= len(data) - length:
            raise ValueError(f"Missing or truncated Park TIFF tag {tag}")
        return memoryview(data)[offset : offset + length]

    header = payload(50435)
    if len(header) < (356 if version == PARK_V1 else 580):
        raise ValueError("Truncated Park measurement header")

    def uint(offset: int) -> int:
        return struct.unpack_from("<I", header, offset)[0]

    def real(offset: int) -> float:
        return struct.unpack_from("<d", header, offset)[0]

    def label(offset: int, chars: int) -> str:
        return bytes(header[offset : offset + 2 * chars]).decode("utf-16-le").split("\0")[0].strip()

    if uint(0) != 0:
        raise ValueError("Only two-dimensional Park image channels are supported")
    nx, ny = uint(100), uint(104)
    width, height = real(140), real(148)
    if nx < 1 or ny < 1 or not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Invalid Park scan geometry")
    kind = 0 if version == PARK_V1 else uint(348)
    dtype = {0: "<i2", 1: "<i4", 2: "<f4"}.get(kind)
    if dtype is None:
        raise ValueError(f"Unsupported Park raster type {kind}")
    raw = payload(50434)
    if len(raw) != nx * ny * np.dtype(dtype).itemsize:
        raise ValueError("Raw measurement length does not match the scan dimensions")

    gain, scale, offset = real(220), real(228) or 1.0, real(236)
    source_unit = label(244, 8)
    if not source_unit or not gain or not all(map(math.isfinite, (gain, scale, offset))):
        raise ValueError("Park measurement calibration is missing or invalid")
    length_factor = LENGTH_TO_NM.get(source_unit.lower().replace("µ", "u").replace("μ", "u"))
    unit = "nm" if length_factor is not None else source_unit
    values = np.frombuffer(raw, dtype=dtype).astype(np.float64).reshape(ny, nx)
    values = (values * scale + offset) * (gain * (length_factor or 1.0))
    if not np.isfinite(values).all():
        raise ValueError("The calibrated raster contains non-finite values")

    metadata = {
        "park_version": "1.0.1" if version == PARK_V1 else "1.0.2",
        "raw_dtype": np.dtype(dtype).name,
        "source_unit": source_unit,
        "data_gain": gain,
        "z_scale": scale,
        "z_offset": offset,
        "scan_rate_hz": real(172),
        "scan_angle_deg": real(108),
        "x_offset_um": real(156),
        "y_offset_um": real(164),
        "header_forward": bool(uint(128)),
        "scan_up": bool(uint(132)),
        "swap_xy": bool(uint(136)),
        "setpoint_header_value": real(180),
        "setpoint_header_unit": label(188, 8),
    }
    if version == PARK_V2:
        metadata["drive_frequency_hz"] = real(368)
    metadata = {
        key: value if not isinstance(value, float) or math.isfinite(value) else None
        for key, value in metadata.items()
    }
    return Scan(path, label(4, 32) or "Unspecified", label(68, 8),
                unit, (width, height), values, metadata)


def load_inputs(paths: list[Path]) -> tuple[list[Scan], list[str]]:
    """Open TIFFs from files, directories, or ZIP members without extracting them."""
    scans, problems = [], []

    def append(name: str, size: int, read) -> None:
        try:
            if size > MAX_TIFF_BYTES:
                raise ValueError("TIFF exceeds the supported size")
            scans.append(read_park_tiff(read(), name))
        except (ValueError, OSError, RuntimeError, NotImplementedError, BadZipFile) as exc:
            problems.append(f"{Path(name).name}: {exc}")

    for path in paths:
        path = path.resolve()
        try:
            if path.suffix.lower() == ".zip" and path.is_file():
                with ZipFile(path) as archive:
                    for member in archive.infolist():
                        name = Path(member.filename)
                        if (member.is_dir() or "__MACOSX" in name.parts
                                or name.name.startswith("._")
                                or name.suffix.lower() not in TIFF_EXTENSIONS):
                            continue
                        append(f"{path.as_posix()}!/{member.filename}", member.file_size,
                               lambda: archive.read(member))
            else:
                files = sorted(path.rglob("*")) if path.is_dir() else [path]
                for file in files:
                    if file.is_file() and file.suffix.lower() in TIFF_EXTENSIONS:
                        append(file.as_posix(), file.stat().st_size, file.read_bytes)
                    elif file == path:
                        problems.append(f"{path.name}: expected a TIFF, directory, or ZIP")
        except (OSError, BadZipFile) as exc:
            problems.append(f"{path.name}: {exc}")
    if not scans and not problems:
        problems.append("No TIFF files found in the selected input")
    return scans, problems


def scan_pairs(scans: list[Scan]) -> list[tuple[Scan, Scan]]:
    """Pair matching, unambiguous filename directions on the same scan grid."""
    groups: dict[tuple, list[Scan]] = {}
    for scan in scans:
        if scan.direction == "Unknown":
            continue
        stem = DIRECTION_TOKEN.sub("", str(Path(scan.path).with_suffix(""))).casefold()
        geometry = tuple(scan.metadata.get(field) for field in (
            "x_offset_um", "y_offset_um", "scan_angle_deg", "scan_up", "swap_xy"
        ))
        key = (stem, scan.channel, scan.mode, scan.unit, scan.size_um, scan.values.shape, geometry)
        groups.setdefault(key, []).append(scan)
    return [
        tuple(sorted(group, key=lambda s: s.direction != "Forward"))
        for group in groups.values()
        if len(group) == 2 and {scan.direction for scan in group} == {"Forward", "Backward"}
    ]


def pair_metrics(forward: Scan, backward: Scan) -> dict:
    f, b = forward.values, backward.values
    delta = f - b
    return {
        "median_forward_minus_backward": float(np.median(delta)),
        "rms_forward_minus_backward": float(np.sqrt(np.mean(delta**2))),
        "pearson_r": float(np.corrcoef(f.ravel(), b.ravel())[0, 1]) if f.std() and b.std() else None,
    }


def _region_grid(scan: Scan, bounds: Region | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = scan.centers_um
    if bounds is None:
        return x, y, scan.values
    x0, x1, y0, y1 = bounds
    if not (0 <= x0 < x1 <= scan.size_um[0] and 0 <= y0 < y1 <= scan.size_um[1]):
        raise ValueError("Region must be within the scan and have positive width and height")
    i0, i1 = np.searchsorted(x, x0), np.searchsorted(x, x1, side="right")
    j0, j1 = np.searchsorted(y, y0), np.searchsorted(y, y1, side="right")
    if i1 - i0 < 2 or j1 - j0 < 2:
        raise ValueError("Select at least two pixel centers along each axis")
    return x[i0:i1], y[j0:j1], scan.values[j0:j1, i0:i1]


def _parallel_slopes(grids: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Least-squares slopes with an independent intercept for each rectangle."""
    x_num = y_num = x_den = y_den = 0.0
    for x, y, z in grids:
        dx, dy = x - x.mean(), y - y.mean()
        x_num += len(y) * np.dot(dx, z.mean(axis=0))
        x_den += len(y) * np.dot(dx, dx)
        y_num += len(x) * np.dot(dy, z.mean(axis=1))
        y_den += len(x) * np.dot(dy, dy)
    if not x_den or not y_den:
        raise ValueError("Plane fitting needs at least two rows and columns")
    return float(x_num / x_den), float(y_num / y_den)


def level_height(scan: Scan, bounds: Region | None = None) -> np.ndarray:
    """Subtract a first-order plane fitted to the specified topography region."""
    if not scan.is_height:
        raise ValueError("Plane correction requires topography with calibrated height units")
    x, y, z = _region_grid(scan, bounds)
    sx, sy = _parallel_slopes([(x, y, z)])
    all_x, all_y = scan.centers_um
    plane = z.mean() + sx * (all_x - x.mean())[None, :] + sy * (all_y - y.mean())[:, None]
    return scan.values - plane


def roughness(scan: Scan, bounds: Region | None = None) -> dict:
    """Areal Sa and Sq in nm after fitting and subtracting a plane in the ROI."""
    if not scan.is_height:
        raise ValueError("Roughness requires topography with calibrated height units")
    x, y, z = _region_grid(scan, bounds)
    sx, sy = _parallel_slopes([(x, y, z)])
    residual = z - z.mean() - sx * (x - x.mean())[None, :] - sy * (y - y.mean())[:, None]
    return {
        "pixels": z.size,
        "Sa_nm": float(np.mean(np.abs(residual))),
        "Sq_nm": float(np.sqrt(np.mean(residual**2))),
    }


def step_height(scan: Scan, a: Region, b: Region) -> dict:
    """Fit two parallel planes and report the signed B minus A height in nm."""
    if not scan.is_height:
        raise ValueError("Step height requires topography with calibrated height units")
    ga, gb = _region_grid(scan, a), _region_grid(scan, b)
    if np.intersect1d(ga[0], gb[0]).size and np.intersect1d(ga[1], gb[1]).size:
        raise ValueError("Regions A and B must not share pixels")
    sx, sy = _parallel_slopes([ga, gb])

    def region_result(grid) -> tuple[float, float]:
        x, y, z = grid
        corrected = z - sx * x[None, :] - sy * y[:, None]
        return float(corrected.mean()), float(corrected.std())

    za, sa = region_result(ga)
    zb, sb = region_result(gb)
    return {
        "B_minus_A_nm": zb - za,
        "slope_x_nm_per_um": sx,
        "slope_y_nm_per_um": sy,
        "A_fit_rms_nm": sa,
        "B_fit_rms_nm": sb,
        "A_pixels": ga[2].size,
        "B_pixels": gb[2].size,
    }


def line_profile(scan: Scan, values: np.ndarray, start: Point, end: Point) -> np.ndarray:
    """Sample a line in scan coordinates with bilinear interpolation."""
    if start == end or not all(
        0 <= x <= scan.size_um[0] and 0 <= y <= scan.size_um[1] for x, y in (start, end)
    ):
        raise ValueError("Choose distinct line endpoints within the scan")
    dx, dy = scan.pitch_um
    ny, nx = scan.values.shape
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    count = math.ceil(max(abs(delta_x) / dx, abs(delta_y) / dy)) + 1
    x, y = np.linspace(start, end, count).T
    px, py = np.clip(x / dx - 0.5, 0, nx - 1), np.clip(y / dy - 0.5, 0, ny - 1)
    i, j = px.astype(int), py.astype(int)
    ii, jj = np.minimum(i + 1, nx - 1), np.minimum(j + 1, ny - 1)
    wx, wy = px - i, py - j
    z = (1 - wy) * ((1 - wx) * values[j, i] + wx * values[j, ii])
    z += wy * ((1 - wx) * values[jj, i] + wx * values[jj, ii])
    return np.column_stack([np.hypot(x - start[0], y - start[1]), x, y, z])
