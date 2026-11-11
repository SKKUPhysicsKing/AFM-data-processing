"""Export Park Systems scan maps, calibrated pixel tables, and measurements."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from afm import Scan, level_height, load_inputs, pair_metrics, roughness, scan_pairs, step_height


def draw_map(ax, scan: Scan, values: np.ndarray, difference: bool = False):
    """Show the native pixel grid with physical axes and robust display limits."""
    low, high = np.percentile(values, [1, 99])
    diverging = difference or scan.channel.casefold() == "error signal"
    if diverging:
        limit = max(abs(low), abs(high))
        low, high = -limit, limit
    if high == low:
        low, high = low - 0.5, high + 0.5
    width, height = scan.size_um
    raster = ax.imshow(values, origin="lower", extent=(0, width, 0, height),
                       interpolation="nearest", cmap="RdBu_r" if diverging else "afmhot",
                       vmin=low, vmax=high)
    ax.set(xlabel="Scan X (µm)", ylabel="Scan Y (µm)")
    return raster


def save_map(path: Path, scan: Scan, values: np.ndarray, title: str,
             difference: bool = False) -> None:
    fig = Figure(figsize=(6.6, 5.4), layout="constrained")
    ax = fig.subplots()
    raster = draw_map(ax, scan, values, difference)
    ax.set_title(title)
    fig.colorbar(raster, ax=ax, label=scan.unit, shrink=0.88)
    fig.savefig(path, dpi=160)


def save_pixels(path: Path, scan: Scan, values: np.ndarray) -> None:
    x, y = scan.centers_um
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("x_um,y_um,value\n")
        for yc, row in zip(y, values):
            np.savetxt(f, np.column_stack((x, np.full_like(x, yc), row)),
                       delimiter=",", fmt="%.10g")


def describe(scan: Scan) -> dict:
    values = scan.values
    return {
        "file": scan.name,
        "channel": scan.channel,
        "mode": scan.mode,
        "direction_from_filename": scan.direction,
        "width_um": scan.size_um[0],
        "height_um": scan.size_um[1],
        "columns": values.shape[1],
        "rows": values.shape[0],
        "pixel_pitch_um": list(scan.pitch_um),
        "value_unit": scan.unit,
        "raw_min": float(values.min()),
        "raw_median": float(np.median(values)),
        "raw_max": float(values.max()),
        **scan.metadata,
    }


def save_json(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Park TIFFs, directories, or ZIPs")
    parser.add_argument("-o", "--output", type=Path, default=Path("afm-results"),
                        help="new output directory (default: afm-results)")
    level = parser.add_mutually_exclusive_group()
    level.add_argument("--plane", action="store_true", help="subtract a full-scan plane from topography")
    level.add_argument("--background", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                       help="fit the topography leveling plane only inside this rectangle (µm)")
    parser.add_argument("--step", type=float, nargs=8, metavar="BOUND",
                        help="X0 X1 Y0 Y1 for region A, then region B (µm)")
    args = parser.parse_args()

    scans, problems = load_inputs(args.inputs)
    if not scans:
        parser.exit(2, "No supported scans loaded.\n" + "\n".join(problems) + "\n")

    background = tuple(args.background) if args.background else None
    regions = (tuple(args.step[:4]), tuple(args.step[4:])) if args.step else None
    if (args.plane or background or regions) and not any(s.is_height for s in scans):
        parser.error("Plane correction and step height require a calibrated topography channel")
    report = {"scans": [], "pairs": [], "skipped_files": problems, "step_regions_um": regions}
    try:
        for scan in scans:
            info = describe(scan)
            info["map_transform"] = "raw"
            if scan.is_height:
                if regions:
                    info["step"] = step_height(scan, *regions)
                    info["roughness_A_after_plane"] = roughness(scan, regions[0])
                    info["roughness_B_after_plane"] = roughness(scan, regions[1])
                if args.plane or background:
                    info["map_transform"] = "plane_from_region" if background else "full_scan_plane"
                    info["plane_region_um"] = background
                    roughness(scan, background)  # Validate the rectangle before creating output.
            report["scans"].append(info)
        args.output.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")

    for i, (scan, info) in enumerate(zip(scans, report["scans"]), 1):
        channel = re.sub(r"[^a-z0-9]+", "_", scan.channel.lower()).strip("_")
        stem = f"{i:02d}_{channel}_{scan.direction.lower()}"
        transform = info["map_transform"]
        values = scan.values if transform == "raw" else level_height(scan, background)
        info["map_png"], info["pixels_csv"] = f"{stem}.png", f"{stem}.csv"
        save_map(args.output / info["map_png"], scan, values,
                 f"{scan.channel} · {scan.direction}\n{transform.replace('_', ' ')}")
        save_pixels(args.output / info["pixels_csv"], scan, values)
    for i, (f, b) in enumerate(scan_pairs(scans), 1):
        metrics = {"forward": f.name, "backward": b.name, "value_unit": f.unit, **pair_metrics(f, b)}
        if f.is_height:
            filename = f"pair_{i:02d}_height_forward_minus_backward.png"
            save_map(args.output / filename, f, f.values - b.values,
                     "Topography · Forward minus backward", difference=True)
            metrics["difference_png"] = filename
        report["pairs"].append(metrics)
    save_json(args.output / "summary.json", report)
    for problem in problems:
        print(f"Skipped: {problem}", file=sys.stderr)
    print(f"Exported {len(scans)} channels and {len(report['pairs'])} direction pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
