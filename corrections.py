"""Height corrections applied to copies of the calibrated AFM data."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from afm import Region, Scan, _region_grid, level_height

BACKGROUNDS = ("Raw", "Full scan plane", "Region A plane", "Quadratic bow")
ROW_FLATTEN = ("Off", "Median offset", "Linear fit")
ZERO_REFERENCES = ("Off", "Scan median", "Region A median")


@dataclass(frozen=True)
class Corrections:
    background: str = "Raw"
    row_flatten: str = "Off"
    zero_reference: str = "Off"
    reference: Region | None = None

    @property
    def label(self) -> str:
        parts = [self.background]
        if self.row_flatten != "Off":
            parts.append(f"Rows: {self.row_flatten}")
        if self.zero_reference != "Off":
            parts.append(f"Zero: {self.zero_reference}")
        return " · ".join(parts)

    def apply(self, scan: Scan) -> np.ndarray:
        if not scan.is_height:
            return scan.values
        if self.background == "Region A plane" or self.zero_reference == "Region A median":
            if self.reference is None:
                raise ValueError("Select region A for the reference.")
        values = scan.values
        if self.background == "Full scan plane":
            values = level_height(scan)
        elif self.background == "Region A plane":
            values = level_height(scan, self.reference)
        elif self.background == "Quadratic bow":
            values = remove_quadratic_background(values)
        if self.row_flatten != "Off":
            median = np.median(values)
            if self.row_flatten == "Linear fit":
                x = np.arange(values.shape[1], dtype=float)
                x -= x.mean()
                denominator = x @ x
                if denominator:
                    slopes = (values @ x) / denominator
                    values = values - slopes[:, None] * x[None, :]
            values = values - np.median(values, axis=1, keepdims=True) + median
        if self.zero_reference == "Scan median":
            values = values - np.median(values)
        elif self.zero_reference == "Region A median":
            _, _, reference = _region_grid(replace(scan, values=values), self.reference)
            values = values - np.median(reference)
        return values


def remove_quadratic_background(values: np.ndarray) -> np.ndarray:
    """Subtract a least-squares surface with terms 1, x, y, x², xy, y²."""
    rows, columns = values.shape
    if min(rows, columns) < 3:
        raise ValueError("Quadratic bow correction needs at least 3 rows and columns.")
    x, y = np.linspace(-1, 1, columns), np.linspace(-1, 1, rows)
    # On a complete rectangular grid the centered xy term is orthogonal
    # to the separate X and Y polynomials. Fit without a tall design matrix.
    x_fit = np.polynomial.Polynomial.fit(x, values.mean(axis=0), 2)(x)
    y_fit = np.polynomial.Polynomial.fit(y, values.mean(axis=1), 2)(y)
    xy = (y @ (values @ x)) / ((x @ x) * (y @ y))
    background = x_fit[None, :] + y_fit[:, None] - values.mean() + xy * np.outer(y, x)
    return values - background
