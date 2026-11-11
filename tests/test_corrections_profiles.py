import unittest

import numpy as np

from afm import Scan
from corrections import Corrections
from profiles import axis_profile, translated_line


def topography(values):
    return Scan("sample.tiff", "Topography", "Tapping", "nm",
                (float(values.shape[1]), float(values.shape[0])), values, {})


class AxisProfileTests(unittest.TestCase):
    def test_uses_native_centers_and_interpolates_between_scan_lines(self):
        z = np.array([[10., 20., 30.], [50., 60., 70.]])
        scan = topography(z)
        horizontal = axis_profile(scan, z, "x", 1)
        np.testing.assert_allclose(horizontal[:, 1:], [[.5, 1, 30], [1.5, 1, 40], [2.5, 1, 50]])
        vertical = axis_profile(scan, z, "y", 2)
        np.testing.assert_allclose(vertical[:, 1:], [[2, .5, 25], [2, 1.5, 65]])

    def test_drag_stops_at_boundary_without_resizing_the_line(self):
        self.assertEqual(translated_line([(1., 1.), (3., 2.)], 10, -10, (5., 5.)),
                         [(3., 0.), (5., 1.)])


class HeightCorrectionTests(unittest.TestCase):
    def test_quadratic_bow_removes_a_tilted_curved_background_including_xy(self):
        y, x = np.mgrid[:7, :12]
        z = 50 + 0.1*x*x + 0.4*y*y - 0.3*x*y + 3*x - 2*y
        scan = topography(z.copy())
        np.testing.assert_allclose(Corrections(background="Quadratic bow").apply(scan), 0, atol=1e-10)
        np.testing.assert_array_equal(scan.values, z)

    def test_row_median_keeps_within_row_step_and_zero_sets_region_reference(self):
        z = np.array([[101., 101., 121., 121.], [51., 51., 71., 71.]])
        scan = topography(z)
        corrected = Corrections(row_flatten="Median offset", zero_reference="Region A median",
                                reference=(0, 2, 0, 2)).apply(scan)
        np.testing.assert_allclose(corrected, [[0, 0, 20, 20], [0, 0, 20, 20]])

    def test_row_linear_removes_distinct_slopes_in_each_scan_line(self):
        z = np.array([[10., 12., 14., 16.], [29., 25., 21., 17.]])
        scan = topography(z)
        corrected = Corrections(row_flatten="Linear fit", zero_reference="Scan median").apply(scan)
        np.testing.assert_allclose(corrected, 0)
