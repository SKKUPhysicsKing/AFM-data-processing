"""Regression checks for the reference plane and the files from a view export."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from PIL import Image

from afm import Scan
from corrections import Corrections, ProfileBaseline
from exporting import save_view
from profiles import sample_profile
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.proj3d import proj_transform
from surface import Camera, orbit_delta


def sample_scan():
    x, y = np.meshgrid(np.arange(12) + .5, np.arange(10) + .5)
    values = 100 + 4*x - 3*y + np.where(x >= 6, 30, 0)
    return Scan("Scan[1]_Topography_Forward.tiff", "Topography", "Tapping", "nm",
                (12., 10.), values, {})


class ReferenceFlatteningTests(unittest.TestCase):
    def test_profile_baseline_subtracts_height_trend_from_entire_map(self):
        scan = sample_scan()
        reference = ProfileBaseline(((0., 4.5), (12., 4.5)), "x", (1., 5.))
        corrected = Corrections(profile_baseline=reference).apply(scan)
        for points in (reference.points, ((0., 7.5), (12., 7.5))):
            profile = sample_profile(scan, corrected, points, "x")
            np.testing.assert_allclose(np.diff(profile[:6, 3]), 0, atol=1e-10)
            self.assertAlmostEqual(profile[-1, 3] - profile[0, 3], 30.)
        np.testing.assert_allclose(corrected[4, :6], 0, atol=1e-10)
        self.assertAlmostEqual(scan.values[4, 0], 88.5)

    def test_region_a_stays_flat_after_removing_row_offsets(self):
        scan = sample_scan()
        x, y = np.meshgrid(np.arange(12) + .5, np.arange(10) + .5)
        # Different sample coverage from row to row shifts the row median.
        values = 7*x - 2*y + 12*(x >= y + 1)
        scan = Scan(scan.path, scan.channel, scan.mode, scan.unit, scan.size_um, values, {})
        corrected = Corrections(background="Region A plane", row_flatten="Median offset",
                                reference=(0, 2, 4, 8)).apply(scan)
        reference = corrected[4:8, :2]
        self.assertAlmostEqual(reference.mean(), 0.)
        np.testing.assert_allclose(np.polyfit(y[4:8, 0], reference.mean(axis=1), 1)[0], 0., atol=1e-10)


class ViewExportTests(unittest.TestCase):
    def test_pngs_and_workbook_capture_the_active_line_and_corrected_map(self):
        scan = sample_scan()
        points = ((0., 4.5), (12., 4.5))
        values = Corrections(background="Region A plane", reference=(1, 5, 2, 7)).apply(scan)
        line = sample_profile(scan, values, points, "x")
        profiles = (("Forward", line), ("Backward", line + [0, 0, 0, 2]))
        report = {"file": "=literal", "map_transform": "Region A plane", "camera_3d": {"azimuth": -21.}}
        camera = Camera(azimuth=-21., elevation=46., zoom=1.2)
        with tempfile.TemporaryDirectory() as output:
            folder = Path(output)
            files = save_view(folder, scan, values, points, "x", profiles, {"A": (1, 5, 2, 7)},
                              ((2, 9), (3, 7)), ((1, 10), (-5, 35)), camera, report)
            self.assertEqual(len(files), 8)
            pngs = [path for path in files if path.suffix == ".png"]
            self.assertEqual(len(pngs), 3)
            for path in pngs:
                with Image.open(path) as image:
                    self.assertGreater(image.width, 1000)
                    image.verify()
            with next(path for path in files if path.suffix == ".xlsx").open("rb") as stream:
                workbook = load_workbook(stream, read_only=True, data_only=True)
                sheet = workbook["Profile Forward"]
                visible = (1 <= line[:, 1]) & (line[:, 1] <= 10)
                np.testing.assert_allclose(list(sheet.values)[1:], line[visible])
                np.testing.assert_allclose(list(workbook["Profile Backward"].values)[1:], profiles[1][1][visible])
                np.testing.assert_allclose(np.array(list(workbook["Map data"].values)[1:])[:, 2], values.ravel())
                self.assertEqual(dict(list(workbook["Settings"].values)[1:])["file"], "=literal")
                workbook.close()

    def test_drag_projects_a_front_point_in_the_same_screen_direction(self):
        ax = Figure().add_subplot(projection="3d")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        ax.set_box_aspect((1, 1, 1))
        for azimuth in (-60, 20, 145):
            camera = Camera(azimuth=azimuth)
            elevation, yaw = np.deg2rad([camera.elevation, camera.azimuth])
            point = 0.6 * np.array([np.cos(elevation) * np.cos(yaw),
                                    np.cos(elevation) * np.sin(yaw), np.sin(elevation)])
            ax.view_init(elev=camera.elevation, azim=camera.azimuth)
            before = proj_transform(*point, ax.get_proj())
            result = orbit_delta(30, 20, 440, camera)
            ax.view_init(elev=result.elevation, azim=result.azimuth)
            after = proj_transform(*point, ax.get_proj())
            self.assertGreater(after[0], before[0])
            self.assertGreater(after[1], before[1])
