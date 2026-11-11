"""Regression checks for Park calibration, geometry, and height measurements."""

import struct
import unittest

import numpy as np

from afm import PARK_MAGIC, PARK_V1, PARK_V2, Scan, level_height, line_profile, read_park_tiff, roughness, step_height


def park_fixture(version, dtype, tiff_order="<"):
    header = bytearray(356 if version == PARK_V1 else 580)
    header[4:24] = "Topography".encode("utf-16-le")
    header[68:82] = "Tapping".encode("utf-16-le")
    header[244:248] = "um".encode("utf-16-le")
    struct.pack_into("<II", header, 100, 3, 2)
    struct.pack_into("<dd", header, 140, 6, 2)
    struct.pack_into("<ddd", header, 220, -0.01, 2, 3)
    if version == PARK_V2:
        struct.pack_into("<I", header, 348, {"<i2": 0, "<i4": 1, "<f4": 2}[dtype])
    raw = np.array([0, 1, -2, 3, 4, 5], dtype=dtype).tobytes()
    data_start = 8 + 2 + 4 * 12 + 4
    tags = [(50432, 4, 1, PARK_MAGIC), (50433, 4, 1, version),
            (50434, 1, len(raw), data_start), (50435, 7, len(header), data_start + len(raw))]
    directory = struct.pack(tiff_order + "H", len(tags))
    directory += b"".join(struct.pack(tiff_order + "HHII", *tag) for tag in tags)
    preamble = b"II" if tiff_order == "<" else b"MM"
    return preamble + struct.pack(tiff_order + "HI", 42, 8) + directory + b"\0" * 4 + raw + header


class ParkReaderTests(unittest.TestCase):
    def test_calibration_uses_signed_gain_offset_and_native_order(self):
        for version, dtype, order in (
            (PARK_V1, "<i2", "<"), (PARK_V2, "<i2", "<"),
            (PARK_V2, "<i4", "<"), (PARK_V2, "<f4", ">"),
        ):
            with self.subTest(version=version, dtype=dtype, order=order):
                scan = read_park_tiff(park_fixture(version, dtype, order), "test_Topography_Backward_002.tiff")
                self.assertEqual((scan.mode, scan.direction, scan.unit), ("Tapping", "Backward", "nm"))
                self.assertEqual(scan.size_um, (6, 2))
                self.assertEqual(scan.pitch_um, (2, 1))
                np.testing.assert_allclose(scan.values, [[-30, -50, 10], [-90, -110, -130]])

    def test_rejects_truncated_measurement(self):
        with self.assertRaisesRegex(ValueError, "truncated Park TIFF tag"):
            read_park_tiff(park_fixture(PARK_V2, "<f4")[:-1])


class HeightTests(unittest.TestCase):
    def test_plane_step_and_profile_on_two_tilted_terraces(self):
        x, y = np.meshgrid(np.arange(8) + 0.5, np.arange(8) + 0.5)
        z = 2 * x - 3 * y + 12 + 25 * (y >= 4)
        scan = Scan("Synthetic.tiff", "Topography", "Tapping", "nm", (8, 8), z, {})
        a, b = (1, 5, 0, 3), (1, 5, 5, 8)
        self.assertAlmostEqual(step_height(scan, a, b)["B_minus_A_nm"], 25.0)
        self.assertAlmostEqual(roughness(scan, a)["Sq_nm"], 0.0)
        leveled = level_height(scan, a)
        np.testing.assert_allclose(leveled[:4], 0.0, atol=1e-10)
        np.testing.assert_allclose(leveled[4:], 25.0, atol=1e-10)
        p = line_profile(scan, z, (1.5, 2.5), (5.5, 2.5))
        np.testing.assert_allclose(p[:, 3], [7.5, 9.5, 11.5, 13.5, 15.5])
        with self.assertRaisesRegex(ValueError, "must not share pixels"):
            step_height(scan, a, a)


if __name__ == "__main__":
    unittest.main()
