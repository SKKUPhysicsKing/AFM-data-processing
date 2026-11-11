from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import launcher


def package(revision, names=None):
    stream = BytesIO()
    prefix = f"{launcher.REPO.split('/')[1]}-{revision}/"
    with ZipFile(stream, "w") as archive:
        for name in names if names is not None else launcher.REQUIRED:
            archive.writestr(prefix + name, b"")
        archive.writestr(prefix + "../unexpected.py", b"")
    return stream.getvalue()


class LauncherTests(unittest.TestCase):
    def test_unpack_requires_all_app_files_and_ignores_parent_paths(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            staging = home / "stage"
            staging.mkdir()
            launcher.unpack_package(package(revision), revision, staging)
            self.assertEqual({p.name for p in staging.iterdir()}, launcher.REQUIRED)
            self.assertFalse((home / "unexpected.py").exists())
            empty = home / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "missing"):
                launcher.unpack_package(package(revision, ["viewer.py"]), revision, empty)

    def test_update_activates_only_after_check_and_falls_back_offline(self):
        first, second = "a" * 40, "b" * 40
        environment = "e" * 24
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            interpreter = launcher.env_python(home / "environments" / environment, windowed=True)
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            status = lambda message: None
            with patch.object(launcher, "current_revision", return_value=first), \
                 patch.object(launcher, "download", return_value=package(first)), \
                 patch.object(launcher, "environment_for", return_value=environment), \
                 patch.object(launcher, "check_package"):
                active = launcher.prepare_app(home, status)
            self.assertEqual(active["revision"], first)

            with patch.object(launcher, "current_revision", return_value=second), \
                 patch.object(launcher, "download", return_value=package(second)), \
                 patch.object(launcher, "environment_for", return_value=environment), \
                 patch.object(launcher, "check_package", side_effect=ValueError("broken revision")), \
                 self.assertLogs(level="ERROR"):
                self.assertEqual(launcher.prepare_app(home, status), active)
            self.assertFalse((home / "versions" / second).exists())
            self.assertEqual(launcher.installed(home), active)

            with patch.object(launcher, "current_revision", return_value=second), \
                 patch.object(launcher, "download", return_value=package(second)), \
                 patch.object(launcher, "environment_for", return_value=environment), \
                 patch.object(launcher, "check_package"):
                self.assertEqual(launcher.prepare_app(home, status)["revision"], second)
            with patch.object(launcher, "current_revision", side_effect=OSError("offline")), \
                 self.assertLogs(level="ERROR"):
                self.assertEqual(launcher.prepare_app(home, status)["revision"], second)
