"""Per-user installer and automatic updater for AFM XE7 ANALYZER."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

APP_NAME = "AFM XE7 ANALYZER"
REPO = "SKKUPhysicsKing/AFM-data-processing"
REF_URL = f"https://api.github.com/repos/{REPO}/git/ref/heads/main"
SHA = re.compile(r"[0-9a-f]{40}")
ENV_ID = re.compile(r"[0-9a-f]{24}")
SOURCE_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\.py")
REQUIRED = {"afm.py", "batch.py", "navigation.py", "surface.py", "viewer.py", "requirements.txt"}
PACKAGE_LIMIT = 24 * 1024**2
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def download(url: str, max_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "AFM-XE7-ANALYZER",
                                    "Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=6) as response:
        target = urlsplit(response.geturl())
        if target.scheme != "https" or target.netloc != urlsplit(url).netloc:
            raise ValueError("The update download redirected outside the expected GitHub host.")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("The update download exceeded its size limit.")
    return data


def current_revision() -> str:
    result = json.loads(download(REF_URL, 256 * 1024))
    ref = result["object"]
    revision = ref["sha"]
    if result["ref"] != "refs/heads/main" or ref["type"] != "commit" or not SHA.fullmatch(revision):
        raise ValueError("GitHub returned an invalid update revision.")
    return revision


def unpack_package(data: bytes, revision: str, destination: Path) -> None:
    prefix = f"{REPO.split('/')[1]}-{revision}/"
    with ZipFile(BytesIO(data)) as archive:
        names = set()
        size = 0
        for member in archive.infolist():
            if not member.filename.startswith(prefix) or member.is_dir():
                continue
            name = member.filename[len(prefix):]
            if name != "requirements.txt" and not SOURCE_NAME.fullmatch(name):
                continue
            size += member.file_size
            if name.casefold() in names or size > PACKAGE_LIMIT:
                raise ValueError("Invalid or oversized update package.")
            names.add(name.casefold())
            (destination / name).write_bytes(archive.read(member))
    if not REQUIRED.issubset({entry.name for entry in destination.iterdir()}):
        raise ValueError("The update package is missing application files.")


def env_python(environment: Path, windowed: bool = False) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / ("pythonw.exe" if windowed else "python.exe")
    return environment / "bin" / "python"


def run_command(args, home: Path, timeout: int, env=None) -> None:
    with (home / "launcher.log").open("ab") as log:
        subprocess.run(args, cwd=home, env=env, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT, check=True,
                       timeout=timeout, creationflags=CREATE_NO_WINDOW)


def environment_for(home: Path, package: Path, status) -> str:
    requirements = package / "requirements.txt"
    interpreter = Path(sys.executable)
    if os.name == "nt":
        interpreter = interpreter.with_name("python.exe")
    digest = hashlib.sha256(requirements.read_bytes() + sys.version.encode()
                            + os.fsencode(interpreter)).hexdigest()[:24]
    environment = home / "environments" / digest
    ready = environment / ".ready"
    if not ready.is_file() or not env_python(environment).is_file():
        status("Installing the analysis libraries…")
        # Use the console interpreter for venv and pip even when a shortcut
        # started this script with pythonw.exe.
        run_command([interpreter, "-I", "-m", "venv", environment], home, timeout=120)
        run_command([env_python(environment), "-I", "-m", "pip", "install",
                     "--disable-pip-version-check", "--no-input", "--only-binary=:all:",
                     "-r", requirements], home, timeout=300)
        ready.touch()
    return digest


def check_package(home: Path, package: Path, environment: str) -> None:
    python = env_python(home / "environments" / environment)
    run_command([python, "-I", "-c",
                 "import sys; sys.path.insert(0, sys.argv[1]); import viewer",
                 package], home, timeout=30)


def atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def installed(home: Path) -> dict | None:
    try:
        state = json.loads((home / "installed.json").read_text(encoding="utf-8"))
        revision, environment = state["revision"], state["environment"]
        if (SHA.fullmatch(revision) and ENV_ID.fullmatch(environment)
                and (home / "versions" / revision / "viewer.py").is_file()
                and env_python(home / "environments" / environment, windowed=True).is_file()):
            return state
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


@contextmanager
def update_lock(home: Path):
    with (home / "update.lock").open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Closing the file also releases the lock after a crash.
        yield


def prepare_app(home: Path, status) -> dict:
    """Activate a complete, tested revision; retain the current one on failure."""
    home.mkdir(parents=True, exist_ok=True)
    try:
        with update_lock(home):
            active = installed(home)
            status("Checking for an update…")
            revision = current_revision()
            if active and revision == active["revision"]:
                return active
            status("Downloading the application…")
            versions = home / "versions"
            versions.mkdir(exist_ok=True)
            package = versions / revision
            if package.is_dir():
                environment = environment_for(home, package, status)
                check_package(home, package, environment)
            else:
                archive = download(f"https://codeload.github.com/{REPO}/zip/{revision}", PACKAGE_LIMIT)
                with tempfile.TemporaryDirectory(prefix=".install-", dir=versions) as stage:
                    staging = Path(stage) / "app"
                    staging.mkdir()
                    unpack_package(archive, revision, staging)
                    environment = environment_for(home, staging, status)
                    status("Checking the new application…")
                    check_package(home, staging, environment)
                    staging.rename(package)
            active = {"revision": revision, "environment": environment}
            atomic_write(home / "installed.json", json.dumps(active).encode("utf-8"))
            return active
    except Exception:
        logging.exception("Could not prepare the latest revision")
        active = installed(home)
        if active:
            status("Opening the installed version…")
            return active
        raise


def create_shortcut(home: Path) -> None:
    launcher = home / "launcher.py"
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        raise FileNotFoundError("This Python installation does not include pythonw.exe.")
    atomic_write(launcher, Path(__file__).read_bytes())
    script = """
$ErrorActionPreference = 'Stop'
$programs = [Environment]::GetFolderPath('Programs')
$wsh = New-Object -ComObject WScript.Shell
$link = $wsh.CreateShortcut((Join-Path $programs 'AFM XE7 ANALYZER.lnk'))
$link.TargetPath = $env:AFM_SHORTCUT_PYTHON
$link.Arguments = '-I "' + $env:AFM_SHORTCUT_SCRIPT + '"'
$link.WorkingDirectory = $env:AFM_SHORTCUT_HOME
$link.Description = 'AFM XE7 ANALYZER'
$link.Save()
"""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    system = Path(os.environ["SystemRoot"])
    powershell = system / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    env = os.environ.copy()
    env.update(AFM_SHORTCUT_PYTHON=str(pythonw), AFM_SHORTCUT_SCRIPT=str(launcher),
               AFM_SHORTCUT_HOME=str(home))
    run_command([powershell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                home, timeout=30, env=env)


def launch_app(home: Path, state: dict) -> None:
    viewer = home / "versions" / state["revision"] / "viewer.py"
    python = env_python(home / "environments" / state["environment"], windowed=True)
    env = os.environ.copy()
    env["AFM_XE7_REVISION"] = state["revision"]
    with (home / "viewer.log").open("ab") as log:
        subprocess.Popen([python, "-E", viewer], cwd=Path.home(), env=env,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=CREATE_NO_WINDOW)


def main() -> int:
    import tkinter as tk
    from tkinter import messagebox, ttk

    if os.name != "nt":
        raise SystemExit("The installed launcher is for Windows. Run viewer.py for source use.")
    installing = "--install" in sys.argv
    home = Path(os.environ["LOCALAPPDATA"]) / "Programs" / APP_NAME
    home.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=home / "launcher.log", encoding="utf-8", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    root = tk.Tk()
    root.title(APP_NAME)
    root.resizable(False, False)
    panel = ttk.Frame(root, padding=20)
    panel.pack(fill="both", expand=True)
    ttk.Label(panel, text=APP_NAME, font=("Segoe UI", 14, "bold")).pack(anchor="w")
    label = tk.StringVar(value="Preparing setup…" if installing else "Starting…")
    ttk.Label(panel, textvariable=label, width=46).pack(anchor="w", pady=(8, 9))
    bar = ttk.Progressbar(panel, mode="indeterminate")
    bar.pack(fill="x")
    bar.start(12)
    # Keep the progress window alive until its worker has released the update lock.
    root.protocol("WM_DELETE_WINDOW", lambda: None)
    results = queue.Queue()
    exit_code = 0

    def worker():
        try:
            active = prepare_app(home, lambda message: results.put(("status", message)))
            if installing:
                create_shortcut(home)
            launch_app(home, active)
            results.put(("done", ""))
        except Exception as exc:
            logging.exception("Application startup failed")
            results.put(("error", str(exc)))

    def poll():
        nonlocal exit_code
        while not results.empty():
            event, value = results.get()
            if event == "status":
                label.set(value)
                continue
            if event == "error":
                exit_code = 1
                messagebox.showerror(APP_NAME, f"{value}\n\nDetails: {home / 'launcher.log'}", parent=root)
            root.destroy()
            return
        root.after(75, poll)

    threading.Thread(target=worker, daemon=True).start()
    root.after(75, poll)
    root.mainloop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
