#!/usr/bin/env python

"""Build Windows installer for nircls using PyInstaller and NSIS."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APP_NAME = "nircls"
APP_VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"


def build_pyinstaller() -> Path:
    """Run PyInstaller to create a standalone executable.

    Returns the path to the generated executable.
    """
    DIST_DIR.mkdir(exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        APP_NAME,
        str(ROOT / "src" / "nircls" / "cli.py"),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return DIST_DIR / f"{APP_NAME}.exe"


def create_nsis_script(exe_path: Path) -> tuple[Path, Path]:
    """Create a minimal NSIS script for the executable.

    Returns a tuple with the NSIS script path and the installer output path.
    """
    nsis_script = DIST_DIR / f"{APP_NAME}.nsi"
    installer_path = DIST_DIR / f"{APP_NAME}-{APP_VERSION}-setup.exe"
    script_content = f"""
!define APP_NAME "{APP_NAME}"
!define APP_VERSION "{APP_VERSION}"
OutFile "{installer_path}"
InstallDir "$PROGRAMFILES\\{APP_NAME}"

Section "Install"
    SetOutPath "$INSTDIR"
    File "{exe_path}"
    CreateShortCut "$DESKTOP\\{APP_NAME}.lnk" "$INSTDIR\\{APP_NAME}.exe"
SectionEnd
"""
    nsis_script.write_text(script_content.strip() + "\n")
    return nsis_script, installer_path


def build_installer() -> Path | None:
    """Build the installer on Windows platforms."""
    if sys.platform != "win32":
        print("Installer creation is only supported on Windows.")
        return None

    exe_path = build_pyinstaller()
    nsis_script, installer_path = create_nsis_script(exe_path)
    subprocess.run(["makensis", str(nsis_script)], check=True)
    print(f"Installer created at {installer_path}")
    return installer_path


if __name__ == "__main__":
    build_installer()
=======
"""Build the dashboard executable using PyInstaller."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dashboard app with PyInstaller")
    parser.add_argument(
        "--dist",
        type=Path,
        default=Path("dist"),
        help="Directory to output the executable.",
    )
    return parser.parse_args()


def build(dist: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    entry = root / "src" / "nircls" / "dashboard.py"
    add_dirs = [".streamlit", "models", "templates"]
    data_args: list[str] = []
    for d in add_dirs:
        src = root / d
        data_args.extend(["--add-data", f"{src}{os.pathsep}{d}"])

    args = [
        "pyinstaller",
        "--onefile",
        "--name",
        "nircls-dashboard",
        "--distpath",
        str(dist),
        *data_args,
        str(entry),
    ]
    result = subprocess.run(args)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    build_dir = root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    spec_file = root / "nircls-dashboard.spec"
    if spec_file.exists():
        spec_file.unlink()


def main() -> None:
    ns = parse_args()
    build(ns.dist)


if __name__ == "__main__":
    main()

