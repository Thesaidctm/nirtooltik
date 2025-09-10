#!/usr/bin/env python

"""Build the dashboard executable using PyInstaller."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dashboard app with PyInstaller"
    )
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

