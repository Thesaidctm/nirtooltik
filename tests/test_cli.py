import os
import pathlib
import subprocess
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import nircls


def test_version_cli():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "nircls.cli", "--version"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.strip() == nircls.__version__
