import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_cli_train_and_creates_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    config_src = project_root / "configs" / "demo.yaml"
    config_text = config_src.read_text().replace("out_dir: results", f"out_dir: {results_dir}")
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(config_text)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(project_root / "data" / "demo.csv", data_dir / "demo.csv")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    cmd = [
        sys.executable,
        "-m",
        "nircls.cli",
        "--no-progress",
        "train",
        "--config",
        str(config_path),
    ]

    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    run_dirs = list(results_dir.glob("train_*"))
    assert run_dirs, "Training run directory was not created"
    run_dir = run_dirs[0]

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "model.joblib").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "figures").is_dir()
