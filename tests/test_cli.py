import os
import pathlib
import shutil
import subprocess
import sys
import json
import io
import textwrap

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import nircls
from nircls.cli import _load_preprocess_steps


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


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    return env


def test_demo_cli_creates_results(tmp_path: pathlib.Path) -> None:
    env = _base_env()
    proc = subprocess.run(
        [sys.executable, "-m", "nircls.cli", "demo"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    metrics_files = list(tmp_path.glob("results/demo_*/train/metrics.json"))
    assert metrics_files
    figures_dir = metrics_files[0].parent / "figures"
    assert figures_dir.is_dir()


def test_train_cli_with_config_creates_artifacts(tmp_path: pathlib.Path) -> None:
    env = _base_env()
    config_src = pathlib.Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"
    data_src = pathlib.Path(__file__).resolve().parents[1] / "data" / "demo.csv"
    cfg_dst = tmp_path / "demo.yaml"
    shutil.copy(config_src, cfg_dst)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(data_src, data_dir / "demo.csv")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nircls.cli",
            "train",
            "--config",
            str(cfg_dst),
            "--astm",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    metrics_files = list(tmp_path.glob("results/train_*/metrics.json"))
    assert metrics_files
    figures_dir = metrics_files[0].parent / "figures"
    assert figures_dir.is_dir()
    assert (metrics_files[0].parent / "astm_classification.json").exists()


def test_run_all_cli_displays_preprocess_descriptions(
    tmp_path: pathlib.Path,
) -> None:
    env = _base_env()
    data_src = pathlib.Path(__file__).resolve().parents[1] / "data" / "demo.csv"
    shutil.copy(data_src, tmp_path / "demo.csv")
    cfg_dst = tmp_path / "config.yaml"
    cfg_dst.write_text(
        textwrap.dedent(
            """
            data:
              input_path: demo.csv
              target_cols: [target]
              out_dir: results
            preprocess:
              steps: [SNV, MSC]
            models:
              - name: PLSDA
                grid: {n_components: [2], method: [logreg]}
                y_scramble_n_iter: 0
            cv:
              strategies:
                - type: holdout
                  test_size: 0.5
            """
        ).strip()
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nircls.cli",
            "--lang",
            "pt",
            "--no-progress",
            "run-all",
            "--config",
            str(cfg_dst),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Variação Normal Padrão" in proc.stdout
    assert "Correção Multiplicativa de Espalhamento" in proc.stdout


def test_summary_cli_aggregates_results(tmp_path: pathlib.Path) -> None:
    env = _base_env()
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()
    (run1 / "metrics.json").write_text(
        json.dumps([{"target": "t", "model": "M", "accuracy_mean": 0.9}])
    )
    (run2 / "metrics.json").write_text(
        json.dumps([{"target": "t", "model": "M", "accuracy_mean": 0.8}])
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nircls.cli",
            "--log-level",
            "WARNING",
            "summary",
            "--results",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    df = pd.read_fwf(io.StringIO(proc.stdout))
    assert len(df) == 2
    assert set(df["run"]) == {"run1", "run2"}


def test_load_preprocess_steps_handles_multiple_sequences(
    tmp_path: pathlib.Path,
) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            [
                {"preprocess_steps": ["SNV", "SG"]},
                {"preprocess_steps": ["MSC", "SNV"]},
                {"preprocess_steps": None},
                {"preprocess_steps": "EPO"},
            ]
        )
    )

    steps = _load_preprocess_steps(metrics_path)

    assert steps == "SNV, SG, MSC, EPO"
