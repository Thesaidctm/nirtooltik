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


def test_train_cli_resume_skips_existing(tmp_path: pathlib.Path) -> None:
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
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    metrics_path = next(tmp_path.glob("results/train_*/metrics.json"))
    run_id = metrics_path.parent.name
    before_mtime = metrics_path.stat().st_mtime

    proc_resume = subprocess.run(
        [
            sys.executable,
            "-m",
            "nircls.cli",
            "train",
            "--config",
            str(cfg_dst),
            "--resume",
            "--run-id",
            run_id,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc_resume.returncode == 0, proc_resume.stderr
    assert metrics_path.stat().st_mtime == before_mtime


def test_train_cli_regression(tmp_path: pathlib.Path) -> None:
    env = _base_env()
    data = pd.DataFrame(
        {
            "0": [0.1, 0.2, 0.3, 0.4],
            "1": [0.5, 0.6, 0.7, 0.8],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    (tmp_path / "reg.csv").write_text(data.to_csv(index=False))
    cfg_dst = tmp_path / "reg.yaml"
    cfg_dst.write_text(
        textwrap.dedent(
            """
            data:
              input_path: reg.csv
              target_cols: [target]
              target_type: regression
              out_dir: results
            preprocess:
              steps: [SNV]
            models:
              - name: PLSRegression
                grid: {n_components: [2]}
            cv:
              strategies:
                - type: kfold
                  n_splits: [2]
            """
        ).strip()
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nircls.cli",
            "--no-progress",
            "train",
            "--config",
            str(cfg_dst),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    metrics_files = list(tmp_path.glob("results/train_*/metrics.json"))
    assert metrics_files
    metrics_path = metrics_files[0]
    with open(metrics_path) as f:
        rows = json.load(f)
    assert rows
    assert "RMSE_mean" in rows[0]
    assert "RMSECV_mean" in rows[0]
    assert "RMSEP" in rows[0]
    assert "RPD" in rows[0]
    assert "MAE_percent_mean" in rows[0]
    assert "Accuracy_mean" not in rows[0]
    train_dir = metrics_path.parent
    assert not (train_dir / "classes.json").exists()
    assert not (train_dir / "astm_classification.json").exists()


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


def test_run_all_cli_can_rename_results_directory(tmp_path: pathlib.Path) -> None:
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
              steps: [SNV]
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
            "--no-progress",
            "run-all",
            "--config",
            str(cfg_dst),
            "--output-name",
            "experimento final",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    renamed_dir = tmp_path / "results" / "experimento_final"
    assert renamed_dir.is_dir()
    train_metrics = renamed_dir / "train" / "metrics.json"
    assert train_metrics.exists()
    state_path = renamed_dir / "run_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["run_id"] == "experimento_final"
    assert state["status"] == "completed"
    summary_dir = renamed_dir / "summary"
    assert summary_dir.exists()


def test_run_all_cli_persists_state_on_failure(tmp_path: pathlib.Path) -> None:
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
            models:
              - name: UnknownModel
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

    assert proc.returncode == 1
    results_dirs = list((tmp_path / "results").iterdir())
    assert results_dirs, proc.stderr
    state_path = results_dirs[0] / "run_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["status"] == "train_failed"
    assert "UnknownModel" in state.get("last_error", "")
    assert "--resume" in proc.stderr


def test_summary_cli_aggregates_results(tmp_path: pathlib.Path) -> None:
    env = _base_env()
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()
    (run1 / "metrics.json").write_text(
        json.dumps(
            [
                {
                    "target": "t",
                    "model": "M",
                    "accuracy_mean": 0.9,
                    "preprocess_steps": ["SNV"],
                    "result": "snv",
                }
            ]
        )
    )
    (run2 / "metrics.json").write_text(
        json.dumps(
            [
                {
                    "target": "t",
                    "model": "M",
                    "accuracy_mean": 0.8,
                    "preprocess_steps": ["MSC"],
                    "result": "msc",
                }
            ]
        )
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
    assert "preprocess_steps" in df.columns
    assert "result" in df.columns
    assert set(df["result"].str.strip()) == {"snv", "msc"}
    preprocess_values = set(df["preprocess_steps"].str.strip())
    assert preprocess_values == {
        "Standard Normal Variate",
        "Multiplicative Scatter Correction",
    }


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

    assert steps == ["SNV", "SG", "MSC", "EPO"]
