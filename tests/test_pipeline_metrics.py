"""Regression tests for metrics persistence helpers."""

from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nircls.processing.pipeline import _coerce_param_value, _save_metrics


def _make_basic_row() -> dict:
    return {
        "file": "dataset.csv",
        "target": "target",
        "model": "Model",
        "preprocess_steps": ["RAW"],
        "result": "RAW",
        "calibrated": False,
    }


def test_save_metrics_handles_best_param_sequences(tmp_path: Path) -> None:
    row = _make_basic_row()
    row["best_hidden_layer_sizes"] = _coerce_param_value([100, 50])
    assert isinstance(row["best_hidden_layer_sizes"], tuple)

    rows = [row]

    _save_metrics(rows, tmp_path)

    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()

    data_first = json.loads(metrics_path.read_text())
    assert data_first == [row | {"best_hidden_layer_sizes": [100, 50]}]

    _save_metrics(rows, tmp_path)

    data_second = json.loads(metrics_path.read_text())
    assert data_second == data_first
