from pathlib import Path
import json

import numpy as np

from nircls.report import generate_report, generate_scientific_report


def test_generate_scientific_report_handles_missing_metrics(tmp_path: Path) -> None:
    metrics = [
        {"model": "good", "F1-macro_mean": 0.5, "F1-macro_std": 0.1},
        {"model": "none", "F1-macro_mean": None, "F1-macro_std": None},
        {"model": "nan", "F1-macro_mean": np.nan, "F1-macro_std": np.nan},
    ]
    cfg = {"models": [{"name": "good"}]}

    report_path = generate_scientific_report(metrics, cfg, tmp_path)

    assert report_path.exists()
    content = report_path.read_text()
    assert "| good | 0.500 ± 0.100 |" in content
    assert "| none | N/A |" in content
    assert "| nan | N/A |" in content


def test_generate_report_handles_mean_only_metrics(tmp_path: Path) -> None:
    metrics = [
        {"model": "good", "Accuracy_mean": 0.75},
        {"model": "none", "Accuracy_mean": None},
        {"model": "nan", "Accuracy_mean": np.nan},
    ]
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    figs = tmp_path / "figures"
    figs.mkdir()
    out_file = tmp_path / "report.md"

    generate_report(metrics_path, figs, out_file)

    content = out_file.read_text()
    assert "| good | 0.750 |" in content
    assert "| none | N/A |" in content
    assert "| nan | N/A |" in content
