from pathlib import Path
import json

import numpy as np

from nircls.report import generate_report, generate_scientific_report


def test_generate_scientific_report_handles_missing_metrics(tmp_path: Path) -> None:
    metrics = [
        {
            "model": "good",
            "F1-macro_mean": 0.5,
            "F1-macro_std": 0.1,
            "F1-macro_ci95_low": 0.3,
            "F1-macro_ci95_high": 0.7,
        },
        {"model": "none", "F1-macro_mean": None, "F1-macro_std": None},
        {"model": "nan", "F1-macro_mean": np.nan, "F1-macro_std": np.nan},
    ]
    cfg = {"models": [{"name": "good"}]}

    report_path = generate_scientific_report(metrics, cfg, tmp_path)

    assert report_path.exists()
    content = report_path.read_text()
    assert "| good | 0.500 ± 0.100 (95% CI: 0.300–0.700) |" in content
    assert "| none | N/A |" in content
    assert "| nan | N/A |" in content
    assert "Metric definitions" in content
    assert "RMSECV" in content


def test_generate_report_handles_mean_only_metrics(tmp_path: Path) -> None:
    metrics = [
        {
            "model": "good",
            "Accuracy_mean": 0.75,
            "Accuracy_ci95_low": 0.55,
            "Accuracy_ci95_high": 0.95,
        },
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
    assert "| good | 0.750 (95% CI: 0.550–0.950) |" in content
    assert "| none | N/A |" in content
    assert "| nan | N/A |" in content
    assert "Metric definitions" in content
    assert "RMSEP" in content
