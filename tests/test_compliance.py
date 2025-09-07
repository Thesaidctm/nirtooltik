import pandas as pd
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.utils.compliance import check_astm_classification


def test_check_astm_classification_flags():
    summary = pd.DataFrame(
        {
            "target": ["t", "t", "t"],
            "class": ["A", "B", "C"],
            "count": [15, 10, 5],
        }
    )
    flags = check_astm_classification(summary, {"external_validation": True})
    assert flags == {
        "min_samples": True,
        "min_per_class": False,
        "external_validation": True,
    }
