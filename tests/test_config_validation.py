import pytest
from nircls.pipeline import run_multiple_eval


def test_run_multiple_eval_requires_cv(tmp_path):
    cfg = {
        "data": {"input_path": str(tmp_path / "data.csv"), "target_cols": ["t"]},
        "models": [{"name": "PLSDA", "grid": {"n_components": [2]}}],
    }
    with pytest.raises(ValueError):
        run_multiple_eval(cfg, tmp_path)
