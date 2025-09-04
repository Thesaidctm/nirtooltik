
import sys, pathlib, numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.pipeline import build_pipeline_from_config


def test_pipeline_fit_predict_with_cv_and_windows():
    X = np.random.RandomState(0).randn(30, 60)
    y = (np.arange(30) % 2).astype(int)
    cfg = {
        "preprocess": {
            "steps": ["SNV", "SG"],
            "sg": {"window": [5, 7], "polyorder": 2, "deriv": 1},
        },
        "models": [{"name": "PLSDA", "grid": {"n_components": [2]}}],
    }
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert "sg__window_length" in grid
    cv = StratifiedKFold(n_splits=3)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy")
    gs.fit(X, y)
