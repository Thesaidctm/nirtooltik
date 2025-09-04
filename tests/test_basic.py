
import sys, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.pipeline import (
    build_pipeline_from_config,
    nested_crossval_run,
    train_and_save,
)


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


def test_end_to_end_cv_and_export(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.rand(20, 6)
    y = np.array(["A"] * 10 + ["B"] * 10)
    df = pd.DataFrame(X, columns=[str(i) for i in range(6)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "data": {"input_path": str(data_file), "target_col": "target"},
        "preprocess": {"steps": ["SNV"]},
        "models": [{"name": "PLSDA", "grid": {"n_components": [2]}}],
        "cv": {"outer_folds": 2, "outer_repeats": 1, "inner_folds": 2},
    }

    out_dir = tmp_path / "cv_run"
    nested_crossval_run(cfg, out_dir, "r1")
    assert (out_dir / "metrics_summary.csv").exists()
    assert any((out_dir / "figures").glob("*.png"))

    train_dir = tmp_path / "train_run"
    train_dir.mkdir()
    train_and_save(cfg, train_dir)
    assert (train_dir / "model.joblib").exists()
    assert (train_dir / "prepared_XY.xlsx").exists()
    assert (train_dir / "model_card.md").exists()
