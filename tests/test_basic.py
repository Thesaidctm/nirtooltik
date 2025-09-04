
import sys, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.pipeline import (
    build_pipeline_from_config,
    run_multiple_eval,
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



def test_logreg_default_grid_contains_C():
    cfg = {"models": [{"name": "LOGREG"}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "LOGREG"
    assert "clf__C" in grid
    assert pipe.named_steps["clf"].max_iter == 5000
def test_mlp_accepts_hidden_layer_and_alpha():
    X = np.random.RandomState(0).randn(20, 10)
def test_decision_tree_with_grid():
    X = np.random.RandomState(0).randn(20, 5)
    y = (np.arange(20) % 2).astype(int)
    cfg = {
        "models": [
            {
                "name": "MLP",
                "grid": {
                    "hidden_layer_sizes": [(5,), (10,)],
                    "alpha": [0.0001, 0.001],
                    "max_iter": [50],
                },
            },
            {
                "name": "DT",
                "grid": {"max_depth": [None, 3], "min_samples_split": [2, 4]},
            },
        ]
    }
    name1, pipe1, grid1 = build_pipeline_from_config(cfg)[0]
    assert "clf__hidden_layer_sizes" in grid1
    assert "clf__alpha" in grid1

    name2, pipe2, grid2 = build_pipeline_from_config(cfg)[1]
    assert name2 == "DT"
    assert "clf__max_depth" in grid2
    assert "clf__min_samples_split" in grid2
    gs = GridSearchCV(pipe2, grid2, cv=2)
def test_knn_model_with_grid():
    X = np.random.RandomState(1).randn(20, 5)
    y = (np.arange(20) % 2).astype(int)
    cfg = {
        "preprocess": {"steps": []},
        "models": [{"name": "KNN", "grid": {"n_neighbors": [3, 5]}}],
    }
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert "clf__n_neighbors" in grid

    cv = StratifiedKFold(n_splits=2)
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
        "cv": {
            "random_state": 0,
            "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
        },
    }

    out_dir = tmp_path / "cv_run"
    run_multiple_eval(cfg, out_dir)
    strat_dir = out_dir / "kfold_2"
    assert (strat_dir / "metrics_summary.csv").exists()
    assert any((strat_dir / "figures").glob("*.png"))

    train_dir = tmp_path / "train_run"
    train_dir.mkdir()
    train_and_save(cfg, train_dir)
    assert (train_dir / "model.joblib").exists()
    assert (train_dir / "prepared_XY.xlsx").exists()
    assert (train_dir / "model_card.md").exists()
