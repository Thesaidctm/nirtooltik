
import sys, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from joblib import dump

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.pipeline import (
    build_pipeline_from_config,
    run_multiple_eval,
    train_and_save,
    predict_with_model,
)
from nircls.io import prepare_inputs


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
            }
        ]
    }
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "MLP"
    assert "clf__hidden_layer_sizes" in grid
    assert "clf__alpha" in grid
    gs = GridSearchCV(pipe, grid, cv=2)
    gs.fit(X, y)
    assert gs.best_params_["clf__hidden_layer_sizes"] in [(5,), (10,)]
    assert gs.best_params_["clf__alpha"] in [0.0001, 0.001]


def test_decision_tree_with_grid():
    X = np.random.RandomState(0).randn(20, 5)
    y = (np.arange(20) % 2).astype(int)
    cfg = {
        "models": [
            {
                "name": "DT",
                "grid": {"max_depth": [None, 3], "min_samples_split": [2, 4]},
            }
        ]
    }
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "DT"
    assert "clf__max_depth" in grid
    assert "clf__min_samples_split" in grid
    gs = GridSearchCV(pipe, grid, cv=2)
    gs.fit(X, y)
    assert gs.best_params_["clf__max_depth"] in [None, 3]
    assert gs.best_params_["clf__min_samples_split"] in [2, 4]

def test_extratrees_model_with_grid():
    X = np.random.RandomState(0).randn(20, 5)
    y = (np.arange(20) % 2).astype(int)
    cfg = {
        "models": [
            {
                "name": "ET",
                "grid": {"n_estimators": [10, 20], "max_depth": [None, 5]},
            }
        ]
    }
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "ET"
    assert "clf__n_estimators" in grid
    assert "clf__max_depth" in grid
    cv = StratifiedKFold(n_splits=2)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy")
    gs.fit(X, y)
    assert gs.best_params_["clf__n_estimators"] in [10, 20]
    assert gs.best_params_["clf__max_depth"] in [None, 5]

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


def test_estimator_without_proba_is_calibrated(monkeypatch):
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.calibration import CalibratedClassifierCV

    class DummyClf(BaseEstimator, ClassifierMixin):
        def __init__(self, constant=0):
            self.constant = constant

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def get_params(self, deep=True):
            return {"constant": self.constant}

        def set_params(self, **params):
            self.constant = params.get("constant", self.constant)
            return self

    def fake_factory(models_cfg):
        return [
            {
                "name": "DUMMY",
                "estimator": DummyClf(),
                "grid": {"constant": [0, 1]},
            }
        ]

    monkeypatch.setattr("nircls.pipeline.make_models_from_config", fake_factory)
    cfg = {"models": [{"name": "DUMMY"}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert isinstance(pipe.named_steps["clf"], CalibratedClassifierCV)
    assert "clf__estimator__constant" in grid


def test_end_to_end_cv_and_export(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.rand(20, 6)
    y = np.array(["A"] * 10 + ["B"] * 10)
    df = pd.DataFrame(X, columns=[str(i) for i in range(6)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "data": {"input_path": str(data_file), "target_cols": ["target"]},
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
    metrics_path = strat_dir / "metrics_summary.xlsx"
    assert metrics_path.exists()
    xls = pd.ExcelFile(metrics_path)
    assert set(xls.sheet_names) == {"validation", "calibration"}
    assert any((strat_dir / "figures").glob("*.png"))
    assert (strat_dir / "report.md").exists()

    train_dir = tmp_path / "train_run"
    train_dir.mkdir()
    train_and_save(cfg, train_dir)
    assert (train_dir / "model.joblib").exists()
    assert (train_dir / "prepared_XY.xlsx").exists()
    assert (train_dir / "model_card.md").exists()



def test_prepare_inputs_with_multiple_targets(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(10, 4)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["t1"] = ["A"] * 10
    df["t2"] = ["B"] * 10
    data_file = tmp_path / "multi.csv"
    df.to_csv(data_file, index=False)
    cfg = {"input_path": str(data_file), "target_cols": "all"}
    ds_list = prepare_inputs(cfg)
    assert len(ds_list) == 2
    assert {d["target"] for d in ds_list} == {"t1", "t2"}

def test_nested_crossval_report(tmp_path):
    rng = np.random.RandomState(2)
    X = rng.rand(20, 4)
    y = np.array(["A"] * 10 + ["B"] * 10)
    df = pd.DataFrame(X, columns=[str(i) for i in range(4)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "data": {"input_path": str(data_file), "target_col": "target"},
        "models": [{"name": "PLSDA", "grid": {"n_components": [2]}}],
        "cv": {
            "inner_folds": 2,
            "random_state": 0,
            "strategies": [
                {"type": "kfold", "n_splits": [2], "n_repeats": 1}
            ],
        },
    }

    out_dir = tmp_path / "nested"
    run_multiple_eval(cfg, out_dir)
    strat_dir = out_dir / "kfold_2"
    metrics_path = strat_dir / "metrics_summary.xlsx"
    assert metrics_path.exists()
    xls = pd.ExcelFile(metrics_path)
    assert set(xls.sheet_names) == {"validation", "calibration"}
    assert any((strat_dir / "figures").glob("*.png"))
    assert (strat_dir / "report.md").exists()


def test_predict_with_model_outputs_y_pred_only_when_y_missing(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(10, 4)
    y = (np.arange(10) % 2)
    model = LogisticRegression(max_iter=50).fit(X, y)
    model_path = tmp_path / "model.joblib"
    dump(model, model_path)

    df = pd.DataFrame(X, columns=[str(i) for i in range(4)])
    data_file = tmp_path / "x_only.xlsx"
    with pd.ExcelWriter(data_file) as writer:
        df.to_excel(writer, sheet_name="X", index=False)

    out_dir = tmp_path / "pred"
    predict_with_model(model_path, data_file, out_dir)
    preds = pd.read_csv(out_dir / "predictions.csv")
    assert "y_pred" in preds.columns and "y_true" not in preds.columns
    assert len(preds) == len(X)


def test_predict_with_model_includes_y_true_when_available(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.rand(8, 3)
    y = (np.arange(8) % 2).astype(str)
    model = LogisticRegression(max_iter=50).fit(X, (np.arange(8) % 2))
    model_path = tmp_path / "model.joblib"
    dump(model, model_path)

    Xdf = pd.DataFrame(X, columns=[str(i) for i in range(3)])
    Ydf = pd.DataFrame({"target": y})
    data_file = tmp_path / "xy.xlsx"
    with pd.ExcelWriter(data_file) as writer:
        Xdf.to_excel(writer, sheet_name="X", index=False)
        Ydf.to_excel(writer, sheet_name="Y", index=False)

    out_dir = tmp_path / "pred_xy"
    predict_with_model(model_path, data_file, out_dir)
    preds = pd.read_csv(out_dir / "predictions.csv")
    assert list(preds.columns) == ["y_true", "y_pred"]
    assert len(preds) == len(X)

