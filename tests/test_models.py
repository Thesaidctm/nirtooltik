import logging
import pathlib
import sys

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.models.plsda import PLSDA  # noqa: E402
from nircls.pipeline import build_pipeline_from_config  # noqa: E402


def test_logreg_default_grid_contains_C_and_balanced_class_weight():
    cfg = {"models": [{"name": "LOGREG"}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "LOGREG"
    assert "clf__C" in grid
    assert pipe.named_steps["clf"].max_iter == 5000
    assert pipe.named_steps["clf"].class_weight == "balanced"


def test_svm_defaults_to_balanced_class_weight():
    cfg = {"models": [{"name": "SVM_RBF"}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "SVM_RBF"
    assert pipe.named_steps["clf"].class_weight == "balanced"


def test_rf_defaults_to_balanced_class_weight():
    cfg = {"models": [{"name": "RF"}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "RF"
    assert pipe.named_steps["clf"].class_weight == "balanced"


def test_parallelism_flag_propagates_to_estimators():
    cfg = {"models": [{"name": "RF"}]}
    name, pipe, _ = build_pipeline_from_config(cfg, n_jobs=3)[0]
    assert name == "RF"
    assert pipe.named_steps["clf"].n_jobs == 3


def test_extratrees_defaults_to_balanced_class_weight():
    cfg = {"models": [{"name": "ET"}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "ET"
    assert pipe.named_steps["clf"].class_weight == "balanced"


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
                    "early_stopping": [True],
                    "validation_fraction": [0.2],
                },
            }
        ]
    }
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "MLP"
    assert "clf__hidden_layer_sizes" in grid
    assert "clf__alpha" in grid
    assert grid["clf__early_stopping"] == [True]
    assert grid["clf__validation_fraction"] == [0.2]
    gs = GridSearchCV(pipe, grid, cv=2, n_jobs=1)
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
    gs = GridSearchCV(pipe, grid, cv=2, n_jobs=1)
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
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=1)
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
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=1)
    gs.fit(X, y)


def test_qda_accepts_params_and_grid():
    X = np.random.RandomState(0).randn(20, 5)
    y = (np.arange(20) % 2).astype(int)
    cfg = {
        "preprocess": {"steps": []},
        "models": [
            {
                "name": "QDA",
                "params": {"reg_param": 0.1},
                "grid": {"reg_param": [0.0, 0.1]},
            }
        ],
    }
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "QDA"
    assert pipe.named_steps["clf"].reg_param == 0.1
    assert "clf__reg_param" in grid
    gs = GridSearchCV(pipe, grid, cv=2, n_jobs=1)
    gs.fit(X, y)
    assert gs.best_params_["clf__reg_param"] in [0.0, 0.1]


def test_qda_model_with_default_grid_includes_regularization():
    X = np.random.RandomState(0).randn(20, 5)
    y = (np.arange(20) % 2).astype(int)
    cfg = {"preprocess": {"steps": []}, "models": [{"name": "QDA"}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "QDA"
    assert "clf__reg_param" in grid
    assert grid["clf__reg_param"] == [0.0, 0.1]
    gs = GridSearchCV(pipe, grid, cv=2, n_jobs=1)
    gs.fit(X, y)
    assert gs.best_params_["clf__reg_param"] in [0.0, 0.1]


def test_gaussian_nb_model_with_default_grid():
    X = np.random.RandomState(0).randn(20, 5)
    y = (np.arange(20) % 2).astype(int)
    cfg = {"models": [{"name": "NB"}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "NB"
    assert isinstance(pipe.named_steps["clf"], GaussianNB)
    assert "clf__var_smoothing" in grid
    gs = GridSearchCV(pipe, grid, cv=2, n_jobs=1)
    gs.fit(X, y)
    assert gs.best_params_["clf__var_smoothing"] in [1e-09]


def test_gaussian_nb_accepts_string_var_smoothing():
    X = np.random.RandomState(0).randn(20, 5)
    y = (np.arange(20) % 2).astype(int)
    cfg = {"models": [{"name": "NB", "grid": {"var_smoothing": ["1e-09"]}}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "NB"
    assert grid["clf__var_smoothing"] == [1e-09]
    gs = GridSearchCV(pipe, grid, cv=2, n_jobs=1)
    gs.fit(X, y)
    assert gs.best_params_["clf__var_smoothing"] in [1e-09]


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


@pytest.mark.parametrize("method", ["logreg", "lda"])
def test_plsda_warns_when_components_high(caplog, method):
    rng = np.random.RandomState(0)
    X = rng.rand(10, 20)
    y = (np.arange(10) % 2).astype(int)
    model = PLSDA(n_components=8, method=method)
    with caplog.at_level(logging.WARNING):
        model.fit(X, y)
    assert any("overfit" in r.message.lower() for r in caplog.records)


def test_plsda_requires_1d_y():
    rng = np.random.RandomState(0)
    X = rng.rand(6, 4)
    y_1d = (np.arange(6) % 2).astype(int)
    model = PLSDA(n_components=2)
    # Should fit without error for 1-D y
    model.fit(X, y_1d)

    y_2d = y_1d.reshape(-1, 1)
    with pytest.raises(
        ValueError, match="PLSDA expects a 1-D array of class labels"
    ):
        model.fit(X, y_2d)


def test_plsda_default_grid_exposes_classifier_params():
    cfg = {"models": [{"name": "PLSDA"}]}
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert name == "PLSDA"
    assert "clf__C" in grid
    assert "clf__reg_param" in grid
    assert "clf__n_neighbors" in grid


def test_plsda_qda_predict_proba_and_grid():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    y = (np.arange(20) % 2).astype(int)
    model = PLSDA(method="qda", reg_param=0.1, n_components=2)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (20, 2)
    gs = GridSearchCV(
        PLSDA(method="qda"),
        {"n_components": [2], "reg_param": [0.0, 0.1]},
        cv=2,
        n_jobs=1,
    )
    gs.fit(X, y)
    assert gs.best_params_["reg_param"] in [0.0, 0.1]
