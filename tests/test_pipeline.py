import sys
import pathlib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from joblib import dump
import json

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from nircls.pipeline import (
    build_pipeline_from_config,
    run_multiple_eval,
    train_and_save,
    predict_with_model,
)
from nircls.io import prepare_inputs, summarize_dataset
from nircls.feature_selection.vip import compute_vip
from nircls.utils.errors import DataError, ModelError


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
    assert (train_dir / "classes.json").exists()


def test_pipeline_runs_with_spxy_cv_method(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(12, 4)
    y = np.array(["A"] * 6 + ["B"] * 6)
    df = pd.DataFrame(X, columns=[str(i) for i in range(4)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "data": {"input_path": str(data_file), "target_cols": ["target"]},
        "models": [{"name": "PLSDA", "grid": {"n_components": [2]}}],
        "cv": {
            "cv_method": "spxy",
            "random_state": 0,
            "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
        },
    }

    out_dir = tmp_path / "cv_spxy"
    run_multiple_eval(cfg, out_dir)
    assert (out_dir / "kfold_2" / "metrics_summary.xlsx").exists()


def test_pipeline_runs_with_group_cv_method(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(12, 4)
    y = np.array(["A"] * 6 + ["B"] * 6)
    groups = np.repeat([0, 1, 2, 3, 4, 5], 2)
    df = pd.DataFrame(X, columns=[str(i) for i in range(4)])
    df["target"] = y
    df["group"] = groups
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "data": {
            "input_path": str(data_file),
            "target_cols": ["target"],
            "group_col": "group",
        },
        "models": [{"name": "PLSDA", "grid": {"n_components": [2]}}],
        "cv": {
            "cv_method": "group",
            "random_state": 0,
            "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
        },
    }

    out_dir = tmp_path / "cv_group"
    run_multiple_eval(cfg, out_dir)
    assert (out_dir / "kfold_2" / "metrics_summary.xlsx").exists()


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


def test_summarize_dataset_columns(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(5, 3)
    y = np.array(["A"] * 3 + ["B"] * 2)
    df = pd.DataFrame(X, columns=[str(i) for i in range(3)])
    df["target"] = y
    df.loc[0, "0"] = np.nan
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)
    cfg = {"input_path": str(data_file), "target_cols": ["target"]}
    ds_list = prepare_inputs(cfg)
    summary = summarize_dataset(ds_list)
    assert set(summary.columns) == {
        "target",
        "class",
        "count",
        "spectral_min",
        "spectral_max",
        "n_nans",
    }


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
    y_labels = np.array(["A", "B", "A", "B", "A", "B", "A", "B"])
    model = LogisticRegression(max_iter=50).fit(X, (np.arange(8) % 2))
    model_path = tmp_path / "model.joblib"
    dump(model, model_path)
    with open(model_path.with_name("classes.json"), "w") as f:
        json.dump(["A", "B"], f)

    Xdf = pd.DataFrame(X, columns=[str(i) for i in range(3)])
    Ydf = pd.DataFrame({"target": y_labels})
    data_file = tmp_path / "xy.xlsx"
    with pd.ExcelWriter(data_file) as writer:
        Xdf.to_excel(writer, sheet_name="X", index=False)
        Ydf.to_excel(writer, sheet_name="Y", index=False)

    out_dir = tmp_path / "pred_xy"
    predict_with_model(model_path, data_file, out_dir)
    preds = pd.read_csv(out_dir / "predictions.csv")
    assert list(preds.columns) == ["y_true", "y_pred"]
    assert set(preds["y_pred"].unique()) <= {"A", "B"}
    assert len(preds) == len(X)


def test_run_multiple_eval_requires_cv(tmp_path):
    cfg = {
        "data": {"input_path": str(tmp_path / "data.csv"), "target_cols": ["t"]},
        "models": [{"name": "PLSDA", "grid": {"n_components": [2]}}],
    }
    with pytest.raises(ModelError):
        run_multiple_eval(cfg, tmp_path)


def test_run_multiple_eval_file_not_found(tmp_path):
    cfg = {
        "data": {"input_path": str(tmp_path / "missing.csv"), "target_cols": ["t"]},
        "models": [{"name": "PLSDA"}],
        "cv": {"strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}]},
    }
    with pytest.raises(DataError, match="Arquivo de dados não encontrado"):
        run_multiple_eval(cfg, tmp_path)


def test_compute_vip_identifies_relevant_feature():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 5)
    y = X[:, 0] * 0.5 + rng.randn(40) * 0.01
    pls = PLSRegression(n_components=1).fit(X, y)
    vip = compute_vip(pls, X, y[:, None])
    assert vip.shape == (5,)
    assert vip[0] == pytest.approx(vip.max())
    assert np.all(vip > 0)


def _dummy_cfg():
    return {
        "data": {"input_path": "dummy", "target_cols": ["t"]},
        "models": [{"name": "PLSDA"}],
    }


def test_train_and_save_file_not_found(monkeypatch, tmp_path):
    cfg = _dummy_cfg()

    def raise_data_error(_):
        raise DataError("Arquivo de dados não encontrado")

    monkeypatch.setattr("nircls.pipeline.prepare_inputs", raise_data_error)
    with pytest.raises(DataError, match="Arquivo de dados não encontrado"):
        train_and_save(cfg, tmp_path)


def test_train_and_save_missing_data_error(monkeypatch, tmp_path):
    cfg = _dummy_cfg()

    def raise_data_error(_):
        raise DataError("dados inexistentes na planilha")

    monkeypatch.setattr("nircls.pipeline.prepare_inputs", raise_data_error)
    with pytest.raises(DataError, match="dados inexistentes"):
        train_and_save(cfg, tmp_path)


def test_train_and_save_model_incompatible(monkeypatch, tmp_path):
    cfg = _dummy_cfg()

    def dummy_prepare(_):
        return [{"X": pd.DataFrame([[0]]), "Y": pd.Series([0]), "file": "f", "target": "t"}]

    def raise_model_error(_):
        raise ModelError("modelo incompatível")

    monkeypatch.setattr("nircls.pipeline.prepare_inputs", dummy_prepare)
    monkeypatch.setattr("nircls.pipeline.build_pipeline_from_config", raise_model_error)
    with pytest.raises(ModelError, match="modelo incompatível"):
        train_and_save(cfg, tmp_path)
