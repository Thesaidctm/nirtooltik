import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest
from joblib import dump
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402
from nircls.feature_selection.vip import compute_vip  # noqa: E402
from nircls.io import prepare_inputs, summarize_dataset  # noqa: E402
from nircls.pipeline import (  # noqa: E402
    build_pipeline_from_config,
    predict_with_model,
    run_multiple_eval,
    train_and_save,
    _format_preprocess_step,
    _save_pls_diagnostics,
)
from nircls.models.plsda import PLSDA  # noqa: E402
from nircls.utils.errors import DataError, ModelError  # noqa: E402
from nircls.utils.palettes import Palette  # noqa: E402

METHODS = ["logreg", "lda"]


def _sanitize_step_label(label: str) -> str:
    return re.sub(r"[^\w.-]+", "_", label.lower()).strip("_")


def test_save_pls_diagnostics_includes_y_loadings(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(12, 4), columns=[f"f{i}" for i in range(4)])
    y = np.array([0, 1] * 6)
    classes = np.array(["class0", "class1"])
    template_pipe = Pipeline([("clf", PLSDA(n_components=2, method="logreg"))])

    def fake_export(fig, out_base, interactive):
        out_base.parent.mkdir(parents=True, exist_ok=True)
        out_base.with_suffix(".png").write_text("png")
        if interactive:
            out_base.with_suffix(".html").write_text("<html></html>")

    monkeypatch.setattr("nircls.utils.plots._export", fake_export)

    artifacts = _save_pls_diagnostics(
        template_pipe=template_pipe,
        best_params=None,
        X=X,
        y=y,
        classes=classes,
        file_path="dataset.csv",
        target="target",
        model="PLSDA",
        preprocess_steps=None,
        out_dir=tmp_path,
        interactive=True,
        palette=Palette.VIRIDIS,
    )

    assert "pls_y_loadings_csv" in artifacts
    assert "pls_y_loadings_png" in artifacts
    assert "pls_y_loadings_html" in artifacts

    y_loadings_csv = tmp_path / artifacts["pls_y_loadings_csv"]
    assert y_loadings_csv.exists()
    y_loadings_df = pd.read_csv(y_loadings_csv)
    assert not y_loadings_df.empty
    assert set(y_loadings_df["feature"]) == {"class0", "class1"}
    assert set(y_loadings_df["component"]) == {"Comp 1", "Comp 2"}
    assert set(y_loadings_df["file"]) == {"dataset.csv"}
    assert set(y_loadings_df["target"]) == {"target"}
    assert set(y_loadings_df["model"]) == {"PLSDA"}
    assert set(y_loadings_df["preprocess_steps"].fillna("None")) == {"None"}

    png_path = tmp_path / artifacts["pls_y_loadings_png"]
    html_path = tmp_path / artifacts["pls_y_loadings_html"]
    assert png_path.exists()
    assert html_path.exists()


@pytest.mark.parametrize("method", METHODS)
def test_pipeline_fit_predict_with_cv_and_windows(method):
    X = np.random.RandomState(0).randn(30, 60)
    y = (np.arange(30) % 2).astype(int)
    cfg = {
        "preprocess": {
            "steps": ["SNV", "SG"],
            "evaluate_individual": False,
            "sg": {"window": [5, 7], "polyorder": 2, "deriv": 1},
        },
        "models": [
            {
                "name": "PLSDA",
                "grid": {"n_components": [2], "method": [method]},
                "overfit_check": True,
            }
        ],
    }
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    assert "sg__window_length" in grid
    cv = StratifiedKFold(n_splits=3)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=1)
    gs.fit(X, y)


@pytest.mark.parametrize("method", METHODS)
def test_end_to_end_cv_and_export(tmp_path, method):
    rng = np.random.RandomState(1)
    X = rng.rand(20, 6)
    y = np.array(["A"] * 10 + ["B"] * 10)
    df = pd.DataFrame(X, columns=[str(i) for i in range(6)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "data": {"input_path": str(data_file), "target_cols": ["target"]},
        "preprocess": {
            "steps": ["SNV", "SG"],
            "sg": {"window": 5, "polyorder": 2, "deriv": 1},
        },
        "models": [
            {
                "name": "PLSDA",
                "grid": {"n_components": [2], "method": [method]},
                "overfit_check": True,
            }
        ],
        "cv": {
            "random_state": 0,
            "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
        },
        "astm": True,
    }

    out_dir = tmp_path / "cv_run"
    outputs = run_multiple_eval(cfg, out_dir, n_jobs=1)
    expected_steps = {"SNV", "SG(window=5,poly=2,deriv=1)"}
    sanitized_steps = {
        _sanitize_step_label(_format_preprocess_step(step) or str(step))
        for step in cfg["preprocess"]["steps"]
    }
    expected_dirs = {f"kfold_2__{label}" for label in sanitized_steps}
    assert len(outputs) == len(expected_dirs)
    observed_dirs = {metrics_file.parent.name for metrics_file, _ in outputs}
    assert observed_dirs == expected_dirs
    observed_labels = set()
    for metrics_file, figs_dir in outputs:
        strat_dir = metrics_file.parent
        metrics_path = strat_dir / "metrics_summary.xlsx"
        assert metrics_path.exists()
        xls = pd.ExcelFile(metrics_path)
        assert set(xls.sheet_names) == {"validation", "calibration"}
        val_df = xls.parse("validation")
        calib_df = xls.parse("calibration")
        assert {"Yscramble_mean", "Yscramble_p95"}.issubset(val_df.columns)
        assert len(set(val_df["preprocess_steps"])) == 1
        assert len(set(calib_df["preprocess_steps"])) == 1
        expected_method_label = f"PLS-{method.upper()}"
        observed_methods = {m for m in val_df.get("method", []) if isinstance(m, str)}
        assert expected_method_label in observed_methods
        assert any((strat_dir / "figures").glob("*.png"))
        assert (strat_dir / "report.md").exists()
        assert (strat_dir / "astm_classification.json").exists()
        tables_dir = strat_dir / "results"
        assert (tables_dir / "pls_loadings.csv").exists()
        assert (tables_dir / "pls_weights.csv").exists()
        assert (tables_dir / "vip.csv").exists()
        assert (tables_dir / "pls_x_scores.csv").exists()
        assert (tables_dir / "pls_y_scores.csv").exists()
        x_scores_df = pd.read_csv(tables_dir / "pls_x_scores.csv")
        y_scores_df = pd.read_csv(tables_dir / "pls_y_scores.csv")
        expected_score_cols = {
            "sample",
            "sample_index",
            "y_true",
            "component",
            "component_index",
            "value",
        }
        assert expected_score_cols.issubset(x_scores_df.columns)
        assert expected_score_cols.issubset(y_scores_df.columns)
        with open(metrics_file) as f:
            metrics_json = json.load(f)
        assert metrics_json, "metrics.json should contain at least one entry"
        assert all("Yscramble_mean" in row for row in metrics_json)
        metrics_by_step = {}
        for row in metrics_json:
            metrics_by_step.setdefault(row["preprocess_steps"], []).append(row)
        assert len(metrics_by_step) == 1
        observed_label = next(iter(metrics_by_step))
        observed_labels.add(observed_label)
        assert set(val_df["preprocess_steps"]) == {observed_label}
        assert set(calib_df["preprocess_steps"]) == {observed_label}
        for entry in metrics_by_step[observed_label]:
            assert entry["best_clf__n_components"] == 2
            assert entry["best_clf__method"] == method
            model_label = entry["model"].upper()
            assert model_label.startswith("PLS-")
            assert model_label.startswith(expected_method_label)
            assert entry["method"] == expected_method_label
    assert observed_labels == expected_steps

    train_dir = tmp_path / "train_run"
    train_dir.mkdir()
    train_and_save(cfg, train_dir, n_jobs=1)
    assert (train_dir / "model.joblib").exists()
    assert (train_dir / "prepared_XY.xlsx").exists()
    assert (train_dir / "model_card.md").exists()
    assert (train_dir / "classes.json").exists()
    assert (train_dir / "astm_classification.json").exists()
    train_tables = train_dir / "results"
    assert (train_tables / "pls_loadings.csv").exists()
    assert (train_tables / "pls_weights.csv").exists()
    assert (train_tables / "vip.csv").exists()
    assert (train_tables / "pls_x_scores.csv").exists()
    assert (train_tables / "pls_y_scores.csv").exists()


def test_metrics_reflect_applied_preprocess_steps(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(12, 6)
    y = np.array(["A"] * 6 + ["B"] * 6)
    df = pd.DataFrame(X, columns=[str(i) for i in range(6)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "data": {"input_path": str(data_file), "target_cols": ["target"]},
        "preprocess": {
            "steps": ["SNV", "SG"],
            "sg": {"window": 5, "polyorder": 2, "deriv": 1},
        },
        "models": [
            {"name": "PLSDA", "grid": {"n_components": [2], "method": ["logreg"]}}
        ],
        "cv": {
            "random_state": 0,
            "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
        },
    }

    out_dir = tmp_path / "cv_steps"
    outputs = run_multiple_eval(cfg, out_dir, n_jobs=1)
    expected_steps = {"SNV", "SG(window=5,poly=2,deriv=1)"}
    sanitized_steps = {
        _sanitize_step_label(_format_preprocess_step(step) or str(step))
        for step in cfg["preprocess"]["steps"]
    }
    expected_dirs = {f"kfold_2__{label}" for label in sanitized_steps}
    assert len(outputs) == len(expected_dirs)
    observed_dirs = {metrics_path.parent.name for metrics_path, _ in outputs}
    assert observed_dirs == expected_dirs
    observed_labels = set()
    for metrics_path, _ in outputs:
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert metrics
        observed = {row["preprocess_steps"] for row in metrics}
        assert len(observed) == 1
        label = observed.pop()
        observed_labels.add(label)
    assert observed_labels == expected_steps


@pytest.mark.parametrize("method", METHODS)
def test_pipeline_runs_with_spxy_cv_method(tmp_path, method):
    rng = np.random.RandomState(0)
    X = rng.rand(12, 4)
    y = np.array(["A"] * 6 + ["B"] * 6)
    df = pd.DataFrame(X, columns=[str(i) for i in range(4)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "data": {"input_path": str(data_file), "target_cols": ["target"]},
        "models": [
            {"name": "PLSDA", "grid": {"n_components": [2], "method": [method]}}
        ],
        "cv": {
            "cv_method": "spxy",
            "random_state": 0,
            "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
        },
    }

    out_dir = tmp_path / "cv_spxy"
    run_multiple_eval(cfg, out_dir, n_jobs=1)
    assert (out_dir / "kfold_2" / "metrics_summary.xlsx").exists()


@pytest.mark.parametrize("method", METHODS)
def test_pipeline_runs_with_group_cv_method(tmp_path, method):
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
        "models": [
            {"name": "PLSDA", "grid": {"n_components": [2], "method": [method]}}
        ],
        "cv": {
            "cv_method": "group",
            "random_state": 0,
            "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
        },
    }

    out_dir = tmp_path / "cv_group"
    run_multiple_eval(cfg, out_dir, n_jobs=1)
    assert (out_dir / "kfold_2" / "metrics_summary.xlsx").exists()


def test_prepare_inputs_with_multiple_targets(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(10, 4)
    df = pd.DataFrame(X, columns=[str(i) for i in range(4)])
    df["t1"] = ["A"] * 10
    df["t2"] = ["B"] * 10
    data_file = tmp_path / "multi.csv"
    df.to_csv(data_file, index=False)
    cfg = {"input_path": str(data_file), "target_cols": "all"}
    ds_list = prepare_inputs(cfg)
    assert len(ds_list) == 2
    assert {d["target"] for d in ds_list} == {"t1", "t2"}
    assert all(ds["Y"].dtype.name == "category" for ds in ds_list)


def test_prepare_inputs_supports_encoding_and_sep_decimal(tmp_path):
    df = pd.DataFrame({"0": [1.5, 2.5], "target": ["A", "B"]})
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False, sep=";", decimal=",", encoding="latin1")
    cfg = {
        "input_path": str(data_file),
        "target_cols": ["target"],
        "sep": ";",
        "decimal": ",",
        "encoding": "latin1",
    }
    ds_list = prepare_inputs(cfg)
    assert ds_list[0]["X"].iloc[0, 0] == pytest.approx(1.5)


def test_prepare_inputs_raises_when_range_unavailable(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(5, 3)
    df = pd.DataFrame(X, columns=[str(i) for i in range(3)])
    df["target"] = ["A"] * 5
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)
    cfg = {
        "input_path": str(data_file),
        "target_cols": ["target"],
        "spectral_range": (10, 20),
    }
    with pytest.raises(DataError, match="Intervalo espectral fora da faixa disponível"):
        prepare_inputs(cfg)


def test_prepare_inputs_raises_on_invalid_path(tmp_path):
    bad_dir = tmp_path / "data_dir"
    bad_dir.mkdir()
    cfg = {"input_path": str(bad_dir)}
    with pytest.raises(DataError, match="inválido"):
        prepare_inputs(cfg)


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
    assert ds_list[0]["Y"].dtype.name == "category"
    summary = summarize_dataset(ds_list)
    assert set(summary.columns) == {
        "target",
        "class",
        "count",
        "spectral_min",
        "spectral_max",
        "n_nans",
    }


@pytest.mark.parametrize("method", METHODS)
def test_nested_crossval_report(tmp_path, method):
    rng = np.random.RandomState(2)
    X = rng.rand(20, 4)
    y = np.array(["A"] * 10 + ["B"] * 10)
    df = pd.DataFrame(X, columns=[str(i) for i in range(4)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "data": {"input_path": str(data_file), "target_col": "target"},
        "models": [
            {"name": "PLSDA", "grid": {"n_components": [2], "method": [method]}}
        ],
        "cv": {
            "inner_folds": 2,
            "random_state": 0,
            "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
        },
    }

    out_dir = tmp_path / "nested"
    run_multiple_eval(cfg, out_dir, n_jobs=1)
    strat_dir = out_dir / "kfold_2"
    metrics_path = strat_dir / "metrics_summary.xlsx"
    assert metrics_path.exists()
    xls = pd.ExcelFile(metrics_path)
    assert set(xls.sheet_names) == {"validation", "calibration"}
    assert any((strat_dir / "figures").glob("*.png"))
    assert (strat_dir / "report.md").exists()


def _simple_dataset(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(20, 4)
    y = np.array(["A"] * 10 + ["B"] * 10)
    df = pd.DataFrame(X, columns=[str(i) for i in range(4)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)
    return data_file


def _simple_cfg(data_file, inner_folds_cfg):
    cv_cfg = {
        "random_state": 0,
        "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
    }
    if inner_folds_cfg != "missing":
        cv_cfg["inner_folds"] = inner_folds_cfg
    return {
        "data": {"input_path": str(data_file), "target_cols": ["target"]},
        "models": [
            {"name": "PLSDA", "grid": {"n_components": [2], "method": ["logreg"]}}
        ],
        "cv": cv_cfg,
    }


@pytest.mark.parametrize("inner_folds_cfg", ["missing", None])
def test_inner_folds_defaults_without_error(tmp_path, inner_folds_cfg):
    data_file = _simple_dataset(tmp_path)
    cfg = _simple_cfg(data_file, inner_folds_cfg)
    run_multiple_eval(cfg, tmp_path / "inner_default", n_jobs=1)


def test_inner_folds_too_small_raises(tmp_path):
    data_file = _simple_dataset(tmp_path)
    cfg = _simple_cfg(data_file, 1)
    with pytest.raises(ModelError, match="inner_folds"):
        run_multiple_eval(cfg, tmp_path / "inner_invalid", n_jobs=1)


def test_inner_folds_invalid_type_raises(tmp_path):
    data_file = _simple_dataset(tmp_path)
    cfg = _simple_cfg(data_file, "foo")
    with pytest.raises(DataError, match="inner_folds"):
        run_multiple_eval(cfg, tmp_path / "inner_invalid_type", n_jobs=1)


def test_predict_with_model_outputs_y_pred_only_when_y_missing(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(10, 4)
    y = np.arange(10) % 2
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


@pytest.mark.parametrize("method", METHODS)
def test_run_multiple_eval_requires_cv(tmp_path, method):
    cfg = {
        "data": {"input_path": str(tmp_path / "data.csv"), "target_cols": ["t"]},
        "models": [
            {"name": "PLSDA", "grid": {"n_components": [2], "method": [method]}}
        ],
    }
    with pytest.raises(ModelError):
        run_multiple_eval(cfg, tmp_path, n_jobs=1)


@pytest.mark.parametrize("method", METHODS)
def test_run_multiple_eval_file_not_found(tmp_path, method):
    cfg = {
        "data": {"input_path": str(tmp_path / "missing.csv"), "target_cols": ["t"]},
        "models": [{"name": "PLSDA", "grid": {"method": [method]}}],
        "cv": {"strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}]},
    }
    with pytest.raises(DataError, match="Arquivo de dados não encontrado"):
        run_multiple_eval(cfg, tmp_path, n_jobs=1)


def test_compute_vip_identifies_relevant_feature():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 5)
    y = X[:, 0] * 0.5 + rng.randn(40) * 0.01
    pls = PLSRegression(n_components=1).fit(X, y)
    vip = compute_vip(pls, X, y[:, None])
    assert vip.shape == (5,)
    assert vip[0] == pytest.approx(vip.max())
    assert np.all(vip > 0)


def _dummy_cfg(method):
    return {
        "data": {"input_path": "dummy", "target_cols": ["t"]},
        "models": [{"name": "PLSDA", "grid": {"method": [method]}}],
    }


@pytest.mark.parametrize("method", METHODS)
def test_train_and_save_file_not_found(monkeypatch, tmp_path, method):
    cfg = _dummy_cfg(method)

    def raise_data_error(_, **kwargs):
        raise DataError("Arquivo de dados não encontrado")

    monkeypatch.setattr("nircls.pipeline.prepare_inputs", raise_data_error)
    with pytest.raises(DataError, match="Arquivo de dados não encontrado"):
        train_and_save(cfg, tmp_path, n_jobs=1)


@pytest.mark.parametrize("method", METHODS)
def test_train_and_save_missing_data_error(monkeypatch, tmp_path, method):
    cfg = _dummy_cfg(method)

    def raise_data_error(_, **kwargs):
        raise DataError("dados inexistentes na planilha")

    monkeypatch.setattr("nircls.pipeline.prepare_inputs", raise_data_error)
    with pytest.raises(DataError, match="dados inexistentes"):
        train_and_save(cfg, tmp_path, n_jobs=1)


@pytest.mark.parametrize("method", METHODS)
def test_train_and_save_model_incompatible(monkeypatch, tmp_path, method):
    cfg = _dummy_cfg(method)

    def dummy_prepare(_, **kwargs):
        return [
            {"X": pd.DataFrame([[0]]), "Y": pd.Series([0]), "file": "f", "target": "t"}
        ]

    def raise_model_error(_):
        raise ModelError("modelo incompatível")

    monkeypatch.setattr("nircls.pipeline.prepare_inputs", dummy_prepare)
    monkeypatch.setattr("nircls.pipeline.build_pipeline_from_config", raise_model_error)
    with pytest.raises(ModelError, match="modelo incompatível"):
        train_and_save(cfg, tmp_path, n_jobs=1)


def test_run_multiple_eval_sends_notification_email(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.rand(12, 4)
    y = np.array(["A"] * 6 + ["B"] * 6)
    df = pd.DataFrame(X, columns=[str(i) for i in range(4)])
    df["target"] = y
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    cfg = {
        "notification_email": "user@example.com",
        "data": {"input_path": str(data_file), "target_cols": ["target"]},
        "models": [
            {"name": "PLSDA", "grid": {"n_components": [2], "method": ["logreg"]}}
        ],
        "cv": {
            "random_state": 0,
            "strategies": [{"type": "kfold", "n_splits": [2], "n_repeats": 1}],
        },
    }

    calls = []

    def fake_enviar_email(subject: str, body: str, to_addr: str) -> None:
        calls.append((subject, body, to_addr))

    monkeypatch.setattr(
        "nircls.pipeline.enviar_email", fake_enviar_email, raising=False
    )

    out_dir = tmp_path / "out"
    run_multiple_eval(cfg, out_dir, n_jobs=1)
    assert len(calls) == 2
    assert all(call[2] == "user@example.com" for call in calls)
