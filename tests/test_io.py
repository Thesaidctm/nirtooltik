from __future__ import annotations

from pathlib import Path
import pathlib
import sys

import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from nircls.io import _extract_targets_and_groups, prepare_inputs
from nircls.pipeline import prepare_inputs as pipeline_prepare_inputs
from nircls.utils.errors import DataError


def test_extract_targets_defaults_to_non_numeric_columns():
    df = pd.DataFrame(
        {
            "feat": [1.0, 2.0, 3.0],
            "target": ["a", "b", "c"],
            "group": ["g1", "g1", "g2"],
        }
    )
    df.attrs["_source_path"] = "dataset.csv"
    target_cols_list, y_list, groups = _extract_targets_and_groups(df, None, "group")

    assert target_cols_list == ["target"]
    assert len(y_list) == 1 and y_list[0][0] == "target"
    assert y_list[0][1].equals(pd.Series(["a", "b", "c"], dtype="category"))
    assert groups is not None
    assert list(groups) == ["g1", "g1", "g2"]
    assert str(groups.dtype) == "category"


def test_extract_targets_all_columns_when_numeric_targets():
    df = pd.DataFrame({"y1": [1, 2, 3], "y2": [4, 5, 6], "group": [10, 11, 12]})
    df.attrs["_source_path"] = Path("sample.xlsx")
    df.attrs["_default_all_target_cols"] = True
    target_cols_list, y_list, groups = _extract_targets_and_groups(df, None, "group")

    assert target_cols_list == ["y1", "y2"]
    assert [name for name, _ in y_list] == ["y1", "y2"]
    assert groups is not None and str(groups.dtype) == "category"


def test_extract_targets_raises_for_missing_target_column():
    df = pd.DataFrame({"existing": [1, 2, 3]})
    df.attrs["_source_path"] = Path("missing.csv")

    with pytest.raises(
        DataError, match="Coluna alvo inexistente não encontrada em missing.csv"
    ):
        _extract_targets_and_groups(df, "inexistente", None)


def test_extract_targets_raises_for_missing_group_column():
    df = pd.DataFrame({"target": ["a", "b"], "other": [1, 2]})
    df.attrs["_source_path"] = "group.csv"

    with pytest.raises(
        DataError, match="Coluna de grupos grupo não encontrada em group.csv"
    ):
        _extract_targets_and_groups(df, None, "grupo")


def test_extract_targets_regression_requires_explicit_columns_when_ambiguous():
    df = pd.DataFrame(
        {"feat": [1.0, 2.0, 3.0], "target": [0.5, 0.7, 0.9], "group": ["g1", "g1", "g2"]}
    )
    df.attrs["_source_path"] = "dataset.csv"

    with pytest.raises(
        DataError,
        match="Não foi possível inferir colunas alvo numéricas. Informe 'target_cols'.",
    ):
        _extract_targets_and_groups(df, None, "group", target_type="regression")


def test_extract_targets_regression_rejects_categorical_values():
    df = pd.DataFrame({"target": ["a", "b", "c"], "feat": [1.0, 2.0, 3.0]})
    df.attrs["_source_path"] = "dataset.csv"

    with pytest.raises(
        DataError, match="Coluna alvo target deve conter valores numéricos para regressão"
    ):
        _extract_targets_and_groups(df, ["target"], None, target_type="regression")


def test_prepare_inputs_regression_returns_float_targets(tmp_path):
    df = pd.DataFrame({"100": [1.0, 2.0, 3.0], "target": [0.5, 0.7, 0.9]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    cfg = {
        "input_path": str(path),
        "target_cols": ["target"],
        "target_type": "regression",
    }

    ds_list = prepare_inputs(cfg)
    assert len(ds_list) == 1
    assert ds_list[0]["Y"].dtype.kind == "f"


def test_prepare_inputs_regression_raises_for_non_numeric_targets(tmp_path):
    df = pd.DataFrame({"100": [1.0, 2.0, 3.0], "target": ["a", "b", "c"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    cfg = {
        "input_path": str(path),
        "target_cols": ["target"],
        "target_type": "regression",
    }

    with pytest.raises(
        DataError, match="Coluna alvo target deve conter valores numéricos para regressão"
    ):
        prepare_inputs(cfg)


def test_prepare_inputs_respects_drop_nans_flag(tmp_path):
    df = pd.DataFrame(
        {
            "100": [1.0, 2.0, float("nan")],
            "200": [0.1, float("nan"), 0.3],
            "target": ["a", "b", "c"],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    cfg_drop = {
        "input_path": str(path),
        "target_cols": ["target"],
        "drop_nans": True,
    }
    dropped = prepare_inputs(cfg_drop)
    assert dropped[0]["n_nans"] == 2
    assert dropped[0]["X"].shape[0] == 1
    assert not dropped[0]["X"].isna().any().any()

    cfg_keep = {
        "input_path": str(path),
        "target_cols": ["target"],
        "drop_nans": False,
    }
    kept = prepare_inputs(cfg_keep)
    assert kept[0]["n_nans"] == 2
    assert kept[0]["X"].shape[0] == 3
    assert kept[0]["X"].isna().any(axis=1).sum() == 2



def test_cache_respects_drop_nans_switch(tmp_path):
    df = pd.DataFrame(
        {
            "100": [1.0, float("nan"), 3.0],
            "target": ["x", "y", "z"],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    out_dir = tmp_path / "out"
    cfg_drop = {
        "input_path": str(path),
        "target_cols": ["target"],
        "drop_nans": True,
        "out_dir": str(out_dir),
    }
    dropped = pipeline_prepare_inputs(cfg_drop)
    assert dropped[0]["X"].shape[0] == 2

    cfg_keep = {
        "input_path": str(path),
        "target_cols": ["target"],
        "drop_nans": False,
        "out_dir": str(out_dir),
    }
    kept = pipeline_prepare_inputs(cfg_keep)
    assert kept[0]["X"].shape[0] == 3
