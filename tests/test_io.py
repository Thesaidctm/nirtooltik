from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nircls.io import _extract_targets_and_groups
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
