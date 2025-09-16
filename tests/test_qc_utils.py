from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pandas.testing as pdt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402
from nircls.utils.qc import _row_hashes, detect_identical_spectra, find_duplicates  # noqa: E402


def test_row_hashes_matches_pandas_hash():
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 3],
            "b": [4, 4, 5, 6],
            "c": ["x", "x", "y", "z"],
        }
    )

    expected = pd.util.hash_pandas_object(df, index=False).groupby(level=0).sum()

    result = _row_hashes(df)

    pdt.assert_series_equal(result, expected)


def test_find_duplicates_returns_duplicate_rows():
    df = pd.DataFrame(
        {
            "feat1": [0, 0, 1, 2, 0],
            "feat2": [1, 1, 2, 3, 1],
            "label": ["a", "a", "b", "c", "a"],
        }
    )

    dup_rows = find_duplicates(df)

    expected = df.iloc[[1, 4]]
    pdt.assert_frame_equal(dup_rows, expected)


def test_detect_identical_spectra_groups_indices():
    X = pd.DataFrame(
        [
            [0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2],
            [0.1, 0.2, 0.3],
            [2.0, 2.1, 2.2],
            [1.0, 1.1, 1.2],
        ]
    )

    groups = detect_identical_spectra(X)

    assert sorted(sorted(group) for group in groups) == [[0, 2], [1, 4]]
