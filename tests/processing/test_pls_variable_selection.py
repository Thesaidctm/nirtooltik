import sys
import types

import numpy as np
from sklearn.datasets import make_regression

# Mock streamlit to avoid optional dependency during import
sys.modules.setdefault("streamlit", types.SimpleNamespace())

from nircls.processing.selection.pls_variable_selection import (
    apply_pls_variable_selection,
    pls_variable_selection,
)


def test_pls_variable_selection_runs_and_selects_variables():
    X, y = make_regression(n_samples=80, n_features=30, noise=0.1, random_state=42)

    result = pls_variable_selection(X, y, max_components=5, cv=3, verbose=False)

    assert result.n_components_opt >= 1
    assert result.X_selected.shape[1] < X.shape[1]

    X_transformed = apply_pls_variable_selection(X, result)
    np.testing.assert_equal(X_transformed.shape, result.X_selected.shape)
