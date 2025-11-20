from __future__ import annotations

from typing import Any, Dict

import numpy as np

from nircls.processing.selection.pls_variable_selection import (
    PLSVariableSelectionResult,
    apply_pls_variable_selection,
    pls_variable_selection,
)


def optimize_pls_with_variable_selection(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int,
    *,
    cv: int = 5,
) -> Dict[str, Any]:
    """
    Wrapper de alto nível para rodar otimização de PLS com seleção de variáveis
    e retornar um dicionário pronto para ser serializado/armazenado.
    """
    result: PLSVariableSelectionResult = pls_variable_selection(
        X, y, max_components=max_components, cv=cv, verbose=True
    )

    return {
        "n_components_opt": int(result.n_components_opt),
        "first_discard_index": int(result.first_discard_index),
        "sorted_indices": result.sorted_indices.tolist(),
        "mse_matrix_shape": list(result.mse_matrix.shape),
        "mse_min": float(np.min(result.mse_matrix[result.mse_matrix > 0])),
    }
