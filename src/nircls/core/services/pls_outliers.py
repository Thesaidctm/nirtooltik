from __future__ import annotations

from typing import Any, Dict

import numpy as np

from nircls.processing.outliers.pls_outlier_detection import (
    compute_pls_outlier_diagnostics,
    optimize_outlier_removal_with_pls,
)


def get_pls_outlier_report(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    max_outliers: int = 50,
    cv: int = 5,
) -> Dict[str, Any]:
    """
    Gera um relatório completo de outliers para um modelo PLS fixo:
    - Q, T2, limites e máscara de outliers;
    - curva de MSE vs número de outliers removidos;
    - número ótimo de outliers a remover.
    """
    diag = compute_pls_outlier_diagnostics(X, y, n_components=n_components)
    opt = optimize_outlier_removal_with_pls(
        X, y, n_components=n_components, max_outliers=max_outliers, cv=cv
    )

    return {
        "Q": diag.Q.tolist(),
        "T2": diag.T2.tolist(),
        "Q_conf": float(diag.Q_conf),
        "T2_conf": float(diag.T2_conf),
        "outlier_mask": diag.outlier_mask.tolist(),
        "mse_curve": opt.mse_curve.tolist(),
        "n_outliers_opt": int(opt.n_outliers_opt),
        "sorted_indices": opt.sorted_indices.tolist(),
    }
