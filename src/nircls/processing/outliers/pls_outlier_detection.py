from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import f
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict


@dataclass
class PLSOutlierDiagnostics:
    """
    Métricas de diagnóstico de outliers para um modelo PLS fixo.

    Attributes
    ----------
    Q : np.ndarray
        Q-residuals por amostra (variação não explicada pelo modelo).
    T2 : np.ndarray
        Hotelling's T-squared por amostra (variação dentro do modelo).
    Q_conf : float
        Limite de confiança (ex.: 95%) para Q-residuals.
    T2_conf : float
        Limite de confiança (ex.: 95%) para Hotelling's T-squared.
    outlier_mask : np.ndarray[bool]
        Vetor booleano (n_samples,) indicando quais amostras são outliers
        (Q > Q_conf ou T2 > T2_conf).
    """

    Q: np.ndarray
    T2: np.ndarray
    Q_conf: float
    T2_conf: float
    outlier_mask: np.ndarray


@dataclass
class PLSOutlierOptimizationResult:
    """
    Resultado da otimização da remoção de outliers via MSE-CV.

    Attributes
    ----------
    mse_curve : np.ndarray
        Vetor de MSE-CV para j = 0..max_outliers_removed-1 amostras removidas.
    n_outliers_opt : int
        Número ótimo de amostras a remover (posição do mínimo da curva).
    sorted_indices : np.ndarray
        Índices das amostras ordenados da mais "distante" para a mais "próxima"
        (com base em sqrt(Q^2 + T2^2)).
    """

    mse_curve: np.ndarray
    n_outliers_opt: int
    sorted_indices: np.ndarray


def _prepare_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    return X, y_arr


def _adjust_n_components(n_components: int, n_samples: int, n_features: int) -> int:
    return min(n_components, n_samples - 1, n_features - 1)


def compute_pls_outlier_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    conf_level: float = 0.95,
) -> PLSOutlierDiagnostics:
    """
    Ajusta um PLS com n_components e calcula:
    - Q-residuals,
    - Hotelling's T-squared,
    - limites de confiança (95% por padrão),
    - máscara de outliers (Q > Q_conf ou T2 > T2_conf).

    Implementa as fórmulas do tutorial:
    - X ≈ T P^T  =>  Err = X - T P^T  =>  Q = soma linha a linha de Err^2
    - T2 = soma linha a linha de (scores / std(scores))^2
    """
    X_arr, y_arr = _prepare_xy(X, y)

    n_samples, n_features = X_arr.shape
    n_comp = _adjust_n_components(n_components, n_samples, n_features)

    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_arr, y_arr)

    T = pls.x_scores_
    P = pls.x_loadings_

    Err = X_arr - np.dot(T, P.T)
    Q = np.sum(Err**2, axis=1)

    T_std = np.std(T, axis=0, ddof=1)
    T_std[T_std == 0] = 1.0
    T_norm = T / T_std
    T2 = np.sum(T_norm**2, axis=1)

    k = n_comp
    m = n_samples
    T2_conf = f.ppf(q=conf_level, dfn=k, dfd=(m - k)) * k * (m - 1) / (m - k)

    Q_positive = Q[Q > 0]
    if Q_positive.size == 0:
        Q_conf = float(np.max(Q))
    else:
        Q_conf = float(np.quantile(Q_positive, conf_level))

    outlier_mask = (Q > Q_conf) | (T2 > T2_conf)

    return PLSOutlierDiagnostics(
        Q=Q,
        T2=T2,
        Q_conf=Q_conf,
        T2_conf=T2_conf,
        outlier_mask=outlier_mask,
    )


def optimize_outlier_removal_with_pls(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    max_outliers: int,
    cv: int = 5,
) -> PLSOutlierOptimizationResult:
    """
    Implementa o algoritmo do tutorial:
    - Calcula Q e T2 para um PLS com n_components.
    - Calcula a "distância combinada" como sqrt(Q^2 + T2^2).
    - Ordena as amostras por essa distância (decrescente).
    - Remove iterativamente 1, 2, ..., max_outliers amostras
      e, para cada j, calcula PLS + MSE em CV.
    - Retorna curva de MSE e posição do mínimo (n_outliers_opt).

    Parâmetros
    ----------
    X, y : dados de calibração
    n_components : número de LVs do PLS (fixo aqui)
    max_outliers : máximo de amostras a tentar remover
    cv : número de folds para cross-validation
    """
    X_arr, y_arr = _prepare_xy(X, y)
    n_samples, n_features = X_arr.shape

    max_outliers_adj = min(max_outliers, n_samples - 5)
    n_comp = _adjust_n_components(n_components, n_samples, n_features)

    diag = compute_pls_outlier_diagnostics(X_arr, y_arr, n_components=n_comp)

    dist = np.sqrt(diag.Q**2 + diag.T2**2)
    sorted_indices = np.flip(np.argsort(dist), axis=0)

    Xc = X_arr[sorted_indices, :]
    yc = y_arr[sorted_indices]

    mse = np.zeros(max_outliers_adj, dtype=float)

    for j in range(max_outliers_adj):
        X_sub = Xc[j:, :]
        y_sub = yc[j:]

        pls = PLSRegression(n_components=n_comp)
        y_cv = cross_val_predict(pls, X_sub, y_sub, cv=cv)

        mse[j] = mean_squared_error(y_sub, y_cv)

    non_zero = mse > 0
    if np.any(non_zero):
        j_opt = int(np.where(mse == np.min(mse[non_zero]))[0][0])
    else:
        j_opt = 0

    return PLSOutlierOptimizationResult(
        mse_curve=mse,
        n_outliers_opt=j_opt,
        sorted_indices=sorted_indices,
    )


__all__ = [
    "PLSOutlierDiagnostics",
    "PLSOutlierOptimizationResult",
    "compute_pls_outlier_diagnostics",
    "optimize_outlier_removal_with_pls",
]
