from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict


@dataclass
class PLSVariableSelectionResult:
    """
    Resultado da otimização de PLS com seleção de variáveis.

    Attributes
    ----------
    X_selected : np.ndarray
        Matriz X com as variáveis selecionadas (colunas filtradas).
    n_components_opt : int
        Número ótimo de componentes de PLS (LVs).
    first_discard_index : int
        Quantidade de variáveis descartadas a partir da ordenação dos coeficientes
        (equivalente ao índice 'mseminy' do tutorial).
    sorted_indices : np.ndarray
        Índices das variáveis de X ordenados por |coeficiente PLS| (ascendente).
    mse_matrix : np.ndarray
        Matriz 2D de MSE em CV com shape (max_components, n_features),
        onde mse[i, j] corresponde ao MSE usando (i+1) componentes de PLS e
        descartando j variáveis da cauda (menores coeficientes).
    """

    X_selected: np.ndarray
    n_components_opt: int
    first_discard_index: int
    sorted_indices: np.ndarray
    mse_matrix: np.ndarray


def _validate_inputs(X: np.ndarray, y: np.ndarray, max_components: int) -> Tuple[np.ndarray, np.ndarray, int]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if y.ndim != 2:
        raise ValueError("y must be a 1D or 2D array.")

    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("X must contain at least two samples for PLS.")

    max_components = min(max_components, n_samples - 1, n_features - 1)
    if max_components < 1:
        raise ValueError("max_components must allow at least one component.")

    return X, y, max_components


def _sort_indices_by_coef(coef: np.ndarray) -> np.ndarray:
    coef = np.asarray(coef)
    if coef.ndim == 1:
        coef = coef.reshape(-1, 1)
    return np.argsort(np.linalg.norm(coef, axis=1))


def pls_variable_selection(
    X: np.ndarray,
    y: np.ndarray,
    max_components: int,
    *,
    cv: int = 5,
    verbose: bool = True,
) -> PLSVariableSelectionResult:
    """
    Otimiza simultaneamente número de componentes de PLS e seleção de variáveis
    usando o método de filtragem de coeficientes descrito no tutorial
    'A variable selection method for PLS in Python'.

    A ideia:
    - Para cada número de componentes (1..max_components):
      - Ajusta um PLS com X completo;
      - Ordena variáveis por |coeficiente PLS| (norma no caso multivariado);
      - Reordena X de acordo com essa ordem;
      - Descarta 1, 2, 3, ... variáveis de menor coeficiente,
        recalcula PLS e MSE em cross-validation;
      - Armazena MSE em matriz mse[lv_idx, discard_idx].
    - No fim, busca o mínimo global de MSE (ignorando zeros) e retorna:
      - X filtrado;
      - número ótimo de componentes;
      - quantas variáveis foram descartadas a partir da cauda;
      - ordem dos índices.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de preditores (n_samples, n_features).
    y : np.ndarray
        Vetor/array de resposta (n_samples,) ou (n_samples, 1).
    max_components : int
        Número máximo de componentes de PLS a testar.
    cv : int, default=5
        Número de folds de cross-validation (sklearn).
    verbose : bool, default=True
        Se True, imprime progresso no stdout.
    """
    X, y, max_components = _validate_inputs(X, y, max_components)

    n_samples, n_features = X.shape

    # Matriz de MSE: linhas = componentes; colunas = número de variáveis descartadas
    mse = np.zeros((max_components, n_features), dtype=float)

    for lv_idx in range(max_components):
        n_comp = lv_idx + 1

        pls_full = PLSRegression(n_components=n_comp)
        pls_full.fit(X, y)

        coef = pls_full.coef_
        sorted_ind = _sort_indices_by_coef(coef)  # ascendente

        Xc = X[:, sorted_ind]

        # Descarta variáveis uma a uma a partir da cauda (menores coeficientes)
        # Garantindo que sempre haja pelo menos n_comp variáveis.
        max_discard = n_features - n_comp

        for discard_idx in range(max_discard + 1):
            X_sub = Xc[:, discard_idx:]

            if X_sub.shape[1] < n_comp:
                break

            pls_sub = PLSRegression(n_components=n_comp)
            y_cv = cross_val_predict(pls_sub, X_sub, y, cv=cv)

            mse[lv_idx, discard_idx] = mean_squared_error(y, y_cv)

        if verbose:
            pct = int(100 * n_comp / max_components)
            print(f"\rPLS variable selection: {pct}% ({n_comp}/{max_components} comps)", end="")

    if verbose:
        print()

    # Encontrar mínimo global de MSE, ignorando zeros (descartes não testados)
    non_zero = mse > 0
    if not np.any(non_zero):
        raise RuntimeError("PLS variable selection failed: all MSE values are zero.")

    # np.argmin flattens; usamos argwhere para recuperar primeira ocorrência
    lv_idx_opt, discard_idx_opt = np.argwhere(mse == np.min(mse[non_zero]))[0]

    n_components_opt = lv_idx_opt + 1

    # Ajusta PLS final com todos os dados e n_components_opt para obter ordem final
    pls_opt = PLSRegression(n_components=n_components_opt)
    pls_opt.fit(X, y)
    coef_opt = pls_opt.coef_
    sorted_indices = _sort_indices_by_coef(coef_opt)

    X_sorted = X[:, sorted_indices]
    X_selected = X_sorted[:, discard_idx_opt:]

    return PLSVariableSelectionResult(
        X_selected=X_selected,
        n_components_opt=n_components_opt,
        first_discard_index=discard_idx_opt,
        sorted_indices=sorted_indices,
        mse_matrix=mse,
    )


def apply_pls_variable_selection(
    X: np.ndarray,
    result: PLSVariableSelectionResult,
) -> np.ndarray:
    """
    Aplica a mesma seleção de variáveis a um novo X (por exemplo, conjunto de teste).

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de preditores com mesmas variáveis originais do treino.
    result : PLSVariableSelectionResult
        Objeto retornado por pls_variable_selection.

    Retorna
    -------
    np.ndarray
        Matriz X transformada com as mesmas variáveis selecionadas do treino.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    X_sorted = X[:, result.sorted_indices]
    return X_sorted[:, result.first_discard_index :]
