from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TransferModel(Protocol):
    """
    Interface geral para modelos de transferência de calibração.
    """

    def transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        ...


@dataclass
class DirectStandardizationModel:
    """
    Implementação simples de Direct Standardization (DS).

    Aproxima:
        X_secondary @ F ≈ X_primary

    onde:
    - X_primary: espectros no instrumento de referência (A)
    - X_secondary: espectros no instrumento secundário (B)
    - F: matriz de transformação (n_features x n_features) estimada por mínimos quadrados.

    Usamos np.linalg.lstsq para estabilidade numérica.
    """

    F: np.ndarray  # shape (n_features, n_features)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.F.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features but F expects {self.F.shape[0]}"
            )
        return X @ self.F


def fit_direct_standardization(
    X_primary: np.ndarray,
    X_secondary: np.ndarray,
    *,
    rcond: float | None = None,
) -> DirectStandardizationModel:
    """
    Ajusta um modelo de Direct Standardization (DS).

    Parâmetros
    ----------
    X_primary : array (n_samples, n_features)
        Espectros medidos no instrumento de referência (A).
    X_secondary : array (n_samples, n_features)
        Espectros das mesmas amostras medidos no instrumento secundário (B),
        já alinhados na mesma grade espectral de X_primary.

    Retorna
    -------
    DirectStandardizationModel
        Objeto com matriz F que transforma espectros de B para o espaço de A.
    """
    Xp = np.asarray(X_primary, dtype=float)
    Xs = np.asarray(X_secondary, dtype=float)

    if Xp.shape != Xs.shape:
        raise ValueError(
            f"X_primary and X_secondary must have same shape, got {Xp.shape} vs {Xs.shape}"
        )

    n_features = Xp.shape[1]
    F = np.zeros((n_features, n_features), dtype=float)

    # Resolve Xs @ F ≈ Xp coluna a coluna
    for j in range(n_features):
        y = Xp[:, j]
        f_j, *_ = np.linalg.lstsq(Xs, y, rcond=rcond)
        F[:, j] = f_j

    return DirectStandardizationModel(F=F)
