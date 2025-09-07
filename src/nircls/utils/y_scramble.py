from __future__ import annotations

"""Utility functions for Y-scrambling overfit assessment."""

from typing import Iterable, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score


def y_scramble_scores(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: Iterable | int | Sequence,
    n_iter: int = 20,
    scoring: str = "balanced_accuracy",
    random_state: Optional[int] = None,
    groups: Optional[Sequence] = None,
) -> np.ndarray:
    """Estimate performance on permuted labels.

    The labels ``y`` are shuffled ``n_iter`` times and the estimator is
    evaluated via cross-validation for each permutation. The mean score of each
    permutation is returned.

    Parameters
    ----------
    estimator:
        Estimator or pipeline adhering to the scikit-learn API.
    X:
        Feature matrix.
    y:
        True labels.
    cv:
        Cross-validation splitter or iterable yielding train/test indices.
    n_iter:
        Number of label permutations.
    scoring:
        Scoring function passed to :func:`sklearn.model_selection.cross_val_score`.
    random_state:
        Seed used when shuffling ``y``.
    groups:
        Optional group labels used when ``cv`` is group-aware.

    Returns
    -------
    np.ndarray
        Array with the average score obtained at each permutation.
    """

    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_iter):
        y_perm = rng.permutation(y)
        try:
            cv_scores = cross_val_score(
                estimator,
                X,
                y_perm,
                cv=cv,
                scoring=scoring,
                groups=groups,
            )
            scores.append(cv_scores.mean())
        except Exception:  # pragma: no cover - best effort
            continue
    return np.asarray(scores, dtype=float)
