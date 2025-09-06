"""Custom dataset splitting utilities."""
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist


def spxy_split(X: np.ndarray, y: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split indices using the SPXY algorithm.

    Parameters
    ----------
    X:
        Feature matrix with shape (n_samples, n_features).
    y:
        Target array with shape (n_samples,) or (n_samples, n_targets).
    n_splits:
        Number of folds to generate.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        A list containing ``(train_idx, val_idx)`` tuples for each fold.
    """

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    n_samples = X.shape[0]
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError("n_splits must be in [2, n_samples]")

    # pairwise distances for X and y, normalised
    dx = cdist(X, X)
    dy = cdist(y, y)
    if np.max(dx) > 0:
        dx = dx / np.max(dx)
    if np.max(dy) > 0:
        dy = dy / np.max(dy)
    d = np.sqrt(dx**2 + dy**2)

    # initial pair: farthest samples
    a, b = divmod(np.argmax(d), n_samples)
    selected = [a, b]
    remaining = [i for i in range(n_samples) if i not in selected]
    order = selected.copy()

    # greedy selection maximising distance to already selected samples
    while remaining:
        dist_to_sel = d[np.ix_(remaining, order)]
        min_dist = dist_to_sel.min(axis=1)
        next_idx = remaining[int(np.argmax(min_dist))]
        order.append(next_idx)
        remaining.remove(next_idx)

    folds = np.array_split(np.array(order), n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n_samples)
    for val_idx in folds:
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)
        splits.append((train_idx, val_idx))
    return splits
