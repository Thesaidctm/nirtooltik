"""Custom dataset splitting utilities."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def _normalise_distance_matrix(dist: np.ndarray) -> np.ndarray:
    """Scale ``dist`` to ``[0, 1]`` when a non-zero maximum exists."""

    max_val = float(np.max(dist))
    if max_val > 0:
        dist = dist / max_val
    return dist


def spxy_split(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    *,
    random_state: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
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

    rng = np.random.default_rng(random_state)

    # pairwise distances for X and y, normalised
    dx = _normalise_distance_matrix(cdist(X, X))
    dy = _normalise_distance_matrix(cdist(y, y))
    d = np.sqrt(dx**2 + dy**2)

    order: List[int]
    if not np.any(d > 0):
        # Degenerate case: every sample is identical. Fall back to a deterministic
        # split that can still be seeded for reproducibility.
        order = list(range(n_samples))
        if random_state is not None:
            rng.shuffle(order)
    else:
        # initial pair: farthest samples
        flat_idx = int(np.argmax(d))
        a, b = divmod(flat_idx, n_samples)
        if a == b:
            # ``np.argmax`` can land on the diagonal when numerical noise creates
            # ties. Select the farthest element ignoring the diagonal.
            mask = ~np.eye(n_samples, dtype=bool)
            flat_idx = int(np.argmax(d[mask]))
            a, b = divmod(flat_idx, n_samples)
        selected = [a]
        if b != a:
            selected.append(b)
        remaining = [i for i in range(n_samples) if i not in selected]
        order = selected.copy()

        # greedy selection maximising distance to already selected samples
        while remaining:
            dist_to_sel = d[np.ix_(remaining, order)]
            if dist_to_sel.size == 0:
                order.extend(remaining)
                break
            min_dist = dist_to_sel.min(axis=1)
            if np.allclose(min_dist, min_dist[0]):
                next_pos = int(rng.integers(len(remaining))) if remaining else 0
            else:
                next_pos = int(np.argmax(min_dist))
            next_idx = remaining.pop(next_pos)
            order.append(next_idx)

    folds = np.array_split(np.array(order), n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n_samples)
    for val_idx in folds:
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)
        splits.append((train_idx, val_idx))
    return splits
