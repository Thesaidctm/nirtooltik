"""Quality control utilities for spectral datasets."""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicated rows in ``df`` using row hashes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Subset of ``df`` containing the duplicated rows.
    """
    row_hashes = pd.util.hash_pandas_object(df, index=False).groupby(level=0).sum()
    dup_idx = row_hashes[row_hashes.duplicated()].index
    return df.loc[dup_idx]


def detect_identical_spectra(X: pd.DataFrame) -> List[List[int]]:
    """Identify identical spectral vectors in ``X``.

    Parameters
    ----------
    X : pandas.DataFrame
        Spectral matrix.

    Returns
    -------
    list of list of int
        Groups of row indices where spectra are identical.
    """
    row_hashes = pd.util.hash_pandas_object(X, index=False).groupby(level=0).sum()
    groups = {}
    for idx, h in row_hashes.items():
        groups.setdefault(h, []).append(int(idx))
    return [idxs for idxs in groups.values() if len(idxs) > 1]


def pca_outliers(X: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Compute PCA scores alongside Hotelling's T² and Q-residuals.

    Parameters
    ----------
    X : pandas.DataFrame
        Spectral matrix.
    n_components : int, optional
        Number of principal components to retain. Default is 2.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns ``PC1``, ``PC2``, ``T2`` and ``Q_residuals``.
    """
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X.values)
    reconstructed = pca.inverse_transform(scores)
    resid = X.values - reconstructed
    q_res = np.sum(resid**2, axis=1)
    t2 = np.sum((scores**2) / pca.explained_variance_, axis=1)
    out = pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"], index=X.index)
    out["T2"] = t2
    out["Q_residuals"] = q_res
    return out
