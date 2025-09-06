from __future__ import annotations
import numpy as np


def compute_vip(pls, X, Y):
    """Compute Variable Importance in Projection (VIP) scores.

    VIP scores summarise the contribution of each predictor in a PLS model.
    For predictor *j* the score is given by::

        VIP_j = sqrt( p * sum_a SSY_a * (w_{j,a}^2 / ||w_a||^2) / SSY_total )

    where ``p`` is the number of predictors, ``SSY_a = ||t_a||^2 * ||q_a||^2`` is
    the variation in ``Y`` explained by component ``a``, ``w_{j,a}`` is the
    weight of variable ``j`` on component ``a``, ``||w_a||`` is the Euclidean
    norm of the weight vector for component ``a`` and ``SSY_total`` is the total
    sum of squares of the centred response ``Y``.

    Parameters
    ----------
    pls : PLSRegression
        Fitted scikit-learn ``PLSRegression`` providing ``x_scores_``,
        ``x_weights_`` and ``y_loadings_``.
    X : array-like of shape (n_samples, n_features)
        Training data used to fit the model.
    Y : array-like of shape (n_samples, n_targets)
        Target data (e.g. one-hot encoded class matrix).

    Returns
    -------
    vip : ndarray of shape (n_features,)
        VIP score for each predictor.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_  # (n_components, n_targets)
    p = X.shape[1]

    Yc = Y - Y.mean(axis=0, keepdims=True)
    SSY_total = (Yc**2).sum() + 1e-12

    SSY = (T**2).sum(axis=0) * (Q**2).sum(axis=1)

    norms_sq = (W**2).sum(axis=0) + 1e-12
    s = (W**2 / norms_sq) @ SSY

    vip = np.sqrt(p * s / SSY_total)
    return vip
