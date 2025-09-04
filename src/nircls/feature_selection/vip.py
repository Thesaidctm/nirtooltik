from __future__ import annotations
import numpy as np

def _ssy(Y):
    Y = np.asarray(Y, dtype=float)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    return (Yc**2).sum()

def compute_vip(pls, X, Y):
    # VIP for PLSRegression following common chemometrics formula.
    # X shape: (n_samples, p), Y one-hot: (n_samples, n_classes)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_  # (n_components, n_classes)
    p = X.shape[1]
    A = T.shape[1]  # n_components
    # Sum of squares explained in Y by each component
    SSY = np.zeros(A)
    for a in range(A):
        t = T[:, a][:, None]        # (n,1)
        q = Q[a, :][None, :]        # (1, C)
        Y_hat_a = t @ q             # contribution of component a
        SSY[a] = (Y_hat_a**2).sum()
    # VIP per variable j
    vip = np.zeros(p)
    denom = SSY.sum() + 1e-12
    for j in range(p):
        s = 0.0
        for a in range(A):
            w_ja = W[j, a]
            w_a = np.linalg.norm(W[:, a]) + 1e-12
            s += SSY[a] * (w_ja / w_a) ** 2
        vip[j] = np.sqrt(p * s / denom)
    return vip
