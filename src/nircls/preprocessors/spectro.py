
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class SNV(BaseEstimator, TransformerMixin):
    """Apply standard normal variate normalization to spectra."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-12
        return (X - mean) / std


class MSC(BaseEstimator, TransformerMixin):
    """Perform multiplicative scatter correction (MSC).

    The mean spectrum of the calibration set is used as reference.
    """

    def fit(self, X, y=None):
        self.ref_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        ref = self.ref_
        x_mean = X.mean(axis=1, keepdims=True)
        ref_mean = ref.mean()
        x_centered = X - x_mean
        ref_centered = ref - ref_mean
        numerator = (x_centered * ref_centered).sum(axis=1, keepdims=True)
        denominator = (x_centered ** 2).sum(axis=1, keepdims=True) + 1e-12
        slope = numerator / denominator
        intercept = ref_mean - slope * x_mean
        return slope * X + intercept


class Detrend(BaseEstimator, TransformerMixin):
    """Remove polynomial trends from spectra.

    Parameters
    ----------
    order : int, default=2
        Polynomial order used for detrending.
    """

    def __init__(self, order: int = 2):
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        x = np.arange(n)
        V = np.vander(x, N=self.order + 1, increasing=True)
        P = V @ np.linalg.pinv(V)
        return X - (X @ P.T)


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    """Apply a Savitzky-Golay filter to spectra.

    Parameters
    ----------
    window_length : int, default=15
        Length of the filter window (must be odd).
    polyorder : int, default=2
        Polynomial order used to fit the samples.
    deriv : int, default=1
        Order of the derivative to compute.
    """

    def __init__(self, window_length=15, polyorder=2, deriv=1):
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.deriv = int(deriv)

    def fit(self, X, y=None):
        wl = self.window_length + (self.window_length + 1) % 2
        if wl > X.shape[1]:
            raise ValueError("window_length > n_features")
        if self.polyorder >= wl:
            raise ValueError("polyorder >= window_length")
        self._wl = wl
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return savgol_filter(
            X,
            window_length=self._wl,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
        )


class EMSC(BaseEstimator, TransformerMixin):
    """Extended multiplicative scatter correction.

    Parameters
    ----------
    degree : int, default=2
        Degree of the polynomial baseline to model.
    """

    def __init__(self, degree: int = 2):
        self.degree = degree

    def fit(self, X, y=None):
        self.ref_ = np.mean(X, axis=0)
        self.x_ = np.linspace(0, 1, X.shape[1])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        terms = [np.ones_like(self.x_), self.x_]
        terms.extend(self.x_ ** k for k in range(2, self.degree + 1))
        A = np.vstack([self.ref_, *terms]).T
        pinv = np.linalg.pinv(A)
        coeffs = pinv @ X.T
        fit = (A @ coeffs).T
        return X - fit + self.ref_


class EPO(BaseEstimator, TransformerMixin):
    """External parameter orthogonalization.

    Parameters
    ----------
    Z : array-like of shape (n_features,) or (n_features, n_vectors), optional
        External parameters to remove from the spectra.
    """

    def __init__(self, Z=None):
        self.Z = Z

    def fit(self, X, y=None):
        if self.Z is None:
            self.P_ = None
        else:
            Z = np.asarray(self.Z).reshape(-1, 1)
            Pz = Z @ np.linalg.pinv(Z)
            self.P_ = np.eye(Z.shape[0]) - Pz
        return self

    def transform(self, X):
        if self.P_ is None:
            return X
        return self.P_ @ X
