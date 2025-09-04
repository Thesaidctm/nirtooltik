
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import savgol_filter

class SNV(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True)+1e-12)

class MSC(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ref_ = np.mean(X, axis=0); return self
    def transform(self, X):
        X = np.asarray(X, dtype=float); ref = self.ref_; out = np.empty_like(X)
        for i in range(X.shape[0]):
            A = np.vstack([X[i], np.ones(X.shape[1])]).T
            coef, *_ = np.linalg.lstsq(A, ref, rcond=None)
            out[i] = (X[i]*coef[0]) + coef[1]
        return out

class Detrend(BaseEstimator, TransformerMixin):
    def __init__(self, order:int=2): self.order=order
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X, dtype=float); n = X.shape[1]; x = np.arange(n)
        V = np.vander(x, N=self.order+1, increasing=True); P = V @ np.linalg.pinv(V)
        return X - (X @ P.T)

class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=15, polyorder=2, deriv=1):
        self.window_length=int(window_length); self.polyorder=int(polyorder); self.deriv=int(deriv)
    def fit(self, X, y=None):
        wl = self.window_length + (self.window_length+1)%2
        if wl > X.shape[1]: raise ValueError("window_length > n_features"); 
        if self.polyorder >= wl: raise ValueError("polyorder >= window_length")
        self._wl = wl; return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return savgol_filter(X, window_length=self._wl, polyorder=self.polyorder, deriv=self.deriv, axis=1)

class EMSC(BaseEstimator, TransformerMixin):
    def __init__(self, degree:int=2): self.degree=degree
    def fit(self, X, y=None):
        self.ref_ = np.mean(X, axis=0); self.x_ = np.linspace(0,1,X.shape[1]); return self
    def transform(self, X):
        X = np.asarray(X, dtype=float); terms = [np.ones_like(self.x_), self.x_] + [self.x_**k for k in range(2,self.degree+1)]
        A = np.vstack([self.ref_, *terms]).T; pinv = np.linalg.pinv(A); out = np.empty_like(X)
        for i in range(X.shape[0]):
            coef = pinv @ X[i]; fit = A @ coef; out[i] = X[i] - (fit - self.ref_)
        return out

class EPO(BaseEstimator, TransformerMixin):
    def __init__(self, Z=None): self.Z=Z
    def fit(self, X, y=None):
        if self.Z is None: self.P_=None
        else:
            Z = np.asarray(self.Z).reshape(-1,1); Pz = Z @ np.linalg.pinv(Z); self.P_ = np.eye(Z.shape[0]) - Pz
        return self
    def transform(self, X):
        if self.P_ is None: return X
        return self.P_ @ X
