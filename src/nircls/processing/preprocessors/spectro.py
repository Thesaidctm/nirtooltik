
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation
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
        return X @ self.P_


class AdaptiveQuantileSegmentNormalizer(BaseEstimator, TransformerMixin):
    """Adaptive quantile-based normalization for NIR spectra.

    This transformer implements a patent-oriented pre-processing strategy that
    dynamically segments spectra according to local activity and aligns each
    segment to a robust global reference. The procedure combines three
    innovations to improve calibration transfer:

    1. **Derivative-driven segmentation** – spectral bands with high
       structural activity are detected using the median absolute deviation of
       first-order derivatives. The routine automatically chooses segment
       boundaries while enforcing a minimum width to avoid overfitting noise.
    2. **Quantile invariant scaling** – within each segment, every spectrum is
       normalized so that its median and interquartile range match those of the
       calibration reference computed during ``fit``. This preserves local
       contrast without assuming Gaussian behaviour.
    3. **Stability-aware blending** – spectra with atypical derivative energy
       are softly blended with the raw signal through a logistic weighting,
       preventing aggressive corrections on unstable measurements.

    Parameters
    ----------
    min_segment_size : int, default=25
        Minimum number of wavelengths per adaptive segment. Larger values make
        the segmentation more conservative.
    blend_strength : float, default=4.0
        Controls the slope of the logistic curve used for stability-aware
        blending. Higher values reduce the contribution of corrected spectra
        when derivative energy deviates from the calibration set.
    """

    def __init__(self, min_segment_size: int = 25, blend_strength: float = 4.0):
        if min_segment_size < 5:
            raise ValueError("min_segment_size must be ≥ 5")
        if blend_strength < 0:
            raise ValueError("blend_strength must be non-negative")
        self.min_segment_size = int(min_segment_size)
        self.blend_strength = float(blend_strength)

    def fit(self, X, y=None):  # noqa: D401 - standard sklearn signature
        """Learn adaptive segments and reference quantiles from ``X``."""

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("AdaptiveQuantileSegmentNormalizer expects 2D input")
        n_samples, n_features = X.shape
        if n_features < 2:
            raise ValueError("At least two spectral features are required")

        diff = np.diff(X, axis=1)
        activity = np.median(np.abs(diff), axis=0)
        activity_center = float(np.median(activity))
        activity_spread = float(
            median_abs_deviation(activity, scale="normal") or 0.0
        )
        if not np.isfinite(activity_spread) or activity_spread == 0.0:
            activity_spread = max(activity_center, 1.0)
        threshold = activity_center + 0.5 * activity_spread

        candidates = np.where(activity > threshold)[0] + 1
        segments = [0]
        for idx in candidates:
            idx = int(idx)
            if idx - segments[-1] < self.min_segment_size:
                continue
            if n_features - idx < self.min_segment_size:
                continue
            if idx not in segments:
                segments.append(idx)
        if segments[-1] != n_features:
            segments.append(n_features)
        if len(segments) == 2:  # enforce at least one adaptive split
            midpoint = n_features // 2
            if midpoint - segments[0] >= self.min_segment_size:
                segments.insert(1, midpoint)
            elif n_features - self.min_segment_size > segments[0]:
                segments.insert(1, n_features - self.min_segment_size)

        segments = sorted({int(s) for s in segments})
        if segments[0] != 0:
            segments.insert(0, 0)
        if segments[-1] != n_features:
            segments.append(n_features)

        pairs = list(zip(segments[:-1], segments[1:]))
        references = []
        for start, end in pairs:
            block = X[:, start:end]
            ref_median = float(np.median(block))
            q75 = np.percentile(block, 75)
            q25 = np.percentile(block, 25)
            ref_iqr = float(q75 - q25)
            if not np.isfinite(ref_iqr) or ref_iqr == 0.0:
                ref_iqr = 1.0
            references.append((start, end, ref_median, ref_iqr))

        energy = np.sqrt(np.mean(diff**2, axis=1))
        energy_center = float(np.median(energy))
        energy_spread = float(
            median_abs_deviation(energy, scale="normal") or 0.0
        )
        if not np.isfinite(energy_spread) or energy_spread == 0.0:
            energy_spread = max(energy_center, 1.0)

        self._segment_pairs_ = pairs
        self._references_ = references
        self._energy_center_ = energy_center
        self._energy_spread_ = energy_spread
        self._n_features_ = n_features
        self._n_samples_fit_ = n_samples
        return self

    def transform(self, X):
        if not hasattr(self, "_references_"):
            raise RuntimeError("fit must be called before transform")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("AdaptiveQuantileSegmentNormalizer expects 2D input")
        if X.shape[1] != self._n_features_:
            raise ValueError(
                "Number of features during transform does not match fitted data"
            )

        Xt = X.copy()
        eps = np.finfo(float).eps
        for start, end, ref_median, ref_iqr in self._references_:
            segment = Xt[:, start:end]
            sample_median = np.median(segment, axis=1, keepdims=True)
            q75 = np.percentile(segment, 75, axis=1, keepdims=True)
            q25 = np.percentile(segment, 25, axis=1, keepdims=True)
            sample_iqr = q75 - q25
            sample_iqr = np.where(np.isfinite(sample_iqr), sample_iqr, 0.0)
            sample_iqr = np.where(np.abs(sample_iqr) < eps, 1.0, sample_iqr)
            normalized = (segment - sample_median) / sample_iqr
            Xt[:, start:end] = normalized * ref_iqr + ref_median

        if self.blend_strength > 0:
            diff = np.diff(X, axis=1)
            energy = np.sqrt(np.mean(diff**2, axis=1))
            spread = self._energy_spread_
            if not np.isfinite(spread) or spread == 0.0:
                spread = 1.0
            stability = (energy - self._energy_center_) / spread
            stability = np.clip(stability, -20, 20)
            weights = 1.0 / (1.0 + np.exp(self.blend_strength * stability))
            Xt = weights[:, None] * Xt + (1.0 - weights)[:, None] * X

        return Xt
