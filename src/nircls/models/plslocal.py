from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

logger = logging.getLogger(__name__)


_REGRESSOR_BUILDERS: Dict[str, Callable[[], RegressorMixin]] = {
    "knn": lambda: KNeighborsRegressor(),
    "radius": lambda: RadiusNeighborsRegressor(),
}


class PLSLocalRegressor(BaseEstimator, RegressorMixin):
    """Partial Least Squares regression followed by a local regressor.

    The PLS stage reduces dimensionality prior to applying a local method such
    as k-nearest neighbours or radius neighbours regression. Hyper-parameters
    of the local estimator can be tuned via the standard scikit-learn grid
    search API.
    """

    def __init__(self, n_components: int = 2, method: str = "knn", **reg_params: Any):
        self.n_components = n_components
        self.method = method
        self.reg_params = dict(reg_params)
        self.pls = PLSRegression(n_components=n_components)
        self.regressor = self._make_regressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSLocalRegressor":
        Y = np.asarray(y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[0] != X.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        self._y_columns = Y.shape[1]
        self.pls = PLSRegression(n_components=self.n_components)
        self.pls.fit(X, Y)
        self.n_components_ = self.pls.x_scores_.shape[1]
        max_comp = min(X.shape[0] - 1, X.shape[1])
        if self.n_components_ > 0.8 * max_comp:
            logger.warning(
                "Using %s PLS components may overfit (limit is %s)",
                self.n_components_,
                max_comp,
            )
        T = self.pls.transform(X)
        target = Y if self._y_columns > 1 else Y.ravel()
        self.regressor = self._make_regressor()
        self.regressor.fit(T, target)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        T = self.pls.transform(X)
        preds = self.regressor.predict(T)
        preds = np.asarray(preds)
        if getattr(self, "_y_columns", 1) == 1:
            return preds.reshape(-1)
        return preds.reshape(-1, self._y_columns)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {"n_components": self.n_components, "method": self.method}
        params.update(self.reg_params)
        return params

    def set_params(self, **params: Any) -> "PLSLocalRegressor":
        updated_reg_params = dict(self.reg_params)
        for key, value in params.items():
            if key in {"n_components", "method"}:
                setattr(self, key, value)
            else:
                updated_reg_params[key] = value
        self.reg_params = updated_reg_params
        self.pls = PLSRegression(n_components=self.n_components)
        self.regressor = self._make_regressor()
        return self

    def _make_regressor(self) -> RegressorMixin:
        try:
            builder = _REGRESSOR_BUILDERS[self.method]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported local regression method: {self.method}") from exc
        reg = builder()
        if self.reg_params:
            allowed = reg.get_params()
            valid_params = {k: v for k, v in self.reg_params.items() if k in allowed}
            if valid_params:
                reg.set_params(**valid_params)
        return reg


__all__ = ["PLSLocalRegressor"]
