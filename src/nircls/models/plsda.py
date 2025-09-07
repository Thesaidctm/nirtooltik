from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)


class PLSDA(BaseEstimator, ClassifierMixin):
    """Partial Least Squares Discriminant Analysis estimator.

    This wrapper combines :class:`PLSRegression` with a probabilistic
    classifier (logistic regression or LDA) while exposing parameters in a
    scikit-learn compatible way.
    """

    def __init__(self, n_components: int = 10, method: str = "logreg", C: float = 1.0):
        self.n_components = n_components
        self.method = method
        self.C = C
        self.pls = PLSRegression(n_components=n_components)
        self.clf = (
            LogisticRegression(max_iter=5000)
            if method == "logreg"
            else LinearDiscriminantAnalysis()
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDA":
        Y = np.eye(len(np.unique(y)))[y]
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
        if isinstance(self.clf, LogisticRegression):
            self.clf.set_params(C=self.C)
        self.clf.fit(T, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        T = self.pls.transform(X)
        return self.clf.predict(T)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        T = self.pls.transform(X)
        return self.clf.predict_proba(T)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"n_components": self.n_components, "method": self.method, "C": self.C}

    def set_params(self, **p: Any) -> "PLSDA":
        for k, v in p.items():
            setattr(self, k, v)
        self.pls = PLSRegression(n_components=self.n_components)
        self.clf = (
            LogisticRegression(max_iter=5000)
            if self.method == "logreg"
            else LinearDiscriminantAnalysis()
        )
        if isinstance(self.clf, LogisticRegression):
            self.clf.set_params(C=self.C)
        return self


__all__ = ["PLSDA"]

