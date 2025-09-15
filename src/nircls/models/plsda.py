from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


_CONSTRUCTORS: Dict[str, Callable[[], ClassifierMixin]] = {
    "logreg": lambda: LogisticRegression(max_iter=5000),
    "lda": lambda: LinearDiscriminantAnalysis(),
    "qda": lambda: QuadraticDiscriminantAnalysis(),
    "svm": lambda: SVC(probability=True),
    "knn": lambda: KNeighborsClassifier(),
}


class PLSDA(BaseEstimator, ClassifierMixin):
    """Partial Least Squares Discriminant Analysis estimator.

    The PLS step reduces dimensionality before applying a probabilistic
    classifier such as logistic regression, LDA, QDA, SVM or kNN. Classifier
    hyperparameters can be tuned via the regular scikit-learn grid search API.
    """

    def __init__(
        self, n_components: int = 10, method: str = "logreg", **clf_params: Any
    ):
        self.n_components = n_components
        self.method = method
        self.clf_params = clf_params
        self.pls = PLSRegression(n_components=n_components)
        self.clf = self._make_clf()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDA":
        classes, y_int = np.unique(y, return_inverse=True)
        self.classes_ = classes
        Y = np.eye(len(classes))[y_int]
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
        self.clf.fit(T, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        T = self.pls.transform(X)
        return self.clf.predict(T)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        T = self.pls.transform(X)
        return self.clf.predict_proba(T)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {"n_components": self.n_components, "method": self.method}
        params.update(self.clf_params)
        return params

    def set_params(self, **p: Any) -> "PLSDA":
        for k, v in p.items():
            if k in {"n_components", "method"}:
                setattr(self, k, v)
            else:
                self.clf_params[k] = v
        self.pls = PLSRegression(n_components=self.n_components)
        self.clf = self._make_clf()
        return self

    def _make_clf(self) -> ClassifierMixin:
        try:
            ctor = _CONSTRUCTORS[self.method]
        except KeyError as e:
            raise ValueError(f"Unsupported method: {self.method}") from e
        clf = ctor()
        if self.clf_params:
            allowed = clf.get_params()
            clf.set_params(**{k: v for k, v in self.clf_params.items() if k in allowed})
        return clf


__all__ = ["PLSDA"]
