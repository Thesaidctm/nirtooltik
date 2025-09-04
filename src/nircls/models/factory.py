
from __future__ import annotations
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class PLSDA(BaseEstimator, ClassifierMixin):
    """Partial Least Squares Discriminant Analysis estimator.

    A minimal wrapper combining :class:`PLSRegression` with a probabilistic
    classifier (logistic regression or LDA). Parameters are exposed in a
    scikit-learn compatible way so the estimator can be used inside pipelines
    and grid-search routines.
    """

    def __init__(self, n_components=10, method="logreg", C=1.0):
        self.n_components = n_components
        self.method = method
        self.C = C
        self.pls = PLSRegression(n_components=n_components)
        self.clf = (
            LogisticRegression(max_iter=5000)
            if method == "logreg"
            else LinearDiscriminantAnalysis()
        )

    def fit(self, X, y):
        Y = np.eye(len(np.unique(y)))[y]
        self.pls.fit(X, Y)
        T = self.pls.transform(X)
        if isinstance(self.clf, LogisticRegression):
            self.clf.set_params(C=self.C)
        self.clf.fit(T, y)
        return self

    def predict(self, X):
        T = self.pls.transform(X)
        return self.clf.predict(T)

    def predict_proba(self, X):
        T = self.pls.transform(X)
        return self.clf.predict_proba(T)

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "method": self.method, "C": self.C}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        # Recreate inner models with updated parameters
        self.pls = PLSRegression(n_components=self.n_components)
        self.clf = (
            LogisticRegression(max_iter=5000)
            if self.method == "logreg"
            else LinearDiscriminantAnalysis()
        )
        if isinstance(self.clf, LogisticRegression):
            self.clf.set_params(C=self.C)
        return self

def make_models_from_config(models_cfg: List[Dict[str, Any]]):
    out = []
    for m in models_cfg:
        name = m["name"]
        if name.upper()=="PLSDA": est = PLSDA()
        elif name.upper()=="SVM_RBF": est = SVC(probability=True)
        elif name.upper()=="SVM_LINEAR": est = SVC(kernel="linear", probability=True)
        elif name.upper()=="RF": est = RandomForestClassifier()
        elif name.upper()=="GB": est = GradientBoostingClassifier()
        elif name.upper()=="MLP": est = MLPClassifier()
        else: raise ValueError(f"Modelo não suportado: {name}")
        out.append({"name": name, "estimator": est, "grid": m.get("grid", {})})
    return out
