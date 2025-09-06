
from __future__ import annotations
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import numpy as np

# Optional third-party estimators
try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - handle any import error
    XGBClassifier = None

try:  # pragma: no cover
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:  # pragma: no cover
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


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
        upper = name.upper()
        if upper == "PLSDA":
            est = PLSDA()
        elif upper == "SVM_RBF":
            est = SVC(probability=True)
        elif upper == "SVM_LINEAR":
            est = SVC(kernel="linear", probability=True)
        elif upper == "RF":
            est = RandomForestClassifier()
        elif upper == "GB":
            est = GradientBoostingClassifier()
        elif upper in ("ET", "EXTRATREES"):
            est = ExtraTreesClassifier()
        elif upper == "ADABOOST":
            est = AdaBoostClassifier()
        elif upper == "LOGREG":
            est = LogisticRegression(max_iter=5000)
        elif upper == "QDA":
            est = QuadraticDiscriminantAnalysis()
        elif upper == "XGBOOST":
            if XGBClassifier is None:
                raise ImportError(
                    "xgboost is not installed. Install with `pip install xgboost` to use XGBOOST."
                )
            est = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        elif upper == "LGBM":
            if LGBMClassifier is None:
                raise ImportError(
                    "lightgbm is not installed. Install with `pip install lightgbm` to use LGBM."
                )
            est = LGBMClassifier()
        elif upper == "CATBOOST":
            if CatBoostClassifier is None:
                raise ImportError(
                    "catboost is not installed. Install with `pip install catboost` to use CATBOOST."
                )
            est = CatBoostClassifier(verbose=False)
        elif upper == "MLP":
            est = MLPClassifier()
        elif upper == "DT":
            est = DecisionTreeClassifier()
        elif upper == "KNN":
            est = KNeighborsClassifier()
        elif upper == "CNN1D":
            from .neural import SpectralCNN  # lazy import
            est = SpectralCNN()
        elif upper == "LSTM1D":
            from .neural import SpectralLSTM  # lazy import
            est = SpectralLSTM()
        elif upper in ("GAUSSIANNB", "NB"):
            est = GaussianNB()
        elif upper == "MULTINB":
            est = MultinomialNB()
        elif upper == "BERNOULLINB":
            est = BernoulliNB()
        else:
            raise ValueError(f"Modelo não suportado: {name}")
        params = m.get("params", {})
        if params:
            est.set_params(**params)
        grid = m.get("grid", {})
        if upper == "LOGREG":
            grid.setdefault("C", [1.0])
        elif upper == "ADABOOST":
            grid.setdefault("n_estimators", [50])
            grid.setdefault("learning_rate", [1.0])
        elif upper == "XGBOOST":
            grid.setdefault("n_estimators", [100])
            grid.setdefault("max_depth", [3])
            grid.setdefault("learning_rate", [0.1])
        elif upper == "LGBM":
            grid.setdefault("n_estimators", [100])
            grid.setdefault("num_leaves", [31])
            grid.setdefault("learning_rate", [0.1])
        elif upper == "CATBOOST":
            grid.setdefault("iterations", [100])
            grid.setdefault("depth", [6])
            grid.setdefault("learning_rate", [0.1])
        elif upper == "CNN1D":
            grid.setdefault("n_filters", [32])
            grid.setdefault("kernel_size", [3])
            grid.setdefault("n_epochs", [10])
        elif upper == "LSTM1D":
            grid.setdefault("hidden_dim", [64])
            grid.setdefault("num_layers", [1])
            grid.setdefault("n_epochs", [10])
        elif upper == "QDA":
            grid.setdefault("reg_param", [0.0])
        elif upper in ("GAUSSIANNB", "NB"):
            grid.setdefault("var_smoothing", [1e-09])
        elif upper == "MULTINB":
            grid.setdefault("alpha", [1.0])
        elif upper == "BERNOULLINB":
            grid.setdefault("alpha", [1.0])
        out.append({"name": name, "estimator": est, "grid": grid})
    return out
