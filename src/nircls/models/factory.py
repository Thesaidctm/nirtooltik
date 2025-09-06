
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


def _xgb_builder():
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not installed. Install with `pip install xgboost` to use XGBOOST."
        )
    return XGBClassifier(use_label_encoder=False, eval_metric="logloss")


def _lgbm_builder():
    if LGBMClassifier is None:
        raise ImportError(
            "lightgbm is not installed. Install with `pip install lightgbm` to use LGBM."
        )
    return LGBMClassifier()


def _catboost_builder():
    if CatBoostClassifier is None:
        raise ImportError(
            "catboost is not installed. Install with `pip install catboost` to use CATBOOST."
        )
    return CatBoostClassifier(verbose=False)


def _cnn_builder():  # pragma: no cover - lazy import
    from .neural import SpectralCNN

    return SpectralCNN()


def _lstm_builder():  # pragma: no cover - lazy import
    from .neural import SpectralLSTM

    return SpectralLSTM()


def _extratrees_builder():
    return ExtraTreesClassifier()


MODEL_BUILDERS = {
    "PLSDA": lambda: PLSDA(),
    "SVM_RBF": lambda: SVC(probability=True),
    "SVM_LINEAR": lambda: SVC(kernel="linear", probability=True),
    "RF": lambda: RandomForestClassifier(),
    "GB": lambda: GradientBoostingClassifier(),
    "ET": _extratrees_builder,
    "EXTRATREES": _extratrees_builder,
    "ADABOOST": lambda: AdaBoostClassifier(),
    "LOGREG": lambda: LogisticRegression(max_iter=5000),
    "QDA": lambda: QuadraticDiscriminantAnalysis(),
    "XGBOOST": _xgb_builder,
    "LGBM": _lgbm_builder,
    "CATBOOST": _catboost_builder,
    "MLP": lambda: MLPClassifier(),
    "DT": lambda: DecisionTreeClassifier(),
    "KNN": lambda: KNeighborsClassifier(),
    "CNN1D": _cnn_builder,
    "LSTM1D": _lstm_builder,
    "GAUSSIANNB": lambda: GaussianNB(),
    "NB": lambda: GaussianNB(),
    "MULTINB": lambda: MultinomialNB(),
    "BERNOULLINB": lambda: BernoulliNB(),
}


DEFAULT_GRIDS = {
    "LOGREG": {"C": [1.0]},
    "ADABOOST": {"n_estimators": [50], "learning_rate": [1.0]},
    "XGBOOST": {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1]},
    "LGBM": {"n_estimators": [100], "num_leaves": [31], "learning_rate": [0.1]},
    "CATBOOST": {"iterations": [100], "depth": [6], "learning_rate": [0.1]},
    "CNN1D": {"n_filters": [32], "kernel_size": [3], "n_epochs": [10]},
    "LSTM1D": {"hidden_dim": [64], "num_layers": [1], "n_epochs": [10]},
    "QDA": {"reg_param": [0.0]},
    "GAUSSIANNB": {"var_smoothing": [1e-09]},
    "NB": {"var_smoothing": [1e-09]},
    "MULTINB": {"alpha": [1.0]},
    "BERNOULLINB": {"alpha": [1.0]},
}

def make_models_from_config(models_cfg: List[Dict[str, Any]]):
    out = []
    for m in models_cfg:
        name = m["name"]
        upper = name.upper()

        builder = MODEL_BUILDERS.get(upper)
        if builder is None:
            raise ValueError(f"Modelo não suportado: {name}")
        est = builder()

        params = m.get("params", {})
        if params:
            est.set_params(**params)

        grid = m.get("grid", {})
        defaults = DEFAULT_GRIDS.get(upper, {})
        for k, v in defaults.items():
            grid.setdefault(k, v)

        out.append({"name": name, "estimator": est, "grid": grid})

    return out
