"""Classification model registry and defaults grouped for clarity."""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from .builders import _build_etc, _build_logreg, _build_rf, _build_svm
from .plsda import PLSDA

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


__all__ = [
    "CLASSIFICATION_MODEL_BUILDERS",
    "CLASSIFICATION_MODEL_DESCRIPTIONS",
    "CLASSIFICATION_DEFAULT_GRIDS",
]


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


CLASSIFICATION_MODEL_DESCRIPTIONS: Dict[str, str] = {
    "PLSDA": "Partial Least Squares Discriminant Analysis",
    "SVM_RBF": "Support Vector Machine (RBF)",
    "SVM_LINEAR": "Support Vector Machine (Linear)",
    "RF": "Random Forest",
    "GB": "Gradient Boosting",
    "ET": "Extra Trees",
    "ADABOOST": "AdaBoost",
    "LOGREG": "Logistic Regression",
    "QDA": "Quadratic Discriminant Analysis",
    "XGBOOST": "XGBoost",
    "LGBM": "LightGBM",
    "CATBOOST": "CatBoost",
    "MLP": "Multilayer Perceptron",
    "DT": "Decision Tree",
    "KNN": "k-Nearest Neighbors",
    "CNN1D": "1D Convolutional Neural Network",
    "LSTM1D": "1D LSTM Network",
    "GAUSSIANNB": "Gaussian Naive Bayes",
    "NB": "Gaussian Naive Bayes",
    "MULTINB": "Multinomial Naive Bayes",
    "BERNOULLINB": "Bernoulli Naive Bayes",
}


CLASSIFICATION_MODEL_BUILDERS = {
    "PLSDA": lambda grid: PLSDA(),
    "SVM_RBF": lambda grid: _build_svm(grid, kernel="rbf"),
    "SVM_LINEAR": lambda grid: _build_svm(grid, kernel="linear"),
    "RF": _build_rf,
    "GB": lambda grid: GradientBoostingClassifier(),
    "ET": _build_etc,
    "EXTRATREES": _build_etc,
    "ADABOOST": lambda grid: AdaBoostClassifier(),
    "LOGREG": _build_logreg,
    "QDA": lambda grid: QuadraticDiscriminantAnalysis(),
    "XGBOOST": lambda grid: _xgb_builder(),
    "LGBM": lambda grid: _lgbm_builder(),
    "CATBOOST": lambda grid: _catboost_builder(),
    "MLP": lambda grid: MLPClassifier(),
    "DT": lambda grid: DecisionTreeClassifier(),
    "KNN": lambda grid: KNeighborsClassifier(),
    "CNN1D": lambda grid: _cnn_builder(),
    "LSTM1D": lambda grid: _lstm_builder(),
    "GAUSSIANNB": lambda grid: GaussianNB(),
    "NB": lambda grid: GaussianNB(),
    "MULTINB": lambda grid: MultinomialNB(),
    "BERNOULLINB": lambda grid: BernoulliNB(),
}


CLASSIFICATION_DEFAULT_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "PLSDA": {
        "n_components": [10],
        "method": ["logreg"],
        "C": [1.0],
        "reg_param": [0.0],
        "n_neighbors": [5],
    },
    "LOGREG": {"C": [1.0]},
    "MLP": {"max_iter": [500, 1000], "early_stopping": [True]},
    "ADABOOST": {"n_estimators": [50], "learning_rate": [1.0]},
    "XGBOOST": {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1]},
    "LGBM": {"n_estimators": [100], "num_leaves": [31], "learning_rate": [0.1]},
    "CATBOOST": {"iterations": [100], "depth": [6], "learning_rate": [0.1]},
    "CNN1D": {
        "n_filters": [32],
        "kernel_size": [3],
        "n_epochs": [10],
        "patience": [0],
        "device": [None],
    },
    "LSTM1D": {
        "hidden_dim": [64],
        "num_layers": [1],
        "n_epochs": [10],
        "patience": [0],
        "device": [None],
    },
    "QDA": {"reg_param": [0.0, 0.1]},
    "GAUSSIANNB": {"var_smoothing": [1e-09]},
    "NB": {"var_smoothing": [1e-09]},
    "MULTINB": {"alpha": [1.0]},
    "BERNOULLINB": {"alpha": [1.0]},
}
