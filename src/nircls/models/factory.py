from __future__ import annotations

import ast
from importlib.metadata import entry_points
from typing import Any, Dict, List

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from .base import BaseModelBuilder

# Helpers for model construction with optional class weighting
from .builders import _build_etc, _build_logreg, _build_rf, _build_svm
from .plsda import PLSDA

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


# Mapping from model identifier to human readable description. This dictionary
# is also populated by third-party plugins registered via entry points.
MODEL_DESCRIPTIONS: Dict[str, str] = {
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

_ENTRYPOINT_GROUP = "nircls.model_builders"


MODEL_BUILDERS = {
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


DEFAULT_GRIDS = {
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
        "device": ["cpu"],
    },
    "LSTM1D": {
        "hidden_dim": [64],
        "num_layers": [1],
        "n_epochs": [10],
        "patience": [0],
        "device": ["cpu"],
    },
    # Include a positive regularization parameter to handle rank-deficient
    # covariance matrices that frequently arise in high dimensional settings.
    # A value of 0.1 provides mild shrinkage while preserving the ability to
    # search for an unregularized solution (0.0).
    "QDA": {"reg_param": [0.0, 0.1]},
    "GAUSSIANNB": {"var_smoothing": [1e-09]},
    "NB": {"var_smoothing": [1e-09]},
    "MULTINB": {"alpha": [1.0]},
    "BERNOULLINB": {"alpha": [1.0]},
}


_EP_LOADED = False


def _load_entry_point_builders() -> None:  # pragma: no cover - simple metadata loader
    """Populate builder mappings from third-party entry points."""

    global _EP_LOADED
    if _EP_LOADED:
        return
    eps = entry_points()
    if hasattr(eps, "select"):
        group_eps = eps.select(group=_ENTRYPOINT_GROUP)
    else:  # Python<3.10
        group_eps = eps.get(_ENTRYPOINT_GROUP, [])
    for ep in group_eps:
        builder_cls = ep.load()
        try:
            builder: BaseModelBuilder = builder_cls()  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive
            continue
        name = builder.name.upper()
        MODEL_BUILDERS[name] = lambda grid, b=builder: b.build(grid)
        grid = builder.default_grid() or {}
        if grid:
            DEFAULT_GRIDS[name] = grid
        if getattr(builder, "description", ""):
            MODEL_DESCRIPTIONS[name] = builder.description
    _EP_LOADED = True


def get_model_builders() -> Dict[str, Any]:
    """Return mapping of available model builders.

    Third-party plugins are loaded lazily on first call.
    """

    _load_entry_point_builders()
    return MODEL_BUILDERS


def get_model_descriptions() -> Dict[str, str]:
    """Return mapping from model identifier to human readable description."""

    _load_entry_point_builders()
    return MODEL_DESCRIPTIONS


def _normalize_grid_values(grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Coerce YAML-loaded grid values to proper Python types.

    Nested lists representing tuples (e.g. ``[100]``) are converted to
    :class:`tuple`. String values are interpreted where possible using
    :func:`ast.literal_eval` so that numeric parameters provided as quoted
    values (e.g. ``"1e-09"``) are coerced to proper Python types. The grid is
    modified in-place and returned for convenience.
    """

    for param, values in list(grid.items()):
        normalized: List[Any] = []
        i = 0
        while i < len(values):
            val = values[i]
            if isinstance(val, list):
                normalized.append(tuple(val))
                i += 1
                continue
            if isinstance(val, str):
                if val.startswith("("):
                    # Handle tuple representations possibly split across
                    # multiple YAML tokens. Concatenate subsequent strings
                    # until a closing parenthesis is found before attempting
                    # to evaluate.
                    parts = [val]
                    if not val.endswith(")"):
                        i += 1
                        while i < len(values):
                            part = values[i]
                            parts.append(part)
                            if isinstance(part, str) and part.endswith(")"):
                                break
                            i += 1
                    joined = " ".join(str(p) for p in parts)
                    try:
                        normalized.append(ast.literal_eval(joined))
                    except Exception:
                        normalized.extend(parts)
                    i += 1
                    continue
                # Attempt to parse numeric strings (e.g. "1e-09")
                try:
                    normalized.append(ast.literal_eval(val))
                except Exception:
                    normalized.append(val)
                i += 1
                continue
            normalized.append(val)
            i += 1
        grid[param] = normalized
    return grid


def make_models_from_config(models_cfg: List[Dict[str, Any]]):
    builders = get_model_builders()
    out = []
    for m in models_cfg:
        m.setdefault("overfit_check", True)
        name = m["name"]
        upper = name.upper()

        builder = builders.get(upper)
        if builder is None:
            raise ValueError(f"Modelo não suportado: {name}")

        grid = _normalize_grid_values(m.get("grid", {}) or {})
        defaults = DEFAULT_GRIDS.get(upper, {})
        for k, v in defaults.items():
            grid.setdefault(k, v)

        est = builder(grid)

        params = m.get("params", {})
        if params:
            # Coerce any quoted numeric values to their Python equivalents
            for k, v in list(params.items()):
                if isinstance(v, str):
                    try:
                        params[k] = ast.literal_eval(v)
                    except Exception:
                        pass
            est.set_params(**params)

        out.append({"name": name, "estimator": est, "grid": grid})

    return out
