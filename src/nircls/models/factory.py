from __future__ import annotations

import ast
from importlib.metadata import entry_points
from typing import Any, Dict, List

from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import BaseModelBuilder


def _apply_random_state(estimator: Any, random_state: int | None) -> Any:
    if random_state is None:
        return estimator
    if not hasattr(estimator, "get_params"):
        return estimator
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        return estimator
    if "random_state" in params:
        try:
            estimator.set_params(random_state=random_state)
        except ValueError:
            pass
    return estimator


def _apply_n_jobs(estimator: Any, n_jobs: int | None) -> Any:
    """Set ``n_jobs``-style parameters on ``estimator`` when supported."""

    if n_jobs is None:
        return estimator
    if not hasattr(estimator, "get_params"):
        return estimator
    try:
        params = estimator.get_params(deep=False)
    except Exception:
        return estimator
    updated: Dict[str, Any] = {}
    if "n_jobs" in params:
        updated["n_jobs"] = n_jobs
    if "nthread" in params:
        updated["nthread"] = n_jobs
    if "thread_count" in params:
        updated["thread_count"] = n_jobs
    if not updated:
        return estimator
    try:
        estimator.set_params(**updated)
    except ValueError:
        return estimator
    return estimator

# Helpers for model construction with optional class weighting
from .builders import _build_etc, _build_logreg, _build_rf, _build_svm
from .plsda import PLSDA
from .plslocal import PLSLocalRegressor

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

REGRESSION_MODEL_DESCRIPTIONS: Dict[str, str] = {
    "PLSREGRESSION": "Partial Least Squares Regression",
    "PLSR": "Partial Least Squares Regression",
    "PLS_REG": "Partial Least Squares Regression",
    "PLSLOCALREGRESSOR": "Partial Least Squares with local regression",
    "PLSLOCAL_REG": "Partial Least Squares with local regression",
    "PLS_LOCAL_REG": "Partial Least Squares with local regression",
    "RANDOMFORESTREGRESSOR": "Random Forest Regressor",
    "RFR": "Random Forest Regressor",
    "RF_REG": "Random Forest Regressor",
    "GRADIENTBOOSTINGREGRESSOR": "Gradient Boosting Regressor",
    "GB_REG": "Gradient Boosting Regressor",
    "EXTRATREESREGRESSOR": "Extra Trees Regressor",
    "ETR": "Extra Trees Regressor",
    "ET_REG": "Extra Trees Regressor",
    "ADABOOSTREGRESSOR": "AdaBoost Regressor",
    "ADABOOST_REG": "AdaBoost Regressor",
    "SVR": "Support Vector Regressor",
    "SVR_REG": "Support Vector Regressor",
    "SVR_LINEAR_REG": "Support Vector Regressor (Linear)",
    "SVR_RBF_REG": "Support Vector Regressor (RBF)",
    "RIDGE": "Ridge Regression",
    "LINEARREGRESSION": "Linear Regression",
    "LINEAR_REG": "Linear Regression",
    "KNNR": "k-Nearest Neighbors Regressor",
    "KNEIGHBORSREGRESSOR": "k-Nearest Neighbors Regressor",
    "KNN_REG": "k-Nearest Neighbors Regressor",
    "MLPREGRESSOR": "Multilayer Perceptron Regressor",
    "MLP_REG": "Multilayer Perceptron Regressor",
    "DECISIONTREEREGRESSOR": "Decision Tree Regressor",
    "DT_REG": "Decision Tree Regressor",
}

MODEL_DESCRIPTIONS: Dict[str, str] = CLASSIFICATION_MODEL_DESCRIPTIONS

_ENTRYPOINT_GROUP = "nircls.model_builders"


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

REGRESSION_MODEL_BUILDERS = {
    "PLSREGRESSION": lambda grid: PLSRegression(),
    "PLSR": lambda grid: PLSRegression(),
    "PLS_REG": lambda grid: PLSRegression(),
    "PLSLOCALREGRESSOR": lambda grid: PLSLocalRegressor(),
    "PLSLOCAL_REG": lambda grid: PLSLocalRegressor(),
    "PLS_LOCAL_REG": lambda grid: PLSLocalRegressor(),
    "RANDOMFORESTREGRESSOR": lambda grid: RandomForestRegressor(),
    "RFR": lambda grid: RandomForestRegressor(),
    "RF_REG": lambda grid: RandomForestRegressor(),
    "GRADIENTBOOSTINGREGRESSOR": lambda grid: GradientBoostingRegressor(),
    "GB_REG": lambda grid: GradientBoostingRegressor(),
    "EXTRATREESREGRESSOR": lambda grid: ExtraTreesRegressor(),
    "ETR": lambda grid: ExtraTreesRegressor(),
    "ET_REG": lambda grid: ExtraTreesRegressor(),
    "ADABOOSTREGRESSOR": lambda grid: AdaBoostRegressor(),
    "ADABOOST_REG": lambda grid: AdaBoostRegressor(),
    "SVR": lambda grid: SVR(),
    "SVR_REG": lambda grid: SVR(),
    "SVR_LINEAR_REG": lambda grid: SVR(kernel="linear"),
    "SVR_RBF_REG": lambda grid: SVR(kernel="rbf"),
    "RIDGE": lambda grid: Ridge(),
    "LINEARREGRESSION": lambda grid: LinearRegression(),
    "LINEAR_REG": lambda grid: LinearRegression(),
    "KNNR": lambda grid: KNeighborsRegressor(),
    "KNEIGHBORSREGRESSOR": lambda grid: KNeighborsRegressor(),
    "KNN_REG": lambda grid: KNeighborsRegressor(),
    "MLPREGRESSOR": lambda grid: MLPRegressor(),
    "MLP_REG": lambda grid: MLPRegressor(),
    "DECISIONTREEREGRESSOR": lambda grid: DecisionTreeRegressor(),
    "DT_REG": lambda grid: DecisionTreeRegressor(),
}

MODEL_BUILDERS = CLASSIFICATION_MODEL_BUILDERS


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
        "device": [None],
    },
    "LSTM1D": {
        "hidden_dim": [64],
        "num_layers": [1],
        "n_epochs": [10],
        "patience": [0],
        "device": [None],
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

REGRESSION_DEFAULT_GRIDS = {
    "PLSREGRESSION": {"n_components": [2]},
    "PLSR": {"n_components": [2]},
    "PLS_REG": {"n_components": [2]},
    "PLSLOCALREGRESSOR": {
        "n_components": [2],
        "method": ["knn"],
        "n_neighbors": [5],
    },
    "PLSLOCAL_REG": {
        "n_components": [2],
        "method": ["knn"],
        "n_neighbors": [5],
    },
    "PLS_LOCAL_REG": {
        "n_components": [2],
        "method": ["knn"],
        "n_neighbors": [5],
    },
    "RANDOMFORESTREGRESSOR": {"n_estimators": [100], "max_depth": [None]},
    "RFR": {"n_estimators": [100], "max_depth": [None]},
    "RF_REG": {"n_estimators": [100], "max_depth": [None]},
    "GRADIENTBOOSTINGREGRESSOR": {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [3],
    },
    "GB_REG": {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]},
    "EXTRATREESREGRESSOR": {"n_estimators": [100], "max_depth": [None]},
    "ETR": {"n_estimators": [100], "max_depth": [None]},
    "ET_REG": {"n_estimators": [100], "max_depth": [None]},
    "ADABOOSTREGRESSOR": {"n_estimators": [50], "learning_rate": [1.0]},
    "ADABOOST_REG": {"n_estimators": [50], "learning_rate": [1.0]},
    "SVR": {"kernel": ["rbf"], "C": [1.0]},
    "SVR_REG": {"kernel": ["rbf"], "C": [1.0]},
    "SVR_LINEAR_REG": {"kernel": ["linear"], "C": [1.0]},
    "SVR_RBF_REG": {"kernel": ["rbf"], "C": [1.0]},
    "RIDGE": {"alpha": [1.0]},
    "LINEARREGRESSION": {},
    "LINEAR_REG": {},
    "KNNR": {"n_neighbors": [5]},
    "KNEIGHBORSREGRESSOR": {"n_neighbors": [5]},
    "KNN_REG": {"n_neighbors": [5]},
    "MLPREGRESSOR": {"max_iter": [500], "early_stopping": [True]},
    "MLP_REG": {"max_iter": [500], "early_stopping": [True]},
    "DECISIONTREEREGRESSOR": {"max_depth": [None]},
    "DT_REG": {"max_depth": [None]},
}

CLASSIFICATION_DEFAULT_GRIDS = DEFAULT_GRIDS

TASK_MODEL_BUILDERS: Dict[str, Dict[str, Any]] = {
    "classification": CLASSIFICATION_MODEL_BUILDERS,
    "regression": REGRESSION_MODEL_BUILDERS,
}

TASK_DEFAULT_GRIDS: Dict[str, Dict[str, Dict[str, List[Any]]]] = {
    "classification": CLASSIFICATION_DEFAULT_GRIDS,
    "regression": REGRESSION_DEFAULT_GRIDS,
}

TASK_MODEL_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "classification": CLASSIFICATION_MODEL_DESCRIPTIONS,
    "regression": REGRESSION_MODEL_DESCRIPTIONS,
}

TASK_SUFFIXES = {"classification": "_CLS", "regression": "_REG"}


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


def _normalize_task_type(task_type: str | None) -> str:
    normalized = str(task_type or "classification").strip().lower()
    if normalized == "regression":
        return "regression"
    return "classification"


def get_model_builders(task_type: str = "classification") -> Dict[str, Any]:
    """Return mapping of available model builders for ``task_type``."""

    normalized = _normalize_task_type(task_type)
    if normalized == "classification":
        _load_entry_point_builders()
    return TASK_MODEL_BUILDERS[normalized]


def get_model_descriptions(task_type: str = "classification") -> Dict[str, str]:
    """Return mapping from model identifier to human readable description."""

    normalized = _normalize_task_type(task_type)
    if normalized == "classification":
        _load_entry_point_builders()
    return TASK_MODEL_DESCRIPTIONS[normalized]


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


def make_models_from_config(
    models_cfg: List[Dict[str, Any]],
    *,
    task_type: str = "classification",
    random_state: int | None = None,
    n_jobs: int | None = None,
):
    normalized_type = _normalize_task_type(task_type)
    builders = get_model_builders(normalized_type)
    defaults = TASK_DEFAULT_GRIDS[normalized_type]
    suffix = TASK_SUFFIXES[normalized_type]
    other_suffix = (
        TASK_SUFFIXES["regression"]
        if normalized_type == "classification"
        else TASK_SUFFIXES["classification"]
    )
    out = []
    for m in models_cfg:
        m.setdefault("overfit_check", True)
        name = m["name"]
        upper = name.upper()

        resolved_key = upper
        builder = builders.get(resolved_key)

        def _attempt(candidate_key: str) -> tuple[str, Any] | tuple[None, None]:
            candidate_builder = builders.get(candidate_key)
            if candidate_builder is None:
                return None, None
            return candidate_key, candidate_builder

        if builder is None and not resolved_key.endswith(suffix):
            candidate_key, candidate_builder = _attempt(f"{resolved_key}{suffix}")
            if candidate_builder is not None:
                resolved_key, builder = candidate_key, candidate_builder
        if builder is None and other_suffix:
            if resolved_key.endswith(other_suffix):
                base = resolved_key[: -len(other_suffix)]
                candidate_key, candidate_builder = _attempt(base)
                if candidate_builder is None and suffix:
                    candidate_key, candidate_builder = _attempt(f"{base}{suffix}")
                if candidate_builder is not None:
                    resolved_key, builder = candidate_key, candidate_builder
            else:
                candidate_key, candidate_builder = _attempt(f"{resolved_key}{other_suffix}")
                if candidate_builder is not None:
                    resolved_key, builder = candidate_key, candidate_builder
        if builder is None:
            raise ValueError(f"Modelo não suportado: {name}")

        grid = _normalize_grid_values(m.get("grid", {}) or {})
        default_grid = defaults.get(resolved_key, {})
        for k, v in default_grid.items():
            grid.setdefault(k, v)

        est = builder(grid)
        est = _apply_random_state(est, random_state)
        est = _apply_n_jobs(est, n_jobs)

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
