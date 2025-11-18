"""Regression model registry and defaults segmented from classification flows."""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .plslocal import PLSLocalRegressor

__all__ = [
    "REGRESSION_DEFAULT_GRIDS",
    "REGRESSION_MODEL_BUILDERS",
    "REGRESSION_MODEL_DESCRIPTIONS",
]


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


REGRESSION_DEFAULT_GRIDS: Dict[str, Dict[str, List[Any]]] = {
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
