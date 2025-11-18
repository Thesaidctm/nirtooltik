"""Model builders and utilities used in nircls processing pipelines."""

from .classification import (
    CLASSIFICATION_DEFAULT_GRIDS,
    CLASSIFICATION_MODEL_BUILDERS,
    CLASSIFICATION_MODEL_DESCRIPTIONS,
)
from .factory import MODEL_DESCRIPTIONS, get_model_builders, get_model_descriptions, make_models_from_config
from .regression import (
    REGRESSION_DEFAULT_GRIDS,
    REGRESSION_MODEL_BUILDERS,
    REGRESSION_MODEL_DESCRIPTIONS,
)

__all__ = [
    "CLASSIFICATION_DEFAULT_GRIDS",
    "CLASSIFICATION_MODEL_BUILDERS",
    "CLASSIFICATION_MODEL_DESCRIPTIONS",
    "MODEL_DESCRIPTIONS",
    "REGRESSION_DEFAULT_GRIDS",
    "REGRESSION_MODEL_BUILDERS",
    "REGRESSION_MODEL_DESCRIPTIONS",
    "get_model_builders",
    "get_model_descriptions",
    "make_models_from_config",
]
