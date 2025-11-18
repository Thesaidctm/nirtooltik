"""Interfaces for external model plugins.

This module defines :class:`BaseModelBuilder`, a minimal protocol that third
party packages can implement and expose via Python ``entry_points`` to extend
``nircls`` with new estimators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from sklearn.base import BaseEstimator


class BaseModelBuilder(ABC):
    """Simple interface for model builders.

    Subclasses must define a :attr:`name` attribute and implement
    :meth:`build`. Optional :attr:`description` and :meth:`default_grid` allow
    integration with the CLI and configuration files.
    """

    #: Identifier used in configuration files and CLI.
    name: str
    #: Human readable description (optional).
    description: str = ""

    @abstractmethod
    def build(self, grid: Dict[str, Any]) -> BaseEstimator:
        """Create and return the model instance."""

    def default_grid(self) -> Dict[str, Any]:  # pragma: no cover - simple default
        """Return default hyperparameter grid.

        Implementations may override this to provide sensible defaults used
        when constructing the search space for the estimator.
        """

        return {}


__all__ = ["BaseModelBuilder"]

