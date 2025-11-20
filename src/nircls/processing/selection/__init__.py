"""Variable selection routines for PLS and related methods."""

from .pls_variable_selection import (
    PLSVariableSelectionResult,
    apply_pls_variable_selection,
    pls_variable_selection,
)

__all__ = [
    "PLSVariableSelectionResult",
    "apply_pls_variable_selection",
    "pls_variable_selection",
]
