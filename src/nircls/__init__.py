"""Top-level package for nircls.

This module exposes key public functions and defines the package version
using ``importlib.metadata`` when available.
"""

from pathlib import Path
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python<3.8 fallback
    from importlib_metadata import PackageNotFoundError, version

try:  # pragma: no cover - runtime convenience
    __version__ = version("nircls")
except PackageNotFoundError:  # pragma: no cover - development fallback
    __version__ = "0.0.0"

from .pipeline import build_pipeline_from_config, train_and_save, run_experiment

DEFAULT_RESULTS_DIR = Path.home() / "nircls_results"

__all__ = [
    "build_pipeline_from_config",
    "train_and_save",
    "run_experiment",
    "DEFAULT_RESULTS_DIR",
]

