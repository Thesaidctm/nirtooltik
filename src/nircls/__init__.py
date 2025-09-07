"""Top-level package for nircls.

This module exposes key public functions and defines the package version
using ``pkg_resources`` when available.
"""

from pathlib import Path
from pkg_resources import DistributionNotFound, get_distribution

try:  # pragma: no cover - runtime convenience
    __version__ = get_distribution("nircls").version
except DistributionNotFound:  # pragma: no cover - development fallback
    __version__ = "0.1.0"

from .pipeline import build_pipeline_from_config, train_and_save, run_experiment

DEFAULT_RESULTS_DIR = Path.home() / "nircls_results"

__all__ = [
    "build_pipeline_from_config",
    "train_and_save",
    "run_experiment",
    "DEFAULT_RESULTS_DIR",
]

