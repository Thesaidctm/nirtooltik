"""Top-level package for nircls.

This module exposes key public functions and defines the package version
based on ``pyproject.toml``.
"""

from importlib.metadata import PackageNotFoundError, version as _version
from pathlib import Path

try:  # pragma: no cover - runtime convenience
    __version__ = _version("nircls")
except PackageNotFoundError:  # pragma: no cover - development fallback
    import tomllib

    project_root = Path(__file__).resolve().parents[2]
    with (project_root / "pyproject.toml").open("rb") as f:
        __version__ = tomllib.load(f)["project"]["version"]

from .pipeline import build_pipeline_from_config, train_and_save

DEFAULT_RESULTS_DIR = Path.home() / "nircls_results"

__all__ = ["build_pipeline_from_config", "train_and_save", "DEFAULT_RESULTS_DIR"]

