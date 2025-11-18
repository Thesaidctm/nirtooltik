from __future__ import annotations

from pathlib import Path

from .cv_runner import explain_model as _explain_model

__all__ = ["explain_model"]


def explain_model(model_path: Path, data_path: Path, out_dir: Path) -> None:
    """Gera visualizações de explicabilidade para um modelo salvo."""

    return _explain_model(model_path, data_path, out_dir)
