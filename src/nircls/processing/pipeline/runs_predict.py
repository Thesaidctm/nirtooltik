from __future__ import annotations

from pathlib import Path

from .cv_runner import predict_with_model as _predict_with_model

__all__ = ["predict_with_model"]


def predict_with_model(model_path: Path, data_path: Path, out_dir: Path) -> None:
    """Gera previsões com um modelo salvo usando a implementação compartilhada."""

    return _predict_with_model(model_path, data_path, out_dir)
