from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

from nircls.config_schema import (
    CVConfig,
    ConfigModel,
    DataConfig,
    KFoldStrategy,
    ModelConfig,
    PreprocessConfig,
)
from nircls.db.models import Dataset


def build_run_config_from_dataset(
    dataset: Dataset,
    *,
    target_cols: List[str],
    model_names: List[str],
    task_type: str = "regression",  # ou "classification"
    preprocess_profile: str = "DEFAULT",
    cv_method: Literal["stratified", "spxy", "group"] = "stratified",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Monta um ConfigModel coerente usando o storage do Dataset.
    Retorna como dict pronto para ser salvo em Run.config.
    """

    storage_path = dataset.storage_path
    if not storage_path:
        raise ValueError("Dataset.storage_path must be set before building config")

    storage_dir = Path(storage_path)
    # Por enquanto, assuma que o arquivo principal é 'data.csv' no diretório do dataset
    data_file = storage_dir / "data.csv"

    data_cfg = DataConfig(
        input_path=str(data_file),
        target_cols=target_cols,
        target_type=task_type,
        out_dir=None,  # a pipeline decide o diretório de saída com base em settings.RESULTS_DIR
    )

    preprocess_cfg = PreprocessConfig(
        steps=[preprocess_profile] if preprocess_profile else [],
    )

    cv_cfg = CVConfig(
        random_state=random_state,
        cv_method=cv_method,
        strategies=[KFoldStrategy(type="kfold", n_splits=n_splits)],
    )

    models_cfg = [ModelConfig(name=name) for name in model_names]

    cfg_model = ConfigModel(
        data=data_cfg,
        preprocess=preprocess_cfg,
        cv=cv_cfg,
        models=models_cfg,
    )
    # Retorne como dict serializável para ser armazenado em Run.config (JSON)
    return cfg_model.model_dump()
