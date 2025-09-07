"""Pydantic models describing expected configuration fields."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError, root_validator

from .utils.misc import RESULTS_DIR


class DataConfig(BaseModel):
    """Dataset related configuration."""

    input_path: str = Field(..., description="Glob or file path for input data")
    x_sheet: Optional[str] = Field("X", description="Worksheet name for features")
    y_sheet: Optional[str] = Field("Y", description="Worksheet name for targets")
    target_cols: Optional[Union[List[str], str]] = Field(
        None, description="Target columns or 'all'"
    )
    target_col: Optional[str] = Field(
        None, description="Alias for single target column"
    )
    group_col: Optional[str] = Field(
        None, description="Column name with group labels for grouped CV"
    )
    spectral_range: Optional[Union[List[float], tuple[float, float]]] = Field(
        None, description="Lower and upper wavelength limits"
    )
    drop_nans: Optional[bool] = True
    out_dir: Optional[str] = Field(
        default_factory=lambda: str(RESULTS_DIR),
        description="Base output directory",
    )
    qc: bool = False

    @root_validator(pre=True)
    def _normalise_target(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "target_col" in values and "target_cols" not in values:
            values["target_cols"] = values["target_col"]
        return values

    class Config:
        extra = "forbid"


class SGConfig(BaseModel):
    window: Union[int, List[int]] = 15
    polyorder: Union[int, List[int]] = 2
    deriv: Union[int, List[int]] = 1

    class Config:
        extra = "forbid"


class PreprocessConfig(BaseModel):
    steps: List[str] = Field(default_factory=list)
    sg: Optional[SGConfig] = None

    class Config:
        extra = "forbid"


class KFoldStrategy(BaseModel):
    type: Literal["kfold"]
    n_splits: Union[int, List[int]]
    n_repeats: Optional[int] = 1

    class Config:
        extra = "forbid"


class LooStrategy(BaseModel):
    type: Literal["loo"]

    class Config:
        extra = "forbid"


class HoldoutStrategy(BaseModel):
    type: Literal["holdout"]
    test_size: float

    class Config:
        extra = "forbid"


CVStrategy = Union[KFoldStrategy, LooStrategy, HoldoutStrategy]


class CVConfig(BaseModel):
    random_state: Optional[int] = 42
    inner_folds: Optional[int] = None
    cv_method: Optional[Literal["stratified", "spxy", "group"]] = "stratified"
    strategies: List[CVStrategy] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class CalibConfig(BaseModel):
    method: Literal["sigmoid", "isotonic"] = "sigmoid"
    cv: int = 5

    class Config:
        extra = "forbid"


class ModelConfig(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = None
    grid: Optional[Dict[str, Any]] = None
    calibration: Optional[CalibConfig] = None
    overfit_check: bool = False

    class Config:
        extra = "forbid"


class ConfigModel(BaseModel):
    data: Optional[DataConfig] = None
    preprocess: Optional[PreprocessConfig] = None
    cv: Optional[CVConfig] = None
    models: Optional[List[ModelConfig]] = None
    astm: bool = False
    use_cache: bool = True

    class Config:
        extra = "forbid"


__all__ = [
    "DataConfig",
    "PreprocessConfig",
    "CVConfig",
    "CalibConfig",
    "ModelConfig",
    "ConfigModel",
    "ValidationError",
]
