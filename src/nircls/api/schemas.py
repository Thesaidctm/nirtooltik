from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    pass


class ProjectRead(ProjectBase):
    project_id: int
    owner_id: Optional[int] = None
    organization_id: Optional[int] = None

    class Config:
        orm_mode = True


class DatasetBase(BaseModel):
    name: str
    version: Optional[str] = None
    description: Optional[str] = None


class DatasetCreate(DatasetBase):
    project_id: Optional[int] = None
    storage_path: Optional[str] = None


class DatasetRead(DatasetBase):
    dataset_id: int
    project_id: Optional[int]
    created_by_id: Optional[int] = None
    storage_path: Optional[str] = None
    organization_id: Optional[int] = None

    class Config:
        orm_mode = True


class RunBase(BaseModel):
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    run_uid: Optional[str] = None


class RunCreate(RunBase):
    project_id: Optional[int] = None
    dataset_id: Optional[int] = None
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    target_cols: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    artifacts: Optional[Dict[str, Any]] = None
    auto_execute: bool = False


class RunRead(RunBase):
    run_id: int
    project_id: Optional[int]
    dataset_id: Optional[int]
    user_id: Optional[int] = None
    config: Dict[str, Any]
    artifacts: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    organization_id: Optional[int] = None

    class Config:
        orm_mode = True


class UserRead(BaseModel):
    user_id: int
    username: str
    email: str
    organization_id: Optional[int] = None

    class Config:
        orm_mode = True


class ModelBase(BaseModel):
    name: str
    target: Optional[str] = None
    status: Literal["candidate", "production", "deprecated"] = "candidate"


class ModelCreate(ModelBase):
    run_id: int
    dataset_id: Optional[int] = None


class ModelRead(ModelBase):
    id: int
    run_id: int
    dataset_id: Optional[int]

    class Config:
        orm_mode = True


class PredictRequest(BaseModel):
    spectra: List[List[float]]


class PredictResponse(BaseModel):
    y_hat: List[float]


class OrganizationCreate(BaseModel):
    name: str


class OrganizationRead(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True


class RunMetricsResponse(BaseModel):
    metrics: Dict[str, Any]


class RunPlotsResponse(BaseModel):
    plots: List[str]


class InstrumentBase(BaseModel):
    name: str
    code: str
    vendor: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None


class InstrumentCreate(InstrumentBase):
    pass


class InstrumentRead(InstrumentBase):
    id: int

    class Config:
        orm_mode = True


class CalibrationTransferBase(BaseModel):
    name: str
    master_model_id: int
    master_instrument_id: int
    slave_instrument_id: int
    method: Literal["ds", "pds"] = "ds"
    standard_dataset_master_id: Optional[int] = None
    standard_dataset_slave_id: Optional[int] = None


class CalibrationTransferCreate(CalibrationTransferBase):
    pass


class CalibrationTransferRead(CalibrationTransferBase):
    id: int
    artifacts: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True
