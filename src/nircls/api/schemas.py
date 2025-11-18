from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    pass


class ProjectRead(ProjectBase):
    project_id: int
    owner_id: Optional[int] = None

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

    class Config:
        orm_mode = True


class RunBase(BaseModel):
    status: Literal["running", "completed", "failed", "cancelled"] = "running"


class RunCreate(RunBase):
    project_id: Optional[int] = None
    dataset_id: Optional[int] = None
    config: Dict[str, Any]
    artifacts: Optional[Dict[str, Any]] = None


class RunRead(RunBase):
    run_id: int
    project_id: Optional[int]
    dataset_id: Optional[int]
    user_id: Optional[int] = None
    config: Dict[str, Any]
    artifacts: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    class Config:
        orm_mode = True


class UserRead(BaseModel):
    user_id: int
    username: str
    email: str

    class Config:
        orm_mode = True
