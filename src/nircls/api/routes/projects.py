from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db
from ...db import crud
from ...db.models import Project, User

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectResponse(ProjectCreate):
    project_id: int
    owner_id: Optional[int] = None

    class Config:
        orm_mode = True


@router.get("", response_model=List[ProjectResponse])
def list_all_projects(db: Session = Depends(get_db)) -> List[Project]:
    return crud.list_projects(db)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project_detail(project_id: int, db: Session = Depends(get_db)) -> Project:
    project = crud.get_project(db, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_new_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Project:
    return crud.create_project(
        db,
        name=payload.name,
        description=payload.description,
        owner_id=current_user.user_id,
    )
