from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ... import i18n
from ..deps import get_current_user, get_db, require_role
from ..schemas import ProjectCreate, ProjectRead
from ...db import crud
from ...db.models import Project, User

router = APIRouter(prefix="/projects", tags=["projects"])

@router.get("", response_model=List[ProjectRead])
def list_all_projects(db: Session = Depends(get_db)) -> List[Project]:
    return crud.list_projects(db)


@router.get("/{project_id}", response_model=ProjectRead)
def get_project_detail(project_id: int, db: Session = Depends(get_db)) -> Project:
    project = crud.get_project(db, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=i18n.t("projects.not_found_by_id").format(project_id=project_id),
        )
    return project


@router.post(
    "",
    response_model=ProjectRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role("admin", "researcher"))],
)
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
