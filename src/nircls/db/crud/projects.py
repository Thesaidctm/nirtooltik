from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from ..models import Project


def list_projects(session: Session, *, organization_id: Optional[int] = None) -> List[Project]:
    query = session.query(Project)
    if organization_id is not None:
        query = query.filter(Project.organization_id == organization_id)
    return query.all()


def get_project(
    session: Session, project_id: int, *, organization_id: Optional[int] = None
) -> Optional[Project]:
    project = session.get(Project, project_id)
    if organization_id is not None and project and project.organization_id != organization_id:
        return None
    return project


def create_project(
    session: Session,
    *,
    name: str,
    description: str | None = None,
    owner_id: int | None = None,
    organization_id: int | None = None,
) -> Project:
    project = Project(
        name=name,
        description=description,
        owner_id=owner_id,
        organization_id=organization_id,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return project
