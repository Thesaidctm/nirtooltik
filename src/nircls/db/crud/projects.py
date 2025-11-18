from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from ..models import Project


def list_projects(session: Session) -> List[Project]:
    return session.query(Project).all()


def get_project(session: Session, project_id: int) -> Optional[Project]:
    return session.get(Project, project_id)


def create_project(
    session: Session,
    *,
    name: str,
    description: str | None = None,
    owner_id: int | None = None,
) -> Project:
    project = Project(name=name, description=description, owner_id=owner_id)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project
