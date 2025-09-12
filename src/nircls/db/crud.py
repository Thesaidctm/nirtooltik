from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from .models import Role, User, Variavel

# User CRUD operations


def create_user(session: Session, username: str, email: str) -> User:
    user = User(username=username, email=email)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def get_user(session: Session, user_id: int) -> Optional[User]:
    return session.get(User, user_id)


def update_user(
    session: Session,
    user_id: int,
    *,
    username: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[User]:
    user = session.get(User, user_id)
    if not user:
        return None
    if username is not None:
        user.username = username
    if email is not None:
        user.email = email
    session.commit()
    session.refresh(user)
    return user


def delete_user(session: Session, user_id: int) -> None:
    user = session.get(User, user_id)
    if user:
        session.delete(user)
        session.commit()


# Role CRUD operations


def create_role(session: Session, name: str) -> Role:
    role = Role(name=name)
    session.add(role)
    session.commit()
    session.refresh(role)
    return role


def get_role(session: Session, role_id: int) -> Optional[Role]:
    return session.get(Role, role_id)


def update_role(
    session: Session, role_id: int, *, name: Optional[str] = None
) -> Optional[Role]:
    role = session.get(Role, role_id)
    if not role:
        return None
    if name is not None:
        role.name = name
    session.commit()
    session.refresh(role)
    return role


def delete_role(session: Session, role_id: int) -> None:
    role = session.get(Role, role_id)
    if role:
        session.delete(role)
        session.commit()


# Variaveis operations


def get_variavel(session: Session, var_id: int) -> Optional[Variavel]:
    return session.get(Variavel, var_id)


def set_variavel(session: Session, nome: str, var_id: Optional[int] = None) -> Variavel:
    if var_id is not None:
        variavel = session.get(Variavel, var_id)
        if variavel:
            variavel.nome = nome
        else:
            variavel = Variavel(var_id=var_id, nome=nome)
            session.add(variavel)
    else:
        variavel = Variavel(nome=nome)
        session.add(variavel)
    session.commit()
    session.refresh(variavel)
    return variavel
