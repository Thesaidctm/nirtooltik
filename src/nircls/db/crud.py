from __future__ import annotations

from typing import Dict, Iterable, Optional

from sqlalchemy.orm import Session

from .models import (
    Role,
    SampleFeatures,
    SampleMetadata,
    SampleTarget,
    User,
    Variavel,
)

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


# Sample operations (X, Y, L tables)


def _get_or_create_sample(
    session: Session,
    sample_id: Optional[int],
    *,
    external_id: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict] = None,
    allow_stub_metadata: bool = True,
) -> SampleMetadata:
    sample = session.get(SampleMetadata, sample_id) if sample_id else None
    if not sample and not allow_stub_metadata:
        raise ValueError(f"Sample {sample_id} not found")

    if not sample:
        sample = SampleMetadata(
            sample_id=sample_id,
            external_id=external_id,
            description=description,
            metadata_json=metadata,
        )
        session.add(sample)
        session.flush()
    else:
        if external_id is not None:
            sample.external_id = external_id
        if description is not None:
            sample.description = description
        if metadata is not None:
            sample.metadata_json = metadata
    return sample


def create_sample(
    session: Session,
    *,
    external_id: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> SampleMetadata:
    sample = _get_or_create_sample(
        session,
        None,
        external_id=external_id,
        description=description,
        metadata=metadata,
        allow_stub_metadata=True,
    )
    session.commit()
    session.refresh(sample)
    return sample


def set_sample_features(
    session: Session,
    sample_id: Optional[int],
    values: Dict[str, float],
    *,
    external_id: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict] = None,
    allow_stub_metadata: bool = True,
) -> SampleFeatures:
    sample = _get_or_create_sample(
        session,
        sample_id,
        external_id=external_id,
        description=description,
        metadata=metadata,
        allow_stub_metadata=allow_stub_metadata,
    )

    if sample.features:
        sample.features.values = values
        features = sample.features
    else:
        features = SampleFeatures(sample=sample, values=values)
        session.add(features)

    session.commit()
    session.refresh(features)
    return features


def replace_sample_targets(
    session: Session,
    sample_id: Optional[int],
    targets: Iterable[tuple[str, str]],
    *,
    external_id: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict] = None,
    allow_stub_metadata: bool = True,
) -> list[SampleTarget]:
    sample = _get_or_create_sample(
        session,
        sample_id,
        external_id=external_id,
        description=description,
        metadata=metadata,
        allow_stub_metadata=allow_stub_metadata,
    )

    sample.targets.clear()
    for target_name, target_value in targets:
        sample.targets.append(
            SampleTarget(target_name=target_name, target_value=target_value)
        )

    session.commit()
    session.refresh(sample)
    return list(sample.targets)


def get_sample(session: Session, sample_id: int) -> Optional[SampleMetadata]:
    return session.get(SampleMetadata, sample_id)
