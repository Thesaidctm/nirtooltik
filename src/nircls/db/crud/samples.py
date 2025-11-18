from __future__ import annotations

from typing import Dict, Iterable, Optional

from sqlalchemy.orm import Session

from ..models import SampleFeatures, SampleMetadata, SampleTarget


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
