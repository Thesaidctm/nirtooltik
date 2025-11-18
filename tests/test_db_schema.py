from __future__ import annotations

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from nircls.db import (
    Base,
    create_sample,
    get_sample,
    replace_sample_targets,
    set_sample_features,
)


def test_xy_l_tables_exist_with_expected_columns() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    inspector = inspect(engine)

    table_names = set(inspector.get_table_names())
    assert {"x", "y", "l"}.issubset(table_names)

    l_columns = {c["name"] for c in inspector.get_columns("l")}
    assert {"sample_id", "external_id", "description", "metadata"}.issubset(l_columns)

    x_columns = {c["name"] for c in inspector.get_columns("x")}
    assert {"x_id", "sample_id", "values"}.issubset(x_columns)

    y_columns = {c["name"] for c in inspector.get_columns("y")}
    assert {"y_id", "sample_id", "target_name", "target_value"}.issubset(y_columns)

    x_fk = inspector.get_foreign_keys("x")
    assert x_fk and x_fk[0]["referred_table"] == "l"

    y_fk = inspector.get_foreign_keys("y")
    assert y_fk and y_fk[0]["referred_table"] == "l"


def test_crud_roundtrip_for_xy_l_tables() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with SessionLocal() as session:
        sample = create_sample(
            session,
            external_id="amostra_001",
            description="tronco A",
            metadata={"site": "A1"},
        )
        set_sample_features(session, sample.sample_id, values={"1100.0": 0.45})
        replace_sample_targets(
            session,
            sample.sample_id,
            targets=[("especie", "Eucalipto"), ("densidade", "0.88")],
        )

        persisted = get_sample(session, sample.sample_id)
        assert persisted is not None
        assert persisted.features is not None
        assert persisted.features.values["1100.0"] == 0.45

        target_names = {t.target_name for t in persisted.targets}
        assert target_names == {"especie", "densidade"}


def test_optional_l_metadata_stub_is_created_on_demand() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    with SessionLocal() as session:
        features = set_sample_features(
            session,
            sample_id=None,
            values={"1100.0": 0.12, "2200.0": 0.98},
        )

        replace_sample_targets(
            session,
            features.sample_id,
            targets=[("especie", "Eucalipto")],
        )

        reloaded = get_sample(session, features.sample_id)
        assert reloaded is not None
        assert reloaded.external_id is None
        assert reloaded.features is not None
        assert set(reloaded.features.values.keys()) == {"1100.0", "2200.0"}
