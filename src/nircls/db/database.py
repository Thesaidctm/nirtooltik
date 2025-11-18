from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

from ..core.settings import settings


def _create_engine(database_url: str):
    url = make_url(database_url)
    connect_args = {}
    if url.get_backend_name() == "sqlite":
        connect_args["check_same_thread"] = False
    return create_engine(url, future=True, connect_args=connect_args)


engine = _create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def _alembic_config_path() -> Path:
    return Path(__file__).resolve().with_name("alembic.ini")


def init_db() -> None:
    """Run Alembic migrations to ensure the schema is up to date."""

    try:
        from alembic import command  # type: ignore
        from alembic.config import Config  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Alembic is required to run database migrations. Install with `pip install alembic`."
        ) from exc

    alembic_cfg = Config(str(_alembic_config_path()))
    alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
    command.upgrade(alembic_cfg, "head")
