from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///nircls.db"

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db() -> None:
    """Create database tables."""
    # Import models so that they are registered with SQLAlchemy's metadata
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
