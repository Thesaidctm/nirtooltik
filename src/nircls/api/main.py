from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import inspect

from ..core.services.users import ensure_dev_admin_user
from ..core.settings import settings
from ..db.database import SessionLocal
from ..utils.errors import AuthError, ConfigError, DataError, DatabaseError, ModelError, PipelineError
from ..utils.logging import setup_logging
from .routes import (
    auth,
    calibration_transfers,
    datasets,
    instruments,
    models,
    organizations,
    projects,
    runs,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="nircls API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _configure_logging() -> None:
    setup_logging()
    logger.info("FastAPI application initialized")


@app.on_event("startup")
async def _ensure_dev_user() -> None:
    if settings.ENVIRONMENT.lower() != "dev":
        return

    db = SessionLocal()
    try:
        bind = db.get_bind()
        inspector = inspect(bind)
        missing_tables = [table for table in ["users", "roles"] if not inspector.has_table(table)]
        if missing_tables:
            logger.warning(
                "Skipping dev admin creation; missing tables: %s", ", ".join(missing_tables)
            )
            return
        ensure_dev_admin_user(db)
    finally:
        db.close()


def _format_error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.exception_handler(DataError)
async def data_error_handler(request: Request, exc: DataError) -> JSONResponse:  # noqa: ARG001
    return _format_error_response(status.HTTP_400_BAD_REQUEST, str(exc))


@app.exception_handler(ConfigError)
async def config_error_handler(request: Request, exc: ConfigError) -> JSONResponse:  # noqa: ARG001
    return _format_error_response(status.HTTP_400_BAD_REQUEST, str(exc))


@app.exception_handler(AuthError)
async def auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:  # noqa: ARG001
    return _format_error_response(status.HTTP_401_UNAUTHORIZED, str(exc))


@app.exception_handler(ModelError)
@app.exception_handler(PipelineError)
async def pipeline_error_handler(request: Request, exc: ModelError | PipelineError) -> JSONResponse:  # noqa: ARG001
    return _format_error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))


@app.exception_handler(DatabaseError)
async def database_error_handler(request: Request, exc: DatabaseError) -> JSONResponse:  # noqa: ARG001
    logger.error("Database error", exc_info=exc)
    return _format_error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal database error")


app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(organizations.router)
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(instruments.router, prefix="/instruments", tags=["instruments"])
app.include_router(
    calibration_transfers.router,
    prefix="/calibration-transfers",
    tags=["calibration_transfers"],
)
