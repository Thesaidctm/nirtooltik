from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from ..utils.errors import AuthError, ConfigError, DataError, DatabaseError, ModelError, PipelineError
from ..utils.logging import setup_logging
from .routes import auth, datasets, models, organizations, projects, runs

logger = logging.getLogger(__name__)

app = FastAPI(title="nircls API")


@app.on_event("startup")
async def _configure_logging() -> None:
    setup_logging()
    logger.info("FastAPI application initialized")


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
