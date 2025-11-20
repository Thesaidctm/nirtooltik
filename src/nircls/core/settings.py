"""Centralized infrastructure settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application-wide infrastructure configuration."""

    DATABASE_URL: str = Field("sqlite:///nircls.db", env="DATABASE_URL")
    JWT_SECRET_KEY: str = Field("nircls-secret", env=("JWT_SECRET_KEY", "SESSION_SECRET"))
    JWT_ALGORITHM: str = Field("HS256", env=("JWT_ALGORITHM", "SESSION_ALGORITHM"))
    ACCESS_TOKEN_EXPIRE_SECONDS: int = Field(
        3600, env=("ACCESS_TOKEN_EXPIRE_SECONDS", "SESSION_TTL_SECONDS")
    )
    ENVIRONMENT: str = Field("dev", env=("ENVIRONMENT", "NIRCLS_ENV", "APP_ENV"))
    DATA_STORAGE_DIR: Path = Field(Path("storage"), env="DATA_STORAGE_DIR")
    RESULTS_DIR: Path = Field(Path("results"), env="NIRCLS_RESULTS")
    RAW_DIR: Path = Field(Path("data/RAW"), env="NIRCLS_RAW")
    MLFLOW_TRACKING_URI: str | None = Field(None, env="MLFLOW_TRACKING_URI")
    STREAMLIT_SERVER_PORT: int = Field(8501, env="STREAMLIT_SERVER_PORT")
    EMAIL_CONFIG_PATH: Path | None = Field(None, env="NIRCLS_EMAIL_CONFIG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

__all__ = ["Settings", "settings"]
