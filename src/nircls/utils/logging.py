import logging
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure logging with a standard format including module names."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)


def log_versions() -> None:
    """Log versions of key third-party libraries."""

    logger = logging.getLogger(__name__)
    try:
        import numpy
        import pandas
        import sklearn

        logger.info(
            "Versions | numpy %s | pandas %s | sklearn %s",
            numpy.__version__,
            pandas.__version__,
            sklearn.__version__,
        )
    except Exception:
        logger.error("Version log failed", exc_info=True)
