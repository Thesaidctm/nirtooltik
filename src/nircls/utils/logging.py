
import logging
import sys
from pathlib import Path

logger = logging.getLogger("nircls")


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure basic logging for the project.

    Args:
        level: Logging level used by the root logger.
        log_file: Optional path to a log file.
    """

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=handlers, force=True)


def log_versions() -> None:
    """Log versions of key third-party libraries."""

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
    except Exception as exc:
        logger.warning("Version log failed: %s", exc)
