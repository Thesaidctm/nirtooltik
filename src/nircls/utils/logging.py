
import logging

logger = logging.getLogger("nircls")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for the project.

    Args:
        level: Logging level used by the root logger.
    """

    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


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
