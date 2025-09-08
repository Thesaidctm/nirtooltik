import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger("nircls")


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure logging with separate console and file handlers."""

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = Console(stderr=True)
    handlers: list[logging.Handler] = [
        RichHandler(console=console, show_path=False, markup=True)
    ]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(message)s",
        force=True,
    )
    logging.getLogger("kaleido").setLevel(logging.WARNING)
    logging.getLogger("choreography").setLevel(logging.WARNING)


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
