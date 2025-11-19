import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RunLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("run_uid", self.extra.get("run_uid"))
        return msg, kwargs


def attach_run_file_handler(
    logger: logging.Logger,
    log_path: Path,
) -> logging.Handler:
    """
    Adiciona um FileHandler ao logger apontando para log_path.
    Retorna o handler criado para permitir remoção posterior.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [run=%(run_uid)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return handler


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure logging with a standard format including module names."""

    class _RunIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
            if not hasattr(record, "run_uid"):
                record.run_uid = "-"
            return True

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
    logging.getLogger().addFilter(_RunIdFilter())


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
