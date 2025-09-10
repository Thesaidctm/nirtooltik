import os
import smtplib
from email.mime.text import MIMEText

from .logging import logger

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
EMAIL_FROM = os.environ.get("EMAIL_FROM")
EMAIL_TO = os.environ.get("EMAIL_TO")
EMAIL_PASS = os.environ.get("EMAIL_PASS")


def enviar_email(assunto: str, corpo: str) -> None:

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from .logging import logger


def _load_config() -> dict:
    """Load email credentials from env vars or a JSON config file.

    Environment variables override file values. If ``NIRCLS_EMAIL_CONFIG`` is
    defined and points to a JSON file, it will be read. Default config file is
    ``~/.nircls_email.json``.
    """
    cfg: dict = {}
    cfg_path = os.getenv("NIRCLS_EMAIL_CONFIG")
    if cfg_path is None:
        default_path = Path.home() / ".nircls_email.json"
        if default_path.exists():
            cfg_path = str(default_path)
    if cfg_path:
        try:
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Could not read email config", exc_info=True)
            cfg = {}
    # Environment variables take precedence
    cfg.update(
        {
            "host": os.getenv("NIRCLS_EMAIL_HOST", cfg.get("host")),
            "port": os.getenv("NIRCLS_EMAIL_PORT", cfg.get("port")),
            "user": os.getenv("NIRCLS_EMAIL_USER", cfg.get("user")),
            "password": os.getenv("NIRCLS_EMAIL_PASS", cfg.get("password")),
            "from_addr": os.getenv("NIRCLS_EMAIL_FROM", cfg.get("from_addr")),
            "to_addr": os.getenv("NIRCLS_EMAIL_TO", cfg.get("to_addr")),
            "use_tls": os.getenv("NIRCLS_EMAIL_TLS", str(cfg.get("use_tls", "1"))),
        }
    )
    return cfg


def enviar_email(subject: str, body: str, to_addr: Optional[str] = None) -> None:

    """Send an email notification.

    Parameters
    ----------
    assunto: str
        Email subject line.
    corpo: str
        Email body text.
    """

    if not all([SMTP_SERVER, SMTP_PORT, EMAIL_FROM, EMAIL_TO, EMAIL_PASS]):
        logger.debug("Skipping email; SMTP configuration incomplete")
        return

    msg = MIMEText(corpo)
    msg["Subject"] = assunto
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.send_message(msg)
    except Exception:
        logger.exception("Failed to send email")

    subject: str
        Email subject.
    body: str
        Plain text body of the message.
    to_addr: str, optional
        Destination address. If not provided, ``NIRCLS_EMAIL_TO`` or the value
        from the configuration file will be used.

    The following environment variables are recognised:
    ``NIRCLS_EMAIL_HOST``, ``NIRCLS_EMAIL_PORT``, ``NIRCLS_EMAIL_USER``,
    ``NIRCLS_EMAIL_PASS``, ``NIRCLS_EMAIL_FROM`` and ``NIRCLS_EMAIL_TO``.
    A JSON configuration file can also be provided via ``NIRCLS_EMAIL_CONFIG``.
    """
    cfg = _load_config()
    host = cfg.get("host")
    port = cfg.get("port")
    user = cfg.get("user")
    password = cfg.get("password")
    from_addr = cfg.get("from_addr") or user
    to = to_addr or cfg.get("to_addr")
    use_tls = str(cfg.get("use_tls", "1")) not in {"0", "false", "False"}

    if not all([host, port, user, password, to]):  # pragma: no cover - best effort
        logger.debug("Incomplete email configuration; skipping notification")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    msg.set_content(body)

    try:  # pragma: no cover - network operations
        with smtplib.SMTP(host, int(port)) as server:
            if use_tls:
                server.starttls()
            server.login(user, password)
            server.send_message(msg)
    except Exception:  # pragma: no cover - best effort
        logger.warning("Failed to send email notification", exc_info=True)

