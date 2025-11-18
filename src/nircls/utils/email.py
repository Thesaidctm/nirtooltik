"""Helpers for sending email notifications.

The credentials and target addresses are resolved from environment
variables or a JSON configuration file as handled by :func:`_load_config`.
"""

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import keyring

from .logging import logger
from ..core.settings import settings


def _load_config() -> dict:
    """Load email credentials from env vars or a JSON config file.

    Environment variables override file values. If ``NIRCLS_EMAIL_CONFIG`` is
    defined and points to a JSON file, it will be read. The default config file
    location is ``~/.nircls_email.json``. Besides email credentials, the
    configuration may include ``results_dir`` used by the dashboard.
    """

    cfg: dict = {}
    cfg_path = settings.EMAIL_CONFIG_PATH
    if cfg_path is None:
        default_path = Path.home() / ".nircls_email.json"
        if default_path.exists():
            cfg_path = default_path
    if cfg_path:
        try:
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            cfg.pop("password", None)
        except Exception:  # pragma: no cover - best effort
            logger.debug("Could not read email config", exc_info=True)
            cfg = {}
    cfg.update(
        {
            "host": os.getenv("NIRCLS_EMAIL_HOST", cfg.get("host")),
            "port": os.getenv("NIRCLS_EMAIL_PORT", cfg.get("port")),
            "user": os.getenv("NIRCLS_EMAIL_USER", cfg.get("user")),
            "from_addr": os.getenv("NIRCLS_EMAIL_FROM", cfg.get("from_addr")),
            "to_addr": os.getenv("NIRCLS_EMAIL_TO", cfg.get("to_addr")),
            "use_tls": os.getenv("NIRCLS_EMAIL_TLS", str(cfg.get("use_tls", "1"))),
            "results_dir": cfg.get("results_dir", str(settings.RESULTS_DIR)),
        }
    )

    password = os.getenv("NIRCLS_EMAIL_PASS")
    if not password and cfg.get("user"):
        try:  # pragma: no cover - best effort
            password = keyring.get_password("nircls", cfg["user"])
        except Exception:
            logger.debug("Could not read password from keyring", exc_info=True)
    if password:
        cfg["password"] = password
    return cfg


def _save_config(cfg: dict) -> None:
    """Persist configuration to ``~/.nircls_email.json``.

    Parameters
    ----------
    cfg:
        Dictionary with configuration fields to save.
    """

    password = cfg.pop("password", None)
    cfg_path = settings.EMAIL_CONFIG_PATH or (Path.home() / ".nircls_email.json")
    try:  # pragma: no cover - best effort
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - best effort
        logger.debug("Could not write email config", exc_info=True)
    if password and cfg.get("user"):
        try:  # pragma: no cover - best effort
            keyring.set_password("nircls", cfg["user"], password)
        except Exception:  # pragma: no cover - best effort
            logger.debug("Could not store password in keyring", exc_info=True)


def enviar_email(subject: str, body: str, to_addr: str) -> None:
    """Send an email notification using credentials from :func:`_load_config`.

    Parameters
    ----------
    subject, body:
        Email subject and message body.
    to_addr:
        Destination address.
    """

    cfg = _load_config()
    host = cfg.get("host")
    port = cfg.get("port")
    user = cfg.get("user")
    password = cfg.get("password")
    from_addr = cfg.get("from_addr") or user
    use_tls = str(cfg.get("use_tls", "1")) not in {"0", "false", "False"}

    if not all([host, port, user, password, to_addr]):  # pragma: no cover - best effort
        logger.debug("Incomplete email configuration; skipping notification")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    try:  # pragma: no cover - network operations
        with smtplib.SMTP(host, int(port)) as server:
            if use_tls:
                server.starttls()
            server.login(user, password)
            server.send_message(msg)
    except Exception:  # pragma: no cover - best effort
        logger.warning("Failed to send email notification", exc_info=True)
