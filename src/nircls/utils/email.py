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
