"""Tests for email utilities."""

import smtplib
from unittest.mock import MagicMock

from nircls.utils.email import enviar_email


def test_enviar_email(monkeypatch):
    """Email is sent using configuration from environment variables."""

    monkeypatch.setenv("NIRCLS_EMAIL_HOST", "smtp.test")
    monkeypatch.setenv("NIRCLS_EMAIL_PORT", "25")
    monkeypatch.setenv("NIRCLS_EMAIL_USER", "user@test")
    monkeypatch.setenv("NIRCLS_EMAIL_PASS", "pass")
    monkeypatch.setenv("NIRCLS_EMAIL_FROM", "from@test")
    monkeypatch.setenv("NIRCLS_EMAIL_TO", "to@test")

    smtp_mock = MagicMock()
    smtp_cls = MagicMock(return_value=smtp_mock)
    smtp_mock.__enter__.return_value = smtp_mock
    monkeypatch.setattr(smtplib, "SMTP", smtp_cls)

    enviar_email("subject", "body")

    smtp_cls.assert_called_once_with("smtp.test", 25)
    smtp_mock.starttls.assert_called_once()
    smtp_mock.login.assert_called_once_with("user@test", "pass")
    smtp_mock.send_message.assert_called_once()
    message = smtp_mock.send_message.call_args[0][0]
    assert message["Subject"] == "subject"
    assert message["From"] == "from@test"
    assert message["To"] == "to@test"
    assert "body" in message.get_content()
