import pytest

from nircls.config_schema import ConfigModel, ValidationError


def test_notification_email_validation():
    # valid email
    cfg = ConfigModel(notification_email="user@example.com")
    assert cfg.notification_email == "user@example.com"
    # invalid email
    with pytest.raises(ValidationError):
        ConfigModel(notification_email="not-an-email")
