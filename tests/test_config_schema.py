import pytest

from nircls.config_schema import ConfigModel, Palette, ValidationError


def test_notification_email_validation():
    # valid email
    cfg = ConfigModel(notification_email="user@example.com")
    assert cfg.notification_email == "user@example.com"
    # invalid email
    with pytest.raises(ValidationError):
        ConfigModel(notification_email="not-an-email")


def test_palette_enum():
    cfg = ConfigModel(palette="colorblind")
    assert cfg.palette is Palette.COLORBLIND
    with pytest.raises(ValueError):
        ConfigModel(palette="invalid")
