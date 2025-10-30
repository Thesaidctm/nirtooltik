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


def test_preprocess_aqs_config_allows_lists():
    cfg = ConfigModel(
        preprocess={
            "steps": ["AQS"],
            "aqs": {
                "min_segment_size": [12, 20],
                "blend_strength": [2.0, 5.0],
            },
        }
    )
    assert cfg.preprocess is not None
    assert cfg.preprocess.aqs is not None
    assert cfg.preprocess.aqs.min_segment_size == [12, 20]
    assert cfg.preprocess.aqs.blend_strength == [2.0, 5.0]
