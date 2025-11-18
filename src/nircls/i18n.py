"""Simple internationalization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

_LOCALE_DIR = Path(__file__).parent / "locales"


def _load_locale(lang: str) -> dict[str, Any]:
    try:
        with (_LOCALE_DIR / f"{lang}.json").open() as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


TRANSLATIONS: dict[str, dict[str, Any]] = {
    lang: _load_locale(lang) for lang in ("pt", "en")
}


def _get_nested(translations: dict[str, Any], key: str) -> str | None:
    current: Any = translations
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current if isinstance(current, str) else None


def t(key: str) -> str:
    """Translate *key* using current language from ``st.session_state``.

    Supports nested keys separated by dots (e.g. ``"auth.invalid"``).
    Falls back to Portuguese or the key itself when no translation is found.
    """
    lang = st.session_state.get("language", "pt")
    translation = _get_nested(TRANSLATIONS.get(lang, {}), key)
    if translation is not None:
        return translation
    fallback = _get_nested(TRANSLATIONS.get("pt", {}), key)
    return fallback if fallback is not None else key


def tr(key: str) -> str:
    """Backward-compatible alias for :func:`t`."""
    return t(key)
