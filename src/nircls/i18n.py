"""Simple internationalization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

_LOCALE_DIR = Path(__file__).parent / "locales"


def _load_locale(lang: str) -> dict[str, str]:
    try:
        with (_LOCALE_DIR / f"{lang}.json").open() as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


TRANSLATIONS: dict[str, dict[str, str]] = {
    lang: _load_locale(lang) for lang in ("pt", "en")
}


def tr(key: str) -> str:
    """Translate *key* using current language from ``st.session_state``.

    Falls back to Portuguese or the key itself when no translation is found.
    """
    lang = st.session_state.get("language", "pt")
    return TRANSLATIONS.get(lang, {}).get(key, TRANSLATIONS.get("pt", {}).get(key, key))
