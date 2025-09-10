from __future__ import annotations

"""Utilities for checking package updates."""

import requests

LATEST_URL = "https://raw.githubusercontent.com/Thesaidctm/nirtooltik/main/latest.json"


def _parse(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def check_latest_version(current: str) -> str | None:
    """Return latest available version from remote endpoint.

    Parameters
    ----------
    current:
        Current version string.

    Returns
    -------
    str | None
        Newer version string if available, otherwise ``None``.
    """

    try:
        resp = requests.get(LATEST_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        latest = data.get("version")
        if isinstance(latest, str) and _parse(latest) > _parse(current):
            return latest
    except Exception:
        return None
    return None


__all__ = ["check_latest_version"]
