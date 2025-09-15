from __future__ import annotations

from enum import Enum
from typing import Dict, List

import plotly.express as px


class Palette(str, Enum):
    """Supported color palettes for plots."""

    BLUES = "blues"
    VIRIDIS = "viridis"
    COLORBLIND = "colorblind"
    BLUES = "blues"


_BLUES: List[str] = list(px.colors.sequential.Blues)
if _BLUES:
    _BLUES[0] = "#ffffff"


_blues = list(px.colors.sequential.Blues)
if _blues:
    _blues[0] = "#ffffff"


PALETTE_MAP: Dict[Palette, List[str]] = {
    Palette.BLUES: _blues,
    Palette.VIRIDIS: px.colors.sequential.Viridis,
    Palette.COLORBLIND: px.colors.qualitative.Safe,
    Palette.BLUES: _BLUES,
}

__all__ = ["Palette", "PALETTE_MAP"]
