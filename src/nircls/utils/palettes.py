from __future__ import annotations

from enum import Enum
from typing import Dict, List

import plotly.express as px


class Palette(str, Enum):
    """Supported color palettes for plots."""

    BLUES = "blues"
    VIRIDIS = "viridis"
    COLORBLIND = "colorblind"


_BLUES: List[str] = list(px.colors.sequential.Blues)
if _BLUES:
    _BLUES[0] = "#ffffff"


PALETTE_MAP: Dict[Palette, List[str]] = {
    Palette.BLUES: _BLUES,
    Palette.VIRIDIS: px.colors.sequential.Viridis,
    Palette.COLORBLIND: px.colors.qualitative.Safe,
}

__all__ = ["Palette", "PALETTE_MAP"]
