from __future__ import annotations

from enum import Enum
from typing import Dict, List

import plotly.express as px


class Palette(str, Enum):
    """Supported color palettes for plots."""

    VIRIDIS = "viridis"
    COLORBLIND = "colorblind"


PALETTE_MAP: Dict[Palette, List[str]] = {
    Palette.VIRIDIS: px.colors.sequential.Viridis,
    Palette.COLORBLIND: px.colors.qualitative.Safe,
}

__all__ = ["Palette", "PALETTE_MAP"]
