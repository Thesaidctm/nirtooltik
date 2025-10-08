from __future__ import annotations

from enum import Enum
from typing import Dict, List

try:  # pragma: no cover - import is trivial but environment-dependent
    import plotly.express as px
except ModuleNotFoundError:  # pragma: no cover - executed when optional dep missing
    px = None


class Palette(str, Enum):
    """Supported color palettes for plots."""

    BLUES = "blues"
    VIRIDIS = "viridis"
    COLORBLIND = "colorblind"


if px is not None:  # pragma: no branch - tiny conditional
    _BLUES: List[str] = list(px.colors.sequential.Blues)
    if _BLUES:
        _BLUES[0] = "#ffffff"
    _PALETTE_MAP: Dict[Palette, List[str]] = {
        Palette.BLUES: _BLUES,
        Palette.VIRIDIS: px.colors.sequential.Viridis,
        Palette.COLORBLIND: px.colors.qualitative.Safe,
    }
else:
    # Fallback palettes extracted from Plotly defaults. Keeping the first colour as
    # white mirrors the behaviour applied when Plotly is available. This allows the
    # configuration schema to validate options even when optional visualisation
    # dependencies are not installed, e.g. when running purely from the CLI.
    _PALETTE_MAP = {
        Palette.BLUES: [
            "#ffffff",
            "#f7fbff",
            "#deebf7",
            "#c6dbef",
            "#9ecae1",
            "#6baed6",
            "#4292c6",
            "#2171b5",
            "#08519c",
            "#08306b",
        ],
        Palette.VIRIDIS: [
            "#440154",
            "#482878",
            "#3e4989",
            "#31688e",
            "#26828e",
            "#1f9e89",
            "#35b779",
            "#6ece58",
            "#b5de2b",
            "#fde725",
        ],
        Palette.COLORBLIND: [
            "#0072B2",
            "#E69F00",
            "#F0E442",
            "#009E73",
            "#56B4E9",
            "#D55E00",
            "#CC79A7",
            "#999999",
        ],
    }


PALETTE_MAP: Dict[Palette, List[str]] = _PALETTE_MAP

__all__ = ["Palette", "PALETTE_MAP"]
