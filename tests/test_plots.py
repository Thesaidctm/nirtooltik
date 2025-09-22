import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from nircls.utils.palettes import PALETTE_MAP, Palette
from nircls.utils.plots import _cycle_visible_colors


def _is_almost_white(color: str) -> bool:
    hex_value = color.strip().lower()
    if not hex_value.startswith("#"):
        return False
    hex_value = hex_value[1:]
    return hex_value in {"fff", "ffffff"}


def test_cycle_visible_colors_skips_leading_white():
    colors = _cycle_visible_colors(Palette.BLUES)
    assert colors, "Expected at least one color in the palette"
    assert not _is_almost_white(colors[0])


def test_cycle_visible_colors_does_not_mutate_palette_map():
    original_first = PALETTE_MAP[Palette.BLUES][0]
    _cycle_visible_colors(Palette.BLUES)
    assert PALETTE_MAP[Palette.BLUES][0] == original_first
