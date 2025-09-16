from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from nircls.utils.palettes import PALETTE_MAP, Palette
from nircls.utils.plots import _export, plot_pls_loadings, plot_pls_weights


def _figure() -> go.Figure:
    return go.Figure(data=[go.Scatter(x=[0], y=[0])])


def test_export_interactive_success(tmp_path, monkeypatch):
    fig = _figure()
    monkeypatch.setattr(
        "nircls.utils.plots.importlib.util.find_spec", lambda name: object()
    )

    def fake_write_image(self, file, *args, **kwargs):
        Path(file).write_text("img")

    def fake_write_html(self, file, **kwargs):
        Path(file).write_text("<html></html>")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)
    monkeypatch.setattr(go.Figure, "write_html", fake_write_html)
    monkeypatch.setattr(go.Figure, "to_json", lambda self: "{}")

    out_base = tmp_path / "plot"
    _export(fig, out_base, interactive=True)

    assert (tmp_path / "plot.png").exists()
    assert (tmp_path / "plot.svg").exists()
    assert (tmp_path / "plot.html").exists()
    assert (tmp_path / "plot.json").exists()


def test_export_static_missing_kaleido(tmp_path, monkeypatch, caplog):
    fig = _figure()
    monkeypatch.setattr(
        "nircls.utils.plots.importlib.util.find_spec", lambda name: None
    )

    called = False

    def fail_write_image(self, file, *args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(go.Figure, "write_image", fail_write_image)
    out_base = tmp_path / "plot"

    with caplog.at_level("WARNING"):
        _export(fig, out_base, interactive=False)

    assert called is False
    assert not (tmp_path / "plot.png").exists()
    assert not (tmp_path / "plot.svg").exists()
    assert "Kaleido engine not found" in caplog.text


def test_export_static_write_image_error(tmp_path, monkeypatch, caplog):
    fig = _figure()
    monkeypatch.setattr(
        "nircls.utils.plots.importlib.util.find_spec", lambda name: object()
    )

    def raise_write_image(self, file, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(go.Figure, "write_image", raise_write_image)
    out_base = tmp_path / "plot"

    with caplog.at_level("WARNING"):
        _export(fig, out_base, interactive=False)

    assert "Export failed for" in caplog.text
    assert not (tmp_path / "plot.png").exists()
    assert not (tmp_path / "plot.svg").exists()


def test_export_interactive_html_error(tmp_path, monkeypatch, caplog):
    fig = _figure()
    monkeypatch.setattr(
        "nircls.utils.plots.importlib.util.find_spec", lambda name: object()
    )

    def fake_write_image(self, file, *args, **kwargs):
        Path(file).write_text("img")

    def fail_write_html(self, file, **kwargs):
        raise ValueError("html fail")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)
    monkeypatch.setattr(go.Figure, "write_html", fail_write_html)
    monkeypatch.setattr(go.Figure, "to_json", lambda self: "{}")

    out_base = tmp_path / "plot"
    with caplog.at_level("WARNING"):
        _export(fig, out_base, interactive=True)

    assert (tmp_path / "plot.png").exists()
    assert (tmp_path / "plot.svg").exists()
    assert (tmp_path / "plot.json").exists()
    assert not (tmp_path / "plot.html").exists()
    assert "Export failed for" in caplog.text


def test_plot_pls_loadings_returns_dataframe_and_exports(tmp_path, monkeypatch):
    export_args = {}

    def fake_export(fig, out_base, interactive):
        export_args["fig"] = fig
        export_args["out_base"] = out_base
        export_args["interactive"] = interactive

    monkeypatch.setattr("nircls.utils.plots._export", fake_export)

    loadings = np.array([[0.1, 0.2], [0.3, 0.4]])
    df = plot_pls_loadings(
        loadings,
        feature_names=["f0", "f1"],
        out_base=tmp_path / "loadings",
        interactive=True,
    )

    expected = pd.DataFrame.from_records(
        [
            {"feature": "f0", "component": "Comp 1", "component_index": 1, "value": 0.1},
            {"feature": "f1", "component": "Comp 1", "component_index": 1, "value": 0.3},
            {"feature": "f0", "component": "Comp 2", "component_index": 2, "value": 0.2},
            {"feature": "f1", "component": "Comp 2", "component_index": 2, "value": 0.4},
        ]
    )
    pd.testing.assert_frame_equal(df, expected)

    assert export_args["out_base"] == tmp_path / "loadings"
    assert export_args["interactive"] is True
    fig = export_args["fig"]
    assert fig.layout.title.text == "PLS Loadings"
    assert fig.layout.yaxis.title.text == "Loading"
    assert list(fig.layout.colorway) == PALETTE_MAP[Palette.VIRIDIS]


def test_plot_pls_loadings_no_components_skips_export(tmp_path, monkeypatch):
    called = False

    def fake_export(fig, out_base, interactive):
        nonlocal called
        called = True

    monkeypatch.setattr("nircls.utils.plots._export", fake_export)

    df = plot_pls_loadings(
        np.empty((2, 0)),
        feature_names=["f0", "f1"],
        out_base=tmp_path / "loadings",
    )

    assert df.empty
    assert called is False


def test_plot_pls_weights_requires_2d_array(tmp_path):
    with pytest.raises(ValueError, match="weights must be a 2-D array"):
        plot_pls_weights(np.array([1.0, 2.0, 3.0]), None, tmp_path / "weights")
