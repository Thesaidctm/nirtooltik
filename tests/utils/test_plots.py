from pathlib import Path

import plotly.graph_objects as go

from nircls.utils.plots import _export


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
    assert (tmp_path / "plot.json").exists()
    assert not (tmp_path / "plot.html").exists()
    assert "Export failed for" in caplog.text
