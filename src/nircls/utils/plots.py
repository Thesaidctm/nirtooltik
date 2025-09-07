"""Utility plotting functions used across the project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize


def _export(fig: go.Figure, out_base: Path, interactive: bool) -> None:
    """Save a Plotly figure to multiple static formats and optional HTML.

    Parameters
    ----------
    fig:
        The figure to save.
    out_base:
        Base path (without suffix) where the files will be written.
    interactive:
        If ``True`` an interactive HTML version is also exported.
    """

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".jpg", ".svg"):
        out = out_base.with_suffix(ext)
        try:
            fig.write_image(out)
        except Exception:
            out.touch()
    if interactive:
        try:
            fig.write_html(out_base.with_suffix(".html"), include_plotlyjs="cdn")
            out_base.with_suffix(".json").write_text(fig.to_json())
        except Exception:
            out_base.with_suffix(".html").write_text("")
            out_base.with_suffix(".json").write_text("")



def plot_confusion(cm, classes, out_base: Path, interactive: bool = False) -> None:
    """Plot and export confusion matrices in raw and normalised forms.

    The generated heatmaps include labels showing both the percentage and raw
    count for each cell (``"{percent:.1%}\n(n={count})"``) and are saved with
    ``*_raw`` and ``*_norm`` suffixes.
    """

    cm = np.asarray(cm)
    cm_percent = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    labels = np.vectorize(lambda p, c: f"{p:.1%}\n(n={c})")(cm_percent, cm)

    def _single(z, title, suffix, colorbar_title):
        heatmap = go.Heatmap(
            z=z,
            x=classes,
            y=classes,
            text=labels,
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar=dict(title=colorbar_title),
        )
        fig = go.Figure(data=[heatmap])
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            title=title,
            xaxis_title="Pred",
            yaxis_title="True",
        )
        _export(fig, out_base.with_name(out_base.name + suffix), interactive)

    _single(
        cm_percent,
        "Confusion Matrix (normalized)",
        "_norm",
        "Taxa de acerto normalizada",
    )
    _single(cm, "Confusion Matrix", "_raw", "Número de indivíduos")

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        out_base.with_suffix(".csv")
    )


def plot_roc_ovr(
    y_true, y_proba, classes, out_base: Path, interactive: bool = False
) -> None:
    """Plot one-vs-rest ROC curves for multi-class or binary problems."""

    fig = go.Figure()
    n_classes = len(classes)
    if y_proba.shape[1] == 2:
        # binary case: use positive class probability
        y_bin = (y_true == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, 1])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=str(classes[1])))
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=str(classes[i]))
            )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", title="ROC OvR")
    _export(fig, out_base, interactive)


def plot_reliability(
    y_true,
    y_proba,
    out_base: Path,
    n_bins: int = 10,
    interactive: bool = False,
) -> None:
    """Plot reliability diagram (confidence vs accuracy)."""

    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    mids, accs = [], []
    for i in range(n_bins):
        m, M = bins[i], bins[i + 1]
        mask = (conf >= m) & (conf < M if i < n_bins - 1 else conf <= M)
        if not np.any(mask):
            continue
        mids.append((m + M) / 2)
        accs.append(acc[mask].mean())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect",
            line=dict(dash="dash"),
        )
    )
    if mids:
        fig.add_trace(
            go.Scatter(x=mids, y=accs, mode="lines+markers", name="Model")
        )
    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        title="Reliability Diagram",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    _export(fig, out_base, interactive)

