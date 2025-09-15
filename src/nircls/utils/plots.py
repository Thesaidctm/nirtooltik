"""Utility plotting functions used across the project."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

from .logging import logger
from .palettes import PALETTE_MAP, Palette

_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIW2P4z8DwHwAFgwJ/lEDNkwAAAABJRU5ErkJggg=="
)
_PLACEHOLDER_SVG = (
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1\" height=\"1\"></svg>"
)


def _write_placeholder_image(out: Path) -> None:
    """Create a minimal placeholder image when Plotly export fails."""

    try:
        if out.suffix.lower() == ".png":
            out.write_bytes(_PLACEHOLDER_PNG)
        elif out.suffix.lower() == ".svg":
            out.write_text(_PLACEHOLDER_SVG, encoding="utf-8")
    except Exception:  # pragma: no cover - best effort fallback
        logger.debug("Could not write placeholder image", exc_info=True)


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
    if importlib.util.find_spec("kaleido") is None:
        logger.warning(
            "Kaleido engine not found. Install with 'pip install -U kaleido' to enable static image export."
        )
    else:
        for suffix in [".png", ".svg"]:
            out = out_base.with_suffix(suffix)
            logger.debug("Exporting %s", out)
            try:
                fig.write_image(out)
                logger.debug("Exported %s", out)
            except Exception as exc:  # pragma: no cover - small message
                logger.warning("Export failed for %s: %s", out, exc)
                if "kaleido" in str(exc).lower() and not out.exists():
                    _write_placeholder_image(out)

    if interactive:
        html_out = out_base.with_suffix(".html")
        json_out = out_base.with_suffix(".json")
        try:
            fig.write_html(html_out, include_plotlyjs="cdn")
            logger.debug("Exported %s", html_out)
        except Exception as exc:  # pragma: no cover - small message
            logger.warning("Export failed for %s: %s", html_out, exc)
        try:
            json_out.write_text(fig.to_json())
            logger.debug("Exported %s", json_out)
        except Exception as exc:  # pragma: no cover - small message
            logger.warning("Export failed for %s: %s", json_out, exc)


def plot_confusion(
    cm,
    classes,
    out_base: Path,
    interactive: bool = False,
    *,
    palette: Palette = Palette.BLUES,
) -> None:
    """Plot and export a confusion matrix with raw counts.

    The heatmap displays only the number of samples per cell and a CSV file is
    also generated including an additional column with the per-class accuracy
    percentage.

    Parameters
    ----------
    cm, classes, out_base, interactive:
        See original function.
    palette:
        Name of the color palette applied to the heatmap.
    """

    cm = np.asarray(cm)
    labels = cm.astype(str)
    colorscale = px.colors.make_colorscale(PALETTE_MAP[palette])
    heatmap = go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        text=labels,
        texttemplate="%{text}",
        colorscale=colorscale,
        zmin=0,
        zmax=cm.max(),
        colorbar=dict(title="Número de indivíduos"),
    )
    fig = go.Figure(data=[heatmap])
    fig.update_xaxes(tickangle=45)
    fig.update_layout(title="Confusion Matrix", xaxis_title="Pred", yaxis_title="True")
    _export(fig, out_base, interactive)

    acc = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(cm.shape[0], dtype=float),
        where=cm.sum(axis=1) != 0,
    )
    df = pd.DataFrame(cm, index=classes, columns=classes)
    df["accuracy_percent"] = acc * 100
    df.to_csv(out_base.with_suffix(".csv"))


def plot_roc_ovr(
    y_true,
    y_proba,
    classes,
    out_base: Path,
    interactive: bool = False,
    *,
    palette: Palette = Palette.VIRIDIS,
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
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=str(classes[i])))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_layout(
        xaxis_title="FPR",
        yaxis_title="TPR",
        title="ROC OvR",
        colorway=PALETTE_MAP[palette],
    )
    _export(fig, out_base, interactive)


def plot_reliability(
    y_true,
    y_proba,
    out_base: Path,
    n_bins: int = 10,
    interactive: bool = False,
    *,
    palette: Palette = Palette.VIRIDIS,
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
            line=dict(dash="dash", color="gray"),
        )
    )
    if mids:
        fig.add_trace(go.Scatter(x=mids, y=accs, mode="lines+markers", name="Model"))
    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        title="Reliability Diagram",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        colorway=PALETTE_MAP[palette],
    )
    _export(fig, out_base, interactive)


def plot_pca_scores(
    scores: pd.DataFrame,
    labels: Optional[Sequence],
    out_base: Path,
    interactive: bool = False,
    *,
    palette: Palette = Palette.VIRIDIS,
) -> None:
    """Scatter plot of PCA scores coloured by treatment labels."""

    missing = {"PC1", "PC2"} - set(scores.columns)
    if missing:
        raise ValueError(f"Scores dataframe missing columns: {sorted(missing)}")

    df = scores.copy()
    if labels is not None:
        if isinstance(labels, pd.Series):
            label_series = labels.reindex(df.index)
        else:
            label_series = pd.Series(labels, index=df.index)
        label_series = label_series.fillna("Unlabeled").astype(str)
    else:
        label_series = pd.Series(["All"] * len(df), index=df.index, dtype="object")

    df["_group"] = label_series

    fig = go.Figure()
    colors = PALETTE_MAP[palette]
    for idx, (label, group) in enumerate(df.groupby("_group", sort=False)):
        color = colors[idx % len(colors)] if colors else None
        fig.add_trace(
            go.Scatter(
                x=group["PC1"],
                y=group["PC2"],
                mode="markers",
                name=str(label),
                marker=dict(color=color) if color else None,
                text=group.index.astype(str),
                hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>Sample=%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="PCA Scores",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="Treatment",
        colorway=colors,
    )
    _export(fig, out_base, interactive)
