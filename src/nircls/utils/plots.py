"""Utility plotting functions used across the project."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from typing import Iterable, Optional, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

try:  # pragma: no cover - import depends on optional dependency
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover - executed when Plotly missing
    px = None
    go = None

from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

from .logging import logger
from .palettes import PALETTE_MAP, Palette

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import plotly.graph_objects as go


_PLOTLY_MISSING_MSG = (
    "Plotly is required for visualisations. Install with `pip install "
    "\"nircls[viz]\"` or `pip install \"nircls[full]\"`."
)


def _require_plotly() -> None:
    if px is None or go is None:
        err = ModuleNotFoundError(_PLOTLY_MISSING_MSG)
        err.name = "plotly"  # type: ignore[attr-defined]
        raise err


def _cycle_visible_colors(palette: Palette) -> list[str]:
    """Return palette colors rotated so the first entry is not almost white."""

    colors = list(PALETTE_MAP[palette])
    if not colors:
        return colors

    def is_almost_white(color: str) -> bool:
        hex_value = color.strip().lower()
        if not hex_value.startswith("#"):
            return False
        hex_value = hex_value[1:]
        return hex_value in {"fff", "ffffff"}

    for idx, color in enumerate(colors):
        if not is_almost_white(color):
            if idx == 0:
                return colors
            return colors[idx:] + colors[:idx]
    return colors

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


def _export(fig: "go.Figure", out_base: Path, interactive: bool) -> None:
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

    _require_plotly()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("kaleido") is None:
        logger.warning(
            "Kaleido engine not found. Install with 'pip install -U kaleido' to enable static image export."
        )
        for suffix in [".png", ".svg"]:
            out = out_base.with_suffix(suffix)
            if not out.exists():
                _write_placeholder_image(out)
    else:
        for suffix in [".png", ".svg"]:
            out = out_base.with_suffix(suffix)
            logger.debug("Exporting %s", out)
            try:
                fig.write_image(out)
                logger.debug("Exported %s", out)
            except Exception as exc:  # pragma: no cover - small message
                logger.warning("Export failed for %s: %s", out, exc)
                if not out.exists():
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


def _coerce_feature_names(
    feature_names: Sequence[object] | pd.Index | None, n_features: int
) -> list[object]:
    """Return a list of feature labels with length ``n_features``."""

    if feature_names is None:
        return list(range(n_features))
    if isinstance(feature_names, pd.Index):
        labels = feature_names.tolist()
    else:
        labels = list(feature_names)
    if len(labels) != n_features:
        raise ValueError(
            "Feature name list length does not match number of features"
        )
    coerced: list[object] = []
    for label in labels:
        if isinstance(label, np.generic):
            coerced.append(label.item())
        else:
            coerced.append(label)
    return coerced


def _pls_component_records(
    values: np.ndarray,
    feature_names: Iterable[object],
) -> list[dict[str, object]]:
    """Build long-form records for PLS component values."""

    records: list[dict[str, object]] = []
    component_labels = [f"Comp {idx + 1}" for idx in range(values.shape[1])]
    for comp_idx, label in enumerate(component_labels, start=1):
        column = values[:, comp_idx - 1]
        for feat, val in zip(feature_names, column, strict=False):
            records.append(
                {
                    "feature": feat,
                    "component": label,
                    "component_index": comp_idx,
                    "value": float(val),
                }
            )
    return records


def _plot_pls_components(
    values: np.ndarray,
    feature_names: Sequence[object] | pd.Index | None,
    out_base: Path,
    interactive: bool,
    title: str,
    yaxis: str,
    *,
    palette: Palette,
) -> pd.DataFrame:
    """Shared implementation for plotting PLS-related component matrices."""

    _require_plotly()
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2-D array")
    n_features, n_components = values.shape
    if n_components == 0:
        return pd.DataFrame(columns=["feature", "component", "component_index", "value"])

    features = _coerce_feature_names(feature_names, n_features)
    colors = _cycle_visible_colors(palette)
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Feature",
        yaxis_title=yaxis,
        colorway=colors,
    )
    component_labels = [f"Comp {i + 1}" for i in range(n_components)]
    for idx, label in enumerate(component_labels):
        fig.add_trace(
            go.Scatter(
                x=features,
                y=values[:, idx],
                mode="lines",
                name=label,
            )
        )
    _export(fig, out_base, interactive)

    records = _pls_component_records(values, features)
    return pd.DataFrame.from_records(records)


def plot_pls_loadings(
    loadings: np.ndarray,
    feature_names: Sequence[object] | pd.Index | None,
    out_base: Path,
    interactive: bool = False,
    *,
    palette: Palette = Palette.VIRIDIS,
) -> pd.DataFrame:
    """Plot PLS loadings and return the underlying values."""

    try:
        return _plot_pls_components(
            loadings,
            feature_names,
            out_base,
            interactive,
            "PLS Loadings",
            "Loading",
            palette=palette,
        )
    except ValueError as exc:  # pragma: no cover - defensive consistency
        if str(exc) == "values must be a 2-D array":
            raise ValueError("loadings must be a 2-D array") from exc
        raise


def plot_pls_y_loadings(
    values: np.ndarray,
    feature_names: Sequence[object] | pd.Index | None,
    out_base: Path,
    interactive: bool = False,
    *,
    palette: Palette = Palette.VIRIDIS,
) -> pd.DataFrame:
    """Plot PLS Y loadings and return the underlying values."""

    return _plot_pls_components(
        values,
        feature_names,
        out_base,
        interactive,
        "PLS Y Loadings",
        "Loading",
        palette=palette,
    )


def plot_pls_weights(
    weights: np.ndarray,
    feature_names: Sequence[object] | pd.Index | None,
    out_base: Path,
    interactive: bool = False,
    *,
    palette: Palette = Palette.VIRIDIS,
) -> pd.DataFrame:
    """Plot PLS weights for each component and return numeric values."""

    try:
        return _plot_pls_components(
            weights,
            feature_names,
            out_base,
            interactive,
            "PLS Weights",
            "Weight",
            palette=palette,
        )
    except ValueError as exc:  # pragma: no cover - defensive consistency
        if str(exc) == "values must be a 2-D array":
            raise ValueError("weights must be a 2-D array") from exc
        raise


def plot_vip(
    vip_scores: np.ndarray,
    feature_names: Sequence[object] | pd.Index | None,
    out_base: Path,
    interactive: bool = False,
    *,
    palette: Palette = Palette.VIRIDIS,
) -> pd.DataFrame:
    """Plot Variable Importance in Projection (VIP) scores."""

    _require_plotly()
    vip_scores = np.asarray(vip_scores, dtype=float).ravel()
    n_features = vip_scores.size
    if n_features == 0:
        return pd.DataFrame(columns=["feature", "vip"])

    features = _coerce_feature_names(feature_names, n_features)
    colors = _cycle_visible_colors(palette)
    color = colors[0] if colors else None
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=features,
            y=vip_scores,
            marker=dict(color=color) if color else None,
            name="VIP",
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", name="VIP = 1")
    fig.update_layout(
        title="Variable Importance in Projection",
        xaxis_title="Feature",
        yaxis_title="VIP Score",
        showlegend=False,
    )
    _export(fig, out_base, interactive)

    df = pd.DataFrame({"feature": features, "vip": vip_scores.astype(float)})
    return df


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

    _require_plotly()
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

    _require_plotly()
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

    _require_plotly()
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

    _require_plotly()
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
