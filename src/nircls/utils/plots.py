"""Utility plotting functions used across the project."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid Tk GUI requirement
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion(cm, classes, out_base: Path, normalize: bool = False):
    if normalize:
        cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    else:
        cmn = cm

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=" .2f" if normalize else "d",
        cmap="viridis",
        ax=ax,
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"size": 12},
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Taxa de acerto normalizada" if normalize else "Número de indivíduos"},
    )

    ax.set_xlabel("Pred", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
    ax.set_title(
        "Confusion Matrix (normalized)" if normalize else "Confusion Matrix",
        fontsize=16,
        fontweight="bold",
    )

    for text in ax.texts:
        text.set_fontsize(12)
        x, y = text.get_position()
        if int(x - 0.5) == int(y - 0.5):
            text.set_fontweight("bold")

    ax.set_aspect("equal")
    plt.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def plot_roc_ovr(y_true, y_proba, classes, out_base: Path):
    """Plot one-vs-rest ROC curves for multi-class or binary problems."""

    plt.figure(figsize=(7, 6))
    n_classes = len(classes)
    if y_proba.shape[1] == 2:
        # binary case: use positive class probability
        y_bin = (y_true == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, 1])
        plt.plot(fpr, tpr, label=str(classes[1]))
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, label=f"{classes[i]}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC OvR")
    plt.legend()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"))
    plt.savefig(out_base.with_suffix(".svg"))
    plt.close()


def plot_reliability(y_true, y_proba, out_base: Path, n_bins: int = 10):
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
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], "--")
    if mids:
        plt.plot(mids, accs, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".png"))
    plt.savefig(out_base.with_suffix(".svg"))
    plt.close()

