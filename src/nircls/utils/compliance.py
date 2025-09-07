"""Compliance checks for ASTM classification guidelines."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def check_astm_classification(summary: pd.DataFrame, metrics: Dict) -> Dict[str, bool]:
    """Evaluate dataset and metrics against basic ASTM classification rules.

    Parameters
    ----------
    summary:
        Output of :func:`summarize_dataset` with counts per class.
    metrics:
        Dictionary containing experiment metadata. Only the key
        ``external_validation`` is inspected to verify if a holdout or
        leave-one-out validation was performed.

    Returns
    -------
    dict
        Mapping with boolean flags for each criterion: ``min_samples``,
        ``min_per_class`` and ``external_validation``.
    """

    total = int(summary["count"].sum()) if not summary.empty else 0
    min_samples = total >= 30

    min_class = int(summary["count"].min()) if not summary.empty else 0
    min_per_class = min_class >= 10

    external_validation = bool(metrics.get("external_validation", False))

    return {
        "min_samples": min_samples,
        "min_per_class": min_per_class,
        "external_validation": external_validation,
    }
