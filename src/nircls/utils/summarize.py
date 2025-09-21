"""Utilities for summarizing experiment results."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _extract_strategy_and_preprocess(run_path: str) -> Tuple[str, str]:
    """Derive strategy and preprocess labels from a run path."""

    if not run_path:
        return "", ""
    parts = Path(run_path).parts
    if not parts:
        return "", ""
    first_part = parts[0]
    if "__" in first_part:
        strategy, preprocess = first_part.split("__", 1)
    else:
        strategy, preprocess = first_part, ""
    return strategy, preprocess


def summarize_runs(base_dir: Path | str) -> pd.DataFrame:
    """Aggregate metrics from multiple runs.

    Parameters
    ----------
    base_dir:
        Directory containing run subfolders with ``metrics.json`` files. Each
        strategy and preprocessing combination has its own subdirectory (for
        example, ``kfold_5__snv``).

    Returns
    -------
    pandas.DataFrame
        Concatenated metrics with an additional ``run`` column indicating the
        strategy and preprocessing step folder. Returns an empty DataFrame if
        no metrics are found.
    """

    base_path = Path(base_dir)
    rows: List[pd.DataFrame] = []
    if not base_path.exists():
        return pd.DataFrame()
    for run_dir in sorted(base_path.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_files = list(run_dir.rglob("metrics.json"))
        if not metrics_files:
            continue
        for mfile in metrics_files:
            try:
                df = pd.read_json(mfile)
            except ValueError:
                continue
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "run", str(mfile.parent.relative_to(base_path)))
            rows.append(df)
    if not rows:
        df = pd.DataFrame()
    else:
        df = pd.concat(rows, ignore_index=True)

    if df.empty:
        return pd.DataFrame(columns=["run", "strategy", "preprocess"])

    df = df.copy()
    strategies: List[str] = []
    preprocesses: List[str] = []
    for run_path in df["run"].astype(str):
        strategy, preprocess = _extract_strategy_and_preprocess(run_path)
        strategies.append(strategy)
        preprocesses.append(preprocess or "")
    df.insert(1, "strategy", strategies)
    df.insert(2, "preprocess", preprocesses)
    return df
