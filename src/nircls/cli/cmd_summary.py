from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..utils.summarize import summarize_runs
from .common import _prepare_summary_display, t

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("summary", help=t("summary_help"))
    parser.add_argument("--results", required=True, help=t("dashboard_results_help"))
    parser.set_defaults(func=run_summary)


def run_summary(args: argparse.Namespace) -> int:
    df = summarize_runs(Path(args.results))
    if not df.empty:
        ordered_cols = [
            "run",
            "target",
            "model",
            "preprocess_steps",
            "result",
        ]
        df = _prepare_summary_display(df, ordered_cols=ordered_cols)
        with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
            logger.info(df.to_string(index=False))
    return 0
