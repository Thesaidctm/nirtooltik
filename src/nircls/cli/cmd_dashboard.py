from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import t


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("dashboard")
    parser.add_argument("--results", required=True, help=t("dashboard_results_help"))
    parser.set_defaults(func=run_dashboard_cmd)


def run_dashboard_cmd(args: argparse.Namespace) -> int:
    try:
        from ..dashboard import run_dashboard
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time dependency guard
        missing = getattr(exc, "name", None) or "streamlit"
        print(t("dashboard_dep_missing", package=missing), file=sys.stderr)
        return 1
    try:
        run_dashboard(Path(args.results))
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        missing = getattr(exc, "name", None) or str(exc)
        print(t("dashboard_dep_missing", package=missing), file=sys.stderr)
        return 1
    return 0
