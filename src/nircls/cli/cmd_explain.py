from __future__ import annotations

import argparse
from pathlib import Path

from ..processing.pipeline import explain_model
from . import common


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("explain")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default=str(common.RESULTS_DIR))
    parser.set_defaults(func=run_explain)


def run_explain(args: argparse.Namespace) -> int:
    explain_model(Path(args.model), Path(args.data), Path(args.output))
    return 0
