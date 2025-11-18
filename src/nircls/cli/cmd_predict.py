from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..processing.pipeline import predict_with_model
from ..utils.errors import DataError, ModelError
from . import common
from .common import t


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("predict")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default=str(common.RESULTS_DIR))
    parser.set_defaults(func=run_predict)


def run_predict(args: argparse.Namespace) -> int:
    try:
        predict_with_model(Path(args.model), Path(args.data), Path(args.output))
    except DataError as e:
        print(t("invalid_data", err=e), file=sys.stderr)
    except ModelError as e:
        print(t("invalid_model", err=e), file=sys.stderr)
    return 0
