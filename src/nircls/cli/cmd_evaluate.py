from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..processing.pipeline import evaluate_saved_model
from ..utils.errors import DataError, ModelError
from .common import t

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.set_defaults(func=run_evaluate)


def run_evaluate(args: argparse.Namespace) -> int:
    try:
        evaluate_saved_model(Path(args.model), Path(args.data))
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
    except ModelError as e:
        logger.error(t("invalid_model", err=e), exc_info=True)
    return 0
