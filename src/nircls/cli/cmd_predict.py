from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..core.settings import settings
from ..processing.pipeline import predict_with_model
from ..utils.errors import DataError, ModelError
from . import common
from .common import t

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("predict")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default=str(settings.RESULTS_DIR))
    parser.set_defaults(func=run_predict)


def run_predict(args: argparse.Namespace) -> int:
    try:
        predict_with_model(Path(args.model), Path(args.data), Path(args.output))
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
    except ModelError as e:
        logger.error(t("invalid_model", err=e), exc_info=True)
    return 0
