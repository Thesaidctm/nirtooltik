from __future__ import annotations

import argparse
import logging

from ..processing.models.factory import get_model_builders, get_model_descriptions
from .common import t

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("list-models", help=t("models_help"))
    parser.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        default="classification",
        help=t("target_type_help"),
    )
    parser.set_defaults(func=run_list_models)


def run_list_models(args: argparse.Namespace) -> int:
    target_type = getattr(args, "target_type", "classification")
    builders = get_model_builders(target_type)
    descs = get_model_descriptions(target_type)
    for name in sorted(builders.keys()):
        logger.info("%s: %s", name, descs.get(name, name))
    return 0
