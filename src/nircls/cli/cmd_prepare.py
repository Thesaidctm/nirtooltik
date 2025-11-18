from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..core.settings import settings
from ..processing.io import export_xy_xlsx, summarize_dataset
from ..processing.pipeline import prepare_inputs
from ..utils.errors import DataError
from ..utils.misc import make_run_id
from . import common
from .common import (
    _print_dataset_summary,
    load_config,
    setup_run_logging,
    t,
)

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prepare-data")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        help=t("target_type_help"),
    )
    parser.set_defaults(func=run_prepare_data)


def run_prepare_data(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, getattr(args, "target_type", None), args.notification_email)
    if cfg is None:
        return 1
    if cfg.data is None:
        logger.error(t("config_invalid_data_section"))
        return 1

    run_id = make_run_id("prepare")
    setup_run_logging(run_id, args.log_level_value, args.log_file)
    out_dir = Path(cfg.data.out_dir or settings.RESULTS_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
        return 1

    summary = summarize_dataset(ds_list)
    _print_dataset_summary(summary)
    if args.run == "one":
        ds_list = ds_list[:1]
    for ds in ds_list:
        out_file = out_dir / f"prepared_{ds['target']}.xlsx"
        export_xy_xlsx(
            out_file,
            ds["X"],
            ds["Y"],
            cfg.data.x_sheet or "X",
            cfg.data.y_sheet or "Y",
        )
    return 0
