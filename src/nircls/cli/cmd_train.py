from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..processing.io import summarize_dataset
from ..processing.pipeline import prepare_inputs, train_and_save
from ..utils.errors import DataError, ModelError
from ..utils.misc import make_run_id
from . import common
from .common import (
    _print_astm_report,
    _print_confusion_paths,
    _print_dataset_summary,
    _print_metrics_table,
    load_config,
    setup_run_logging,
    t,
)

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        help=t("target_type_help"),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=t("interactive_help"),
    )
    parser.add_argument(
        "--astm",
        action="store_true",
        help=t("astm_help"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=t("resume_help"),
    )
    parser.add_argument("--run-id", help=t("run_id_help"))
    parser.set_defaults(func=run_train)


def run_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, getattr(args, "target_type", None), args.notification_email)
    if cfg is None:
        return 1
    if cfg.data is None or not cfg.models:
        logger.error(t("config_invalid_data_models_sections"))
        return 1

    cfg.astm = args.astm
    if args.resume and not args.run_id:
        logger.error(t("resume_requires_run_id"))
        return 1

    run_id = args.run_id or make_run_id("train")
    setup_run_logging(run_id, args.log_level_value, args.log_file)
    out_dir = Path(cfg.data.out_dir or common.RESULTS_DIR) / run_id
    if args.resume and not out_dir.exists():
        logger.error(t("resume_dir_missing", path=out_dir))
        return 1
    if args.resume:
        logger.info(t("resume_from", path=out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
        return 1

    summary = summarize_dataset(ds_list)
    _print_dataset_summary(summary)
    try:
        metrics_path, figs_dir = train_and_save(
            cfg.dict(exclude_defaults=True, exclude_none=True),
            out_dir,
            interactive=args.interactive,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            mlflow_uri=args.mlflow_uri,
            gs_verbose=args.cv_verbose,
            show_progress=not args.no_progress,
            log_memory=args.log_memory,
            resume=args.resume,
        )
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
        return 1
    except ModelError as e:
        logger.error(t("invalid_model", err=e), exc_info=True)
        return 1

    if metrics_path:
        _print_metrics_table(metrics_path)
    if figs_dir:
        _print_confusion_paths(figs_dir)
    report_path = out_dir / "report.md"
    if report_path.exists():
        logger.info(t("full_report", path=report_path))
    if args.astm:
        _print_astm_report(out_dir / "astm_classification.json")
    logger.info(t("results_in", path=out_dir))
    return 0
