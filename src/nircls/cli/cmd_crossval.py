from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..core.settings import settings
from ..processing.io import summarize_dataset
from ..processing.pipeline import prepare_inputs, run_multiple_eval
from ..utils.errors import DataError, ModelError
from ..utils.misc import make_run_id
from . import common
from .common import (
    _describe_preprocess_steps,
    _load_preprocess_steps,
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
    parser = subparsers.add_parser("crossval")
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
    parser.set_defaults(func=run_crossval)


def run_crossval(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, getattr(args, "target_type", None), args.notification_email)
    if cfg is None:
        return 1
    if cfg.data is None or not cfg.models or cfg.cv is None:
        logger.error(t("config_invalid_data_models_cv_sections"))
        return 1

    try:
        ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
        return 1

    summary = summarize_dataset(ds_list)
    _print_dataset_summary(summary)
    if args.run == "one" and ds_list:
        cfg.data.target_cols = ds_list[0]["target"]
    if args.resume and not args.run_id:
        logger.error(t("resume_requires_run_id"))
        return 1

    run_id = args.run_id or make_run_id("cv")
    setup_run_logging(run_id, args.log_level_value, args.log_file)
    out_dir = Path(cfg.data.out_dir or settings.RESULTS_DIR) / run_id
    if args.resume and not out_dir.exists():
        logger.error(t("resume_dir_missing", path=out_dir))
        return 1
    if args.resume:
        logger.info(t("resume_from", path=out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg.astm = args.astm
    try:
        outputs, summary_dir = run_multiple_eval(
            cfg.dict(exclude_defaults=True, exclude_none=True),
            out_dir,
            interactive=args.interactive,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            mlflow_uri=args.mlflow_uri,
            log_memory=args.log_memory,
            resume=args.resume,
        )
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
        return 1
    except ModelError as e:
        logger.error(t("invalid_model", err=e), exc_info=True)
        return 1

    for metrics_path, figs_dir in outputs:
        step_descs = _describe_preprocess_steps(_load_preprocess_steps(metrics_path))
        if step_descs:
            header = f"== {metrics_path.parent.name} ({', '.join(step_descs)}) =="
        else:
            header = f"== {metrics_path.parent.name} =="
        logger.info(header)
        _print_metrics_table(metrics_path)
        _print_confusion_paths(figs_dir)
        rpt = metrics_path.parent / "report.md"
        if rpt.exists():
            logger.info(t("full_report", path=rpt))
        if args.astm:
            _print_astm_report(metrics_path.parent / "astm_classification.json")
    logger.info(t("results_in", path=out_dir))
    logger.info("Summary saved in %s", summary_dir)
    return 0
