from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from ..config_schema import ConfigModel, ValidationError
from ..processing.io import export_xy_xlsx
from ..processing.pipeline import prepare_inputs, run_multiple_eval, train_and_save
from ..utils.errors import DataError, ModelError
from ..utils.misc import ensure_dirs, make_run_id
from . import common
from .common import (
    _print_confusion_paths,
    _print_metrics_table,
    setup_run_logging,
    t,
)

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("demo")
    parser.set_defaults(func=run_demo)


def run_demo(args: argparse.Namespace) -> int:
    demo_cfg = Path(__file__).resolve().parents[3] / "configs" / "demo.yaml"
    with open(demo_cfg) as f:
        raw = yaml.safe_load(f)
    raw["data"]["input_path"] = str(
        (demo_cfg.parent.parent / raw["data"]["input_path"]).resolve()
    )
    if args.notification_email:
        raw["notification_email"] = args.notification_email
    try:
        cfg = ConfigModel(**raw)
    except ValidationError as e:  # pragma: no cover - user feedback
        logger.error(t("demo_config_invalid", err=e), exc_info=True)
        return 1

    run_id = make_run_id("demo")
    setup_run_logging(run_id, args.log_level_value, args.log_file)
    base_dir = Path(cfg.data.out_dir or common.RESULTS_DIR) / run_id
    ensure_dirs(base_dir)

    ds_list = []
    try:
        ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
    if ds_list:
        first = ds_list[0]
        prep_file = base_dir / f"prepared_{first['target']}.xlsx"
        export_xy_xlsx(
            prep_file,
            first["X"],
            first["Y"],
            cfg.data.x_sheet or "X",
            cfg.data.y_sheet or "Y",
        )
        logger.info(t("data_prepared_in", file=prep_file))
    train_dir = base_dir / "train"
    ensure_dirs(train_dir)
    try:
        metrics_path, figs_dir = train_and_save(
            cfg.dict(exclude_defaults=True, exclude_none=True),
            train_dir,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            mlflow_uri=args.mlflow_uri,
            gs_verbose=args.cv_verbose,
            show_progress=not args.no_progress,
            log_memory=args.log_memory,
        )
        if metrics_path:
            _print_metrics_table(metrics_path)
        if figs_dir:
            _print_confusion_paths(figs_dir)
        logger.info(t("model_trained_in", path=train_dir))
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
    except ModelError as e:
        logger.error(t("invalid_model", err=e), exc_info=True)
    cv_dir = base_dir / "cv"
    ensure_dirs(cv_dir)
    try:
        outputs, summary_dir = run_multiple_eval(
            cfg.dict(exclude_defaults=True, exclude_none=True),
            cv_dir,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            mlflow_uri=args.mlflow_uri,
            log_memory=args.log_memory,
        )
        for metrics_path, figs_dir in outputs:
            _print_metrics_table(metrics_path)
            _print_confusion_paths(figs_dir)
        logger.info(t("cv_results_in", path=cv_dir))
        logger.info("Summary saved in %s", summary_dir)
    except DataError as e:
        logger.error(t("invalid_data", err=e), exc_info=True)
    except ModelError as e:
        logger.error(t("invalid_model", err=e), exc_info=True)
    return 0
