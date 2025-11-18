from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, cast

import yaml  # type: ignore[import-untyped]

from ..core.settings import settings
from ..processing.io import summarize_dataset
from ..processing.models.factory import get_model_builders
from ..processing.pipeline import prepare_inputs, run_multiple_eval, train_and_save
from ..utils.errors import DataError, ModelError
from ..utils.misc import ensure_dirs, now_ts
from . import common
from .common import (
    PREPROCESS_DESCRIPTIONS,
    _deep_update,
    _describe_preprocess_steps,
    _load_preprocess_steps,
    _print_confusion_paths,
    _print_dataset_summary,
    _print_metrics_table,
    _sanitize_output_name,
    _write_run_state,
    _load_run_state,
    setup_run_logging,
    t,
)

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run-all", help=t("run_all_help"))
    parser.add_argument("--config", help=t("run_all_config_help"))
    parser.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        help=t("target_type_help"),
    )
    parser.add_argument(
        "--all-preprocess",
        action="store_true",
        help=t("run_all_preprocess_help"),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help=t("run_all_models_help"),
    )
    parser.add_argument("--output-name", help=t("run_all_output_name_help"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help=t("resume_help"),
    )
    parser.add_argument("--run-id", help=t("run_id_help"))
    parser.set_defaults(func=run_run_all)


def run_run_all(args: argparse.Namespace) -> int:
    builders = get_model_builders()
    default_cfg: Dict[str, Any] = {
        "data": {
            "input_path": str(settings.RAW_DIR),
            "out_dir": str(settings.RESULTS_DIR),
        },
        "models": [{"name": "PLSDA"}],
        "cv": {"strategies": [{"type": "kfold", "n_splits": 5}]},
    }
    if args.config:
        try:
            with open(args.config) as f:
                user_cfg = cast(Dict[str, Any], yaml.safe_load(f) or {})
        except FileNotFoundError:
            logger.error(t("config_not_found"))
            return 1
        _deep_update(default_cfg, user_cfg)
    if args.target_type:
        default_cfg.setdefault("data", {})["target_type"] = args.target_type
    if args.all_preprocess:
        default_cfg["preprocess"] = {"steps": list(PREPROCESS_DESCRIPTIONS)}
    if args.all_models:
        models_cfg: list[dict[str, Any]] = []
        for name, builder in builders.items():
            try:
                builder({})
            except ImportError as e:
                logger.error(t("model_dep_missing", name=name, err=e), exc_info=True)
                continue
            models_cfg.append({"name": name})
        default_cfg["models"] = models_cfg
    if args.notification_email:
        default_cfg["notification_email"] = args.notification_email
    if args.resume and not args.run_id:
        logger.error(t("resume_requires_run_id"))
        return 1

    run_id = args.run_id or now_ts()
    setup_run_logging(run_id, args.log_level_value, args.log_file)
    data_cfg = cast(Dict[str, Any], default_cfg["data"])
    base_out_dir = Path(data_cfg.get("out_dir", settings.RESULTS_DIR))
    base_dir = base_out_dir / run_id
    rename_target: Optional[Path] = None
    desired_name = None
    if args.output_name:
        desired_name = _sanitize_output_name(args.output_name)
        if not desired_name:
            logger.error(t("run_all_invalid_output_name"))
            return 1
        rename_target = base_out_dir / desired_name
        if rename_target.exists() and rename_target != base_dir:
            logger.error(t("run_all_output_exists", path=rename_target))
            return 1
    if args.resume and not base_dir.exists():
        logger.error(t("resume_dir_missing", path=base_dir))
        return 1
    if args.resume:
        logger.info(t("resume_from", path=base_dir))
    ensure_dirs(base_dir)

    state = _load_run_state(base_dir) or {
        "run_id": run_id,
        "train_completed": False,
        "cv_completed": False,
        "status": "initialized",
    }
    state["run_id"] = run_id
    state["status"] = "preparing_data"
    state.pop("last_error", None)
    _write_run_state(base_dir, state)

    data_cfg["out_dir"] = str(base_dir)
    try:
        ds_list = prepare_inputs(data_cfg, use_cache=bool(default_cfg.get("use_cache", True)))
    except DataError as e:
        state["status"] = "data_failed"
        state["last_error"] = str(e)
        _write_run_state(base_dir, state)
        logger.error(t("invalid_data", err=e), exc_info=True)
        return 1

    summary = summarize_dataset(ds_list)
    _print_dataset_summary(summary)
    if args.run == "one" and ds_list:
        data_cfg["target_cols"] = ds_list[0]["target"]
    train_dir = base_dir / "train"
    cv_dir = base_dir / "cv"
    ensure_dirs(train_dir, cv_dir)

    state["status"] = "training"
    _write_run_state(base_dir, state)
    try:
        train_metrics, train_figs = train_and_save(
            default_cfg,
            train_dir,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            mlflow_uri=args.mlflow_uri,
            gs_verbose=args.cv_verbose,
            show_progress=not args.no_progress,
            log_memory=args.log_memory,
            resume=args.resume,
        )
    except (DataError, ModelError, ValueError) as e:
        state["status"] = "train_failed"
        state["last_error"] = str(e)
        _write_run_state(base_dir, state)
        logger.error(t("run_all_train_failed", err=e), exc_info=True)
        logger.warning(t("run_all_resume_hint", run_id=run_id))
        return 1

    state["train_completed"] = True
    state["status"] = "train_completed"
    state.pop("last_error", None)
    _write_run_state(base_dir, state)
    model_path = train_dir / "model.joblib"
    report_path = train_dir / "report.md"

    state["status"] = "running_cv"
    _write_run_state(base_dir, state)
    try:
        outputs, summary_dir = run_multiple_eval(
            default_cfg,
            cv_dir,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            mlflow_uri=args.mlflow_uri,
            log_memory=args.log_memory,
            resume=args.resume,
        )
    except (DataError, ModelError, ValueError) as e:
        state["status"] = "cv_failed"
        state["last_error"] = str(e)
        _write_run_state(base_dir, state)
        logger.error(t("run_all_cv_failed", err=e), exc_info=True)
        logger.warning(t("run_all_resume_hint", run_id=run_id))
        return 1

    state["cv_completed"] = True
    state["status"] = "completed"
    state.pop("last_error", None)
    _write_run_state(base_dir, state)

    for cv_metrics, cv_figs in outputs:
        step_descs = _describe_preprocess_steps(_load_preprocess_steps(cv_metrics))
        if step_descs:
            header = f"== {cv_metrics.parent.name} ({', '.join(step_descs)}) =="
        else:
            header = f"== {cv_metrics.parent.name} =="
        logger.info(header)
        _print_metrics_table(cv_metrics)
        _print_confusion_paths(cv_figs)
        rpt = cv_metrics.parent / "report.md"
        if rpt.exists():
            logger.info(t("full_report", path=rpt))

    if train_metrics:
        steps_list = _load_preprocess_steps(train_metrics)
        suffix = ""
        step_descs = _describe_preprocess_steps(steps_list)
        if step_descs:
            suffix = f" ({', '.join(step_descs)})"
        logger.info(t("final_model", steps=suffix))
        _print_metrics_table(train_metrics)
        _print_confusion_paths(train_figs)
        if report_path.exists():
            logger.info(t("full_report", path=report_path))
        if model_path.exists():
            logger.info(t("model_saved", path=model_path))

    if desired_name and rename_target and rename_target != base_dir:
        shutil.move(str(base_dir), str(rename_target))
        base_dir = rename_target
        run_id = desired_name
        state["run_id"] = run_id
        _write_run_state(base_dir, state)
        logger.info(t("run_all_results_renamed", path=base_dir))

    summary_dir_path = base_dir / "summary"
    logger.info("Summary saved in %s", summary_dir_path)
    logger.info(t("results_in", path=base_dir))
    return 0
