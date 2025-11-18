from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Union

import yaml  # type: ignore[import-untyped]

from ..config_schema import ConfigModel
from ..core.settings import settings
from ..processing.models.factory import get_model_builders, get_model_descriptions
from ..processing.pipeline import run_multiple_eval, train_and_save
from ..utils.errors import DataError, ModelError
from ..utils.misc import ensure_dirs, make_run_id
from .common import (
    PREPROCESS_DESCRIPTIONS,
    _get_preprocess_description,
    _parse_params,
    _print_confusion_paths,
    _print_metrics_table,
    setup_run_logging,
    t,
)

logger = logging.getLogger(__name__)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("wizard")
    parser.set_defaults(func=run_wizard)


def run_wizard(args: argparse.Namespace) -> int:
    try:  # pragma: no cover - interactive session
        import questionary
        from questionary import Choice
    except Exception:  # pragma: no cover - missing optional dependency
        logger.error(t("questionary_not_installed"), exc_info=True)
        return 1

    def _validate_path(p: str) -> Union[bool, str]:
        return Path(p).exists() or t("wizard_file_not_found")

    input_path = questionary.text(t("wizard_data_path"), validate=_validate_path).ask()
    target = questionary.text(
        t("wizard_target_col"),
        validate=lambda t_val: bool(t_val) or t("wizard_target_required"),
    ).ask()

    builders = get_model_builders()
    model_descs = get_model_descriptions()
    model_choices = [
        Choice(title=f"{k} - {model_descs.get(k, k)}", value=k) for k in builders.keys()
    ]
    selected_models = questionary.checkbox(
        t("wizard_models"),
        choices=model_choices,
        validate=lambda m: bool(m) or t("wizard_models_select"),
    ).ask()

    models_cfg = []
    for name in selected_models:
        params_raw = questionary.text(t("wizard_models_params", name=name)).ask()
        model_entry: dict[str, Any] = {"name": name}
        if params_raw:
            params = _parse_params(params_raw)
            if params:
                model_entry["params"] = params
        models_cfg.append(model_entry)

    pre_choices = [
        Choice(title=f"{k} - {_get_preprocess_description(k)}", value=k)
        for k in PREPROCESS_DESCRIPTIONS
    ]
    preprocess_steps = questionary.checkbox(
        t("wizard_preprocessors"), choices=pre_choices
    ).ask()

    out_dir_str = questionary.text(t("wizard_out_dir_prompt"), default=".").ask() or "."
    notification_email = questionary.text("Notification email (optional):").ask()
    out_path = Path(out_dir_str)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg_dict = {
        "data": {"input_path": input_path, "target_cols": target},
        "models": models_cfg,
        "cv": {"strategies": [{"type": "holdout", "test_size": 0.2}]},
    }
    if preprocess_steps:
        cfg_dict["preprocess"] = {"steps": preprocess_steps}
    if notification_email:
        cfg_dict["notification_email"] = notification_email
    cfg_file = out_path / "config.yaml"
    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg_dict, f)
    logger.info(t("wizard_config_saved", file=cfg_file))
    run_now = questionary.confirm(t("wizard_run_now")).ask()
    if run_now:
        cfg_obj = ConfigModel(**cfg_dict)
        run_id = make_run_id("wizard")
        setup_run_logging(run_id, args.log_level_value, args.log_file)
        base_dir = Path(cfg_obj.data.out_dir or settings.RESULTS_DIR) / run_id
        ensure_dirs(base_dir)
        train_dir = base_dir / "train"
        ensure_dirs(train_dir)
        try:
            metrics_path, figs_dir = train_and_save(
                cfg_obj.dict(),
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
            logger.info(t("wizard_train_in", path=train_dir))
            cv_dir = base_dir / "cv"
            ensure_dirs(cv_dir)
            outputs, summary_dir = run_multiple_eval(
                cfg_obj.dict(),
                cv_dir,
                random_state=args.seed,
                n_jobs=args.n_jobs,
                mlflow_uri=args.mlflow_uri,
                log_memory=args.log_memory,
            )
            for metrics_path, figs_dir in outputs:
                _print_metrics_table(metrics_path)
                _print_confusion_paths(figs_dir)
            logger.info(t("wizard_eval_in", path=cv_dir))
            logger.info("Summary saved in %s", summary_dir)
        except DataError as e:
            logger.error(t("invalid_data", err=e), exc_info=True)
        except ModelError as e:
            logger.error(t("invalid_model", err=e), exc_info=True)
    return 0
