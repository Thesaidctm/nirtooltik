from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from ..utils.logging import log_versions, setup_logging
from ..utils.misc import ensure_dirs, set_seed
from . import (
    cmd_crossval,
    cmd_dashboard,
    cmd_demo,
    cmd_evaluate,
    cmd_explain,
    cmd_list_models,
    cmd_predict,
    cmd_prepare,
    cmd_run_all,
    cmd_summary,
    cmd_train,
    cmd_wizard,
)
from . import common
from .common import __version__, set_language, t

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    default_lang = common.LANG
    parser = argparse.ArgumentParser(prog="nircls", description=t("prog_desc"))
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--lang", choices=["pt", "en"], default=default_lang, help=t("lang_help"))
    parser.add_argument("--run", choices=["one", "all"], default="all", help=t("run_help"))
    parser.add_argument("--seed", type=int, default=42, help=t("seed_help"))
    parser.add_argument("--mlflow", help=t("mlflow_help"))
    parser.add_argument("--n-jobs", type=int, default=1, help=t("n_jobs_help"))
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file (overrides default)",
    )
    parser.add_argument(
        "--cv-verbose",
        type=int,
        default=0,
        help="GridSearchCV verbosity",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Log memory usage during processing",
    )
    parser.add_argument(
        "--notification-email",
        help="Email address to receive notifications",
    )
    sp = parser.add_subparsers(dest="cmd", required=True)
    cmd_prepare.register_subparser(sp)
    cmd_train.register_subparser(sp)
    cmd_crossval.register_subparser(sp)
    cmd_demo.register_subparser(sp)
    cmd_list_models.register_subparser(sp)
    cmd_run_all.register_subparser(sp)
    cmd_wizard.register_subparser(sp)
    cmd_evaluate.register_subparser(sp)
    cmd_predict.register_subparser(sp)
    cmd_explain.register_subparser(sp)
    cmd_dashboard.register_subparser(sp)
    cmd_summary.register_subparser(sp)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``nircls`` command-line interface."""

    argv = list(argv) if argv is not None else sys.argv[1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang", choices=["pt", "en"], default="en")
    ns, argv = pre.parse_known_args(argv)
    set_language(ns.lang)

    parser = _build_parser()
    args = parser.parse_args(argv)

    set_language(args.lang)
    set_seed(args.seed)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    if args.log_file:
        ensure_dirs(args.log_file.parent)
    setup_logging(level=log_level, log_file=args.log_file)
    log_versions()

    args.log_level_value = log_level
    args.n_jobs = max(1, args.n_jobs)
    args.mlflow_uri = args.mlflow

    if args.mlflow_uri:
        try:
            import mlflow
        except Exception:
            logger.error(t("mlflow_not_installed"), exc_info=True)
            return 1
        mlflow.set_tracking_uri(args.mlflow_uri)

    if hasattr(args, "func"):
        return args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
