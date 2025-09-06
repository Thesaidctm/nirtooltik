"""Command-line interface for the NIR species classification toolkit.

This module exposes the :func:`main` function which backs the ``nircls``
command-line utility for preparing data, training models, evaluating saved
models, making predictions and generating explanations.
"""

from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

from .io import prepare_inputs, export_xy_xlsx
from .pipeline import (
    run_multiple_eval,
    train_and_save,
    evaluate_saved_model,
    predict_with_model,
    explain_model,
)
from .utils.logging import setup_logging, log_versions
from .utils.misc import ensure_dirs, make_run_id

def main(argv=None):
    """Run the ``nircls`` command-line interface.

    Parameters
    ----------
    argv : list[str] | None
        Optional list of arguments. When ``None`` (default), ``sys.argv`` is
        used.
    """

    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="nircls", description="NIR Species Classification Toolkit"
    )
    parser.add_argument(
        "--run",
        choices=["one", "all"],
        default="all",
        help="Rodar somente uma ou todas as análises",
    )
    sp = parser.add_subparsers(dest="cmd", required=True)
    p_prep = sp.add_parser("prepare-data")
    p_prep.add_argument("--config", required=True)
    p_tr = sp.add_parser("train")
    p_tr.add_argument("--config", required=True)
    p_cv = sp.add_parser("crossval")
    p_cv.add_argument("--config", required=True)
    p_ev = sp.add_parser("evaluate")
    p_ev.add_argument("--model", required=True)
    p_ev.add_argument("--data", required=True)
    p_pd = sp.add_parser("predict")
    p_pd.add_argument("--model", required=True)
    p_pd.add_argument("--data", required=True)
    p_pd.add_argument("--output", default="results")
    p_ex = sp.add_parser("explain")
    p_ex.add_argument("--model", required=True)
    p_ex.add_argument("--data", required=True)
    p_ex.add_argument("--output", default="results")
    args = parser.parse_args(argv)
    setup_logging()
    log_versions()
    if args.cmd == "prepare-data":
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        run_id = make_run_id("prepare")
        out_dir = Path(cfg.get("data", {}).get("out_dir", "results")) / run_id
        ensure_dirs(out_dir)
        ds_list = prepare_inputs(cfg["data"])
        if args.run == "one":
            ds_list = ds_list[:1]
        for ds in ds_list:
            out_file = out_dir / f"prepared_{ds['target']}.xlsx"
            export_xy_xlsx(
                out_file,
                ds["X"],
                ds["Y"],
                cfg["data"].get("x_sheet", "X"),
                cfg["data"].get("y_sheet", "Y"),
            )
    elif args.cmd == "train":
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        run_id = make_run_id("train")
        out_dir = Path(cfg.get("data", {}).get("out_dir", "results")) / run_id
        ensure_dirs(out_dir)
        train_and_save(cfg, out_dir)
    elif args.cmd == "crossval":
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if args.run == "one":
            first = prepare_inputs(cfg["data"])
            if first:
                cfg["data"]["target_cols"] = first[0]["target"]
        run_id = make_run_id("cv")
        out_dir = Path(cfg.get("data", {}).get("out_dir", "results")) / run_id
        ensure_dirs(out_dir)
        run_multiple_eval(cfg, out_dir)
    elif args.cmd == "evaluate":
        evaluate_saved_model(Path(args.model), Path(args.data))
    elif args.cmd == "predict":
        predict_with_model(Path(args.model), Path(args.data), Path(args.output))
    elif args.cmd == "explain":
        explain_model(Path(args.model), Path(args.data), Path(args.output))

if __name__ == "__main__":
    main()
