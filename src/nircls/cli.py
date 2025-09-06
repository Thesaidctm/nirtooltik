"""Command-line interface for the NIR species classification toolkit.

This module exposes the :func:`main` function which backs the ``nircls``
command-line utility for preparing data, training models, evaluating saved
models, making predictions and generating explanations.
"""

from __future__ import annotations

import argparse
import sys
import os
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
from .config_schema import ConfigModel, ValidationError

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
    parser.add_argument(
        "--ai-summary",
        action="store_true",
        help="Gerar resumo via ChatGPT no relatório",
    )
    parser.add_argument(
        "--openai-key",
        help="Chave da API OpenAI (ou usar OPENAI_API_KEY)",
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
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    if args.cmd in {"prepare-data", "train", "crossval"}:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        try:
            cfg = ConfigModel(**raw)
        except ValidationError as e:  # pragma: no cover - user feedback
            print(f"Configuração inválida: {e}", file=sys.stderr)
            return 1

    if args.cmd == "prepare-data":
        if cfg.data is None:
            print("Configuração inválida: seção 'data' é obrigatória", file=sys.stderr)
            return 1
        run_id = make_run_id("prepare")
        out_dir = Path(cfg.data.out_dir or "results") / run_id
        ensure_dirs(out_dir)
        ds_list = prepare_inputs(cfg.data.dict())
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
    elif args.cmd == "train":
        if cfg.data is None or not cfg.models:
            print(
                "Configuração inválida: seções 'data' e 'models' são obrigatórias",
                file=sys.stderr,
            )
            return 1
        run_id = make_run_id("train")
        out_dir = Path(cfg.data.out_dir or "results") / run_id
        ensure_dirs(out_dir)
        train_and_save(
            cfg.dict(),
            out_dir,
            ai_summary=args.ai_summary,
            openai_key=openai_key,
        )
    elif args.cmd == "crossval":
        if cfg.data is None or not cfg.models or cfg.cv is None:
            print(
                "Configuração inválida: seções 'data', 'models' e 'cv' são obrigatórias",
                file=sys.stderr,
            )
            return 1
        if args.run == "one":
            first = prepare_inputs(cfg.data.dict())
            if first:
                cfg.data.target_cols = first[0]["target"]
        run_id = make_run_id("cv")
        out_dir = Path(cfg.data.out_dir or "results") / run_id
        ensure_dirs(out_dir)
        run_multiple_eval(
            cfg.dict(),
            out_dir,
            ai_summary=args.ai_summary,
            openai_key=openai_key,
        )
    elif args.cmd == "evaluate":
        evaluate_saved_model(Path(args.model), Path(args.data))
    elif args.cmd == "predict":
        predict_with_model(Path(args.model), Path(args.data), Path(args.output))
    elif args.cmd == "explain":
        explain_model(Path(args.model), Path(args.data), Path(args.output))

if __name__ == "__main__":
    main()
