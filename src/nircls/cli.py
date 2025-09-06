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
from typing import Any

from .io import prepare_inputs, export_xy_xlsx
from .pipeline import (
    run_multiple_eval,
    train_and_save,
    evaluate_saved_model,
    predict_with_model,
    explain_model,
)
from .dashboard import run_dashboard
from .utils.logging import setup_logging, log_versions
from .utils.misc import ensure_dirs, make_run_id
from .config_schema import ConfigModel, ValidationError


def _parse_params(s: str) -> dict[str, Any]:
    """Parse ``key=value`` pairs into a dictionary."""
    params: dict[str, Any] = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k.strip()] = yaml.safe_load(v.strip())
    return params

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
    p_demo = sp.add_parser("demo")
    p_wz = sp.add_parser("wizard")
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
    p_db = sp.add_parser("dashboard")
    p_db.add_argument("--results", required=True, help="Pasta com resultados")
    args = parser.parse_args(argv)
    setup_logging()
    log_versions()
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
        )
    elif args.cmd == "demo":
        demo_cfg = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"
        with open(demo_cfg) as f:
            raw = yaml.safe_load(f)
        raw["data"]["input_path"] = str(
            (demo_cfg.parent.parent / raw["data"]["input_path"]).resolve()
        )
        try:
            cfg = ConfigModel(**raw)
        except ValidationError as e:  # pragma: no cover - user feedback
            print(f"Configuração inválida: {e}", file=sys.stderr)
            return 1
        run_id = make_run_id("demo")
        base_dir = Path(cfg.data.out_dir or "results") / run_id
        ensure_dirs(base_dir)
        ds_list = prepare_inputs(cfg.data.dict())
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
            print(f"Dados preparados em: {prep_file}")
        train_dir = base_dir / "train"
        ensure_dirs(train_dir)
        train_and_save(cfg.dict(), train_dir)
        print(f"Modelo treinado em: {train_dir}")
        cv_dir = base_dir / "cv"
        ensure_dirs(cv_dir)
        run_multiple_eval(cfg.dict(), cv_dir)
        print(f"Resultados de cross-val em: {cv_dir}")
    elif args.cmd == "wizard":
        input_path = input("Caminho dos dados: ").strip()
        target = input("Coluna alvo: ").strip()
        models_raw = input(
            "Modelos (separados por vírgula, ex: RF,SVM_RBF): "
        ).strip()
        models_cfg = []
        for name in models_raw.split(","):
            name = name.strip()
            if not name:
                continue
            params_raw = input(
                f"Parâmetros para {name} (k=v, separados por vírgula, opcional): "
            ).strip()
            model_entry: dict[str, Any] = {"name": name}
            if params_raw:
                params = _parse_params(params_raw)
                if params:
                    model_entry["params"] = params
            models_cfg.append(model_entry)
        out_dir_str = input(
            "Diretório para salvar config.yaml (padrão: .): "
        ).strip() or "."
        out_path = Path(out_dir_str)
        out_path.mkdir(parents=True, exist_ok=True)
        cfg_dict = {
            "data": {"input_path": input_path, "target_cols": target},
            "models": models_cfg,
            "cv": {"strategies": [{"type": "holdout", "test_size": 0.2}]},
        }
        cfg_file = out_path / "config.yaml"
        with open(cfg_file, "w") as f:
            yaml.safe_dump(cfg_dict, f)
        print(f"Configuração salva em: {cfg_file}")
        run_now = input("Iniciar treino e avaliação agora? (s/n): ").strip().lower()
        if run_now.startswith("s"):
            cfg_obj = ConfigModel(**cfg_dict)
            run_id = make_run_id("wizard")
            base_dir = Path(cfg_obj.data.out_dir or "results") / run_id
            ensure_dirs(base_dir)
            train_dir = base_dir / "train"
            ensure_dirs(train_dir)
            train_and_save(cfg_obj.dict(), train_dir)
            cv_dir = base_dir / "cv"
            ensure_dirs(cv_dir)
            run_multiple_eval(cfg_obj.dict(), cv_dir)
            print(f"Treino em: {train_dir}")
            print(f"Avaliação em: {cv_dir}")
    elif args.cmd == "evaluate":
        evaluate_saved_model(Path(args.model), Path(args.data))
    elif args.cmd == "predict":
        predict_with_model(Path(args.model), Path(args.data), Path(args.output))
    elif args.cmd == "explain":
        explain_model(Path(args.model), Path(args.data), Path(args.output))
    elif args.cmd == "dashboard":
        run_dashboard(Path(args.results))

if __name__ == "__main__":
    main()
