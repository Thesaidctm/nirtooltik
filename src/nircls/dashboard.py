from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import argparse
import json
import tempfile
from datetime import datetime
import copy
import re
import webbrowser

import pandas as pd
import yaml
import plotly.graph_objects as go

from .io import prepare_inputs
from .pipeline import train_and_save, run_multiple_eval, predict_with_model
from .report import generate_report
from .models.factory import MODEL_BUILDERS, DEFAULT_GRIDS
from . import DEFAULT_RESULTS_DIR


def _load_plotly_html(path: Path) -> go.Figure | None:
    """Return a Plotly figure reconstructed from an HTML file.

    Parameters
    ----------
    path:
        Path to the HTML file produced by ``fig.write_html``.
    """

    try:
        html = path.read_text(encoding="utf-8")
        match = re.search(
            r"Plotly.newPlot\(\"[^\"]+\",\s*(\[.*?\]),\s*(\{.*?\})",
            html,
            re.S,
        )
        if not match:
            return None
        data = json.loads(match.group(1))
        layout = json.loads(match.group(2))
        return go.Figure(data=data, layout=layout)
    except Exception:
        return None


def _figure_caption(stem: str) -> str:
    """Map figure filename stems to human-friendly captions."""

    suffix = stem.split("__")[-1]
    return {
        "confusion_raw": "Matriz de Confusão (Bruta)",
        "confusion_norm": "Matriz de Confusão (Normalizada)",
        "roc": "Curva ROC",
        "reliability": "Diagrama de Confiabilidade",
    }.get(suffix, stem)

def run_dashboard(results_dir: str | Path) -> int:
    """Launch the Streamlit dashboard.

    Parameters
    ----------
    results_dir : str | Path
        Root directory where run artefacts will be stored.
    """
    module_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(module_path),
        "--",
        "--results",
        str(results_dir),
    ]
    proc = subprocess.Popen(cmd)
    try:
        port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    return proc.wait()


def main() -> int:
    """Entry point used by the ``nircls-dashboard`` console script."""
    return run_dashboard(DEFAULT_RESULTS_DIR)


def _render_dashboard(results_dir: Path) -> None:
    """Render the Streamlit application that orchestrates experiments."""
    import streamlit as st

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    st.set_page_config(page_title="nircls Dashboard")
    st.title("nircls Dashboard")

    tab_prep, tab_train, tab_eval, tab_cfg = st.tabs(
        ["Preparar Dados", "Treinar", "Avaliar", "Gerar Configuração"]
    )

    def _display_figures(fig_dir: Path) -> None:
        bases = {
            p.with_suffix("")
            for p in fig_dir.glob("*")
            if p.suffix in {".png", ".html"}
        }
        for base in sorted(bases):
            html_file = base.with_suffix(".html")
            png_file = base.with_suffix(".png")
            caption = _figure_caption(base.name)
            fig = _load_plotly_html(html_file) if html_file.exists() else None
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(caption)
            elif png_file.exists():
                st.image(str(png_file), caption=caption)
            if png_file.exists():
                with png_file.open("rb") as f:
                    st.download_button(
                        f"Download {png_file.name}",
                        f,
                        file_name=png_file.name,
                        mime="image/png",
                    )

    # ------------------------------------------------------------------
    # Preparar Dados
    # ------------------------------------------------------------------
    with tab_prep:
        st.header("Preparar Dados")
        with st.form("prepare_form"):
            uploaded = st.file_uploader(
                "Arquivo de entrada", type=["csv", "xlsx", "xls"]
            )
            spectral = st.text_input(
                "Intervalo espectral (ex: 900,1700)", value=""
            )
            target_col = None
            if uploaded is not None:
                uploaded.seek(0)
                if uploaded.name.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)
                cols = [
                    c
                    for c in df.columns
                    if not pd.api.types.is_numeric_dtype(df[c])
                ] or list(df.columns)
                target_col = st.selectbox("Coluna alvo", cols)
                uploaded.seek(0)
            submit_prep = st.form_submit_button("Preparar")

        if submit_prep and uploaded and target_col:
            tmpdir = Path(tempfile.mkdtemp())
            data_path = tmpdir / uploaded.name
            data_path.write_bytes(uploaded.getbuffer())
            data_cfg = {"input_path": str(data_path), "target_col": target_col}
            if spectral:
                try:
                    lo, hi = [float(x.strip()) for x in spectral.split(",")]
                    data_cfg["spectral_range"] = [lo, hi]
                except Exception:
                    st.warning("Intervalo espectral ignorado (formato inválido).")
            progress = st.progress(0)
            status = st.empty()
            status.write("Processando...")
            progress.progress(10)
            ds_list = prepare_inputs(data_cfg)
            progress.progress(100)
            status.write("Concluído!")
            st.success(f"{len(ds_list)} conjunto(s) preparado(s).")
            if ds_list:
                st.dataframe(ds_list[0]["X"].head())
            st.session_state["data_cfg"] = data_cfg

    # ------------------------------------------------------------------
    # Treinar
    # ------------------------------------------------------------------
    with tab_train:
        st.header("Treinar")
        if "data_cfg" not in st.session_state:
            st.info("Prepare os dados na aba anterior.")
        else:
            model_names = sorted(MODEL_BUILDERS.keys())
            with st.form("train_form"):
                model_name = st.selectbox("Modelo", model_names)
                grid_text = st.text_input(
                    "Grid de parâmetros (JSON)", value=""
                )
                submit_train = st.form_submit_button("Treinar")

            if submit_train:
                try:
                    grid = json.loads(grid_text) if grid_text.strip() else {}
                except Exception:
                    st.error("Grid de parâmetros inválido. Use JSON.")
                    grid = {}
                cfg = {
                    "data": st.session_state["data_cfg"],
                    "models": [{"name": model_name, "grid": grid}],
                }
                run_id = datetime.now().strftime("train_%Y%m%d_%H%M%S")
                out_dir = results_dir / run_id
                progress = st.progress(0)
                status = st.empty()
                status.write("Treinando...")
                progress.progress(10)
                train_and_save(cfg, out_dir, interactive=True)
                progress.progress(100)
                status.write("Concluído!")
                st.success(f"Modelo salvo em {out_dir}")
                st.session_state["model_dir"] = out_dir

                metrics_path = out_dir / "metrics.json"
                if metrics_path.exists():
                    try:
                        with metrics_path.open() as f:
                            metrics = json.load(f)
                        if metrics:
                            st.dataframe(pd.DataFrame(metrics))
                    except Exception as exc:
                        st.error(f"Erro ao carregar métricas: {exc}")
                fig_dir = out_dir / "figures"
                if fig_dir.exists():
                    _display_figures(fig_dir)

    # ------------------------------------------------------------------
    # Avaliar
    # ------------------------------------------------------------------
    with tab_eval:
        st.header("Avaliar")
        if "data_cfg" not in st.session_state:
            st.info("Prepare os dados na aba 'Preparar Dados'.")
        else:
            model_names = sorted(MODEL_BUILDERS.keys())
            with st.form("eval_form"):
                model_name = st.selectbox(
                    "Modelo", model_names, key="eval_model"
                )
                grid_text = st.text_input(
                    "Grid de parâmetros (JSON)", value="", key="eval_grid"
                )
                n_splits = st.number_input(
                    "Número de folds", min_value=2, value=5, step=1
                )
                submit_eval = st.form_submit_button("Executar avaliação")

            if submit_eval:
                try:
                    grid = json.loads(grid_text) if grid_text.strip() else {}
                except Exception:
                    st.error("Grid de parâmetros inválido. Use JSON.")
                    grid = {}
                cfg = {
                    "data": st.session_state["data_cfg"],
                    "models": [{"name": model_name, "grid": grid}],
                    "cv": {
                        "random_state": 0,
                        "strategies": [
                            {
                                "type": "kfold",
                                "n_splits": [int(n_splits)],
                                "n_repeats": 1,
                            }
                        ],
                    },
                }
                run_id = datetime.now().strftime("cv_%Y%m%d_%H%M%S")
                out_dir = results_dir / run_id
                progress = st.progress(0)
                status = st.empty()
                status.write("Executando avaliação...")
                progress.progress(10)
                run_multiple_eval(cfg, out_dir, interactive=True)
                progress.progress(100)
                status.write("Concluído!")
                st.success(f"Resultados salvos em {out_dir}")
                st.session_state["eval_dir"] = out_dir

                # Exibir métricas e figuras geradas
                for metrics_path in out_dir.rglob("metrics.json"):
                    strat_dir = metrics_path.parent
                    st.subheader(f"Métricas - {strat_dir.name}")
                    try:
                        with metrics_path.open() as f:
                            metrics = json.load(f)
                        if metrics:
                            st.dataframe(pd.DataFrame(metrics))
                    except Exception as exc:  # pragma: no cover - display error
                        st.error(f"Erro ao carregar métricas: {exc}")
                    fig_dir = strat_dir / "figures"
                    if fig_dir.exists():
                        _display_figures(fig_dir)

                    pdf_key = f"pdf_{strat_dir.name}"
                    if st.button("Gerar relatório PDF", key=pdf_key):
                        pdf_path = strat_dir / "report.pdf"
                        try:
                            generate_report(
                                metrics_path, fig_dir, pdf_path
                            )
                            with pdf_path.open("rb") as f:
                                st.download_button(
                                    "Download relatório PDF",
                                    f,
                                    file_name=pdf_path.name,
                                    mime="application/pdf",
                                    key=f"dl_{pdf_key}",
                                )
                        except Exception as exc:  # pragma: no cover
                            st.error(f"Erro ao gerar PDF: {exc}")

        st.subheader("Predições com modelo salvo")
        with st.form("pred_form"):
            model_file = st.file_uploader(
                "Modelo (.joblib)", type=["joblib"], key="pred_model"
            )
            data_file = st.file_uploader(
                "Dados para predição", type=["csv", "xlsx", "xls"], key="pred_data"
            )
            submit_pred = st.form_submit_button("Gerar predições")

        if submit_pred and model_file and data_file:
            tmpdir = Path(tempfile.mkdtemp())
            model_path = tmpdir / model_file.name
            model_path.write_bytes(model_file.getbuffer())
            data_path = tmpdir / data_file.name
            data_path.write_bytes(data_file.getbuffer())
            out_dir = results_dir / datetime.now().strftime("pred_%Y%m%d_%H%M%S")
            progress = st.progress(0)
            status = st.empty()
            status.write("Gerando predições...")
            progress.progress(10)
            predict_with_model(model_path, data_path, out_dir)
            progress.progress(100)
            status.write("Concluído!")
            pred_file = out_dir / "predictions.csv"
            st.success(f"Arquivo predictions.csv em {out_dir}")
            if pred_file.exists():
                with pred_file.open("rb") as f:
                    st.download_button(
                        "Download predictions.csv",
                        f,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )


    # ------------------------------------------------------------------
    # Gerar Configuração
    # ------------------------------------------------------------------
    with tab_cfg:
        st.header("Gerar Configuração")
        cfg_dir = Path(__file__).resolve().parents[2] / "configs"
        templates = {
            p.name: yaml.safe_load(p.read_text()) for p in cfg_dir.glob("*.yaml")
        }
        tpl_name = st.selectbox(
            "Template", ["<vazio>"] + sorted(templates.keys())
        )
        tpl_cfg = copy.deepcopy(templates.get(tpl_name, {}))
        data_cfg = tpl_cfg.get("data", {})
        model_cfg = tpl_cfg.get("models", [{}])[0]
        pre_cfg = tpl_cfg.get("preprocess", {})
        with st.form("config_form"):
            upload = st.file_uploader(
                "Arquivo de dados (Excel)", type=["xlsx", "xls"], key="cfg_file"
            )
            x_sheet = st.text_input(
                "Planilha X", value=data_cfg.get("x_sheet", "X")
            )
            y_sheet = st.text_input(
                "Planilha Y", value=data_cfg.get("y_sheet", "Y")
            )
            pre_opts = ["SNV", "MSC", "DETREND", "SG", "EMSC", "EPO", "SCALE"]
            pre_sel = st.multiselect(
                "Pré-processadores", pre_opts, default=pre_cfg.get("steps", [])
            )
            model_names = sorted(MODEL_BUILDERS.keys())
            default_model = model_cfg.get("name", model_names[0])
            model_idx = (
                model_names.index(default_model)
                if default_model in model_names
                else 0
            )
            model_name = st.selectbox("Modelo", model_names, index=model_idx)
            default_grid = model_cfg.get(
                "grid", DEFAULT_GRIDS.get(model_name.upper(), {})
            )
            grid_text = st.text_input(
                "Hiperparâmetros (JSON)", value=json.dumps(default_grid)
            )
            submit_cfg = st.form_submit_button("Salvar config")

        if submit_cfg:
            if upload is None:
                st.error("Envie um arquivo de dados.")
            else:
                try:
                    grid = json.loads(grid_text) if grid_text.strip() else {}
                except Exception:
                    st.error("Grid de parâmetros inválido. Use JSON.")
                    grid = {}
                run_id = datetime.now().strftime("cfg_%Y%m%d_%H%M%S")
                out_dir = results_dir / run_id
                out_dir.mkdir(parents=True, exist_ok=True)
                data_path = out_dir / upload.name
                data_path.write_bytes(upload.getbuffer())
                cfg = copy.deepcopy(tpl_cfg)
                cfg.setdefault("data", {})
                cfg["data"].update(
                    {
                        "input_path": data_path.as_posix(),
                        "x_sheet": x_sheet,
                        "y_sheet": y_sheet,
                    }
                )
                pre_cfg = cfg.get("preprocess", {})
                pre_cfg["steps"] = pre_sel
                cfg["preprocess"] = pre_cfg
                cfg["models"] = [{"name": model_name, "grid": grid}]
                config_file = out_dir / "config.yaml"
                with config_file.open("w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
                st.success(f"Configuração salva em {config_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Diretório de saída")
    ns = parser.parse_args()
    _render_dashboard(Path(ns.results))
