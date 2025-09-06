from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import argparse
import json
import tempfile
from datetime import datetime

import pandas as pd

from .io import prepare_inputs
from .pipeline import train_and_save, run_multiple_eval, predict_with_model
from .models.factory import MODEL_BUILDERS
from . import DEFAULT_RESULTS_DIR


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
    return subprocess.run(cmd, check=False).returncode


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

    tab_prep, tab_train, tab_eval = st.tabs(
        ["Preparar Dados", "Treinar", "Avaliar"]
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
                train_and_save(cfg, out_dir)
                progress.progress(100)
                status.write("Concluído!")
                st.success(f"Modelo salvo em {out_dir}")
                st.session_state["model_dir"] = out_dir

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
                run_multiple_eval(cfg, out_dir)
                progress.progress(100)
                status.write("Concluído!")
                st.success(f"Resultados salvos em {out_dir}")
                st.session_state["eval_dir"] = out_dir

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
            st.success(f"Arquivo predictions.csv em {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Diretório de saída")
    ns = parser.parse_args()
    _render_dashboard(Path(ns.results))
