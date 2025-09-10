from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from nircls.config_schema import ConfigModel, ValidationError
from nircls.io import prepare_inputs, summarize_dataset
from nircls.models.factory import MODEL_DESCRIPTIONS
from nircls.pipeline import run_experiment
from nircls.utils.email import enviar_email
from nircls.utils.misc import RESULTS_DIR
from nircls.utils.summarize import summarize_runs


def _figure_caption(stem: str) -> str:
    """Map figure filename stems to human-friendly captions."""

    suffix = stem.split("__")[-1]
    return {
        "confusion": "Matriz de Confusão",
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
    return run_dashboard(RESULTS_DIR)


def _render_dashboard(results_dir: Path) -> None:
    """Render Streamlit app to visualize existing results."""
    import streamlit as st

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    st.set_page_config(page_title="nircls Dashboard", page_icon="🧪", layout="wide")
    st.title("nircls Dashboard")
    st.info("Execute `nircls run-all` antes de abrir o dashboard.")

    run_dirs = [
        p
        for p in sorted(results_dir.iterdir())
        if p.is_dir() and any(p.glob("**/metrics.json"))
    ]

    def _display_figures(fig_dir: Path) -> None:
        bases = {
            p.with_suffix("")
            for p in fig_dir.glob("*")
            if p.suffix in {".png", ".json", ".html"}
        }
        for base in sorted(bases):
            json_file = base.with_suffix(".json")
            png_file = base.with_suffix(".png")
            html_file = base.with_suffix(".html")
            caption = _figure_caption(base.name)
            if json_file.exists():
                try:
                    fig = pio.from_json(json_file.read_text())
                except Exception:
                    fig = None
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(caption)
                elif png_file.exists():
                    st.image(str(png_file), caption=caption)
            elif html_file.exists():
                try:
                    import streamlit.components.v1 as components

                    components.html(html_file.read_text(), height=600, scrolling=True)
                    st.caption(caption)
                except Exception:
                    if png_file.exists():
                        st.image(str(png_file), caption=caption)
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
            if html_file.exists():
                with html_file.open("r") as f:
                    st.download_button(
                        f"Download {html_file.name}",
                        f,
                        file_name=html_file.name,
                        mime="text/html",
                    )

    config_tab, prep_tab, exec_tab, results_tab, summary_tab, help_tab = st.tabs(
        [
            "Configurar & Executar",
            "Preparar Dados",
            "Execução",
            "Resultados",
            "Resumo",
            "Ajuda",
        ]
    )

    with config_tab:
        config_path = Path.home() / ".nircls" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        def _load_cfg() -> dict:
            try:
                with config_path.open() as f:
                    return json.load(f)
            except Exception:
                return {}

        def _save_cfg(data: dict) -> None:
            with config_path.open("w") as f:
                json.dump(data, f, indent=2)

        cfg_defaults = _load_cfg()
        for key in ["input_path", "modelos", "preprocess", "email"]:
            st.session_state.setdefault(key, cfg_defaults.get(key, ""))

        with st.form("run_form"):
            preparar = st.checkbox(
                "Preparar", help="Executar preparação dos dados. Veja a aba Ajuda"
            )
            treinar = st.checkbox(
                "Treinar", help="Executar etapa de treino. Veja a aba Ajuda"
            )
            validar = st.checkbox(
                "Validar",
                help="Executar validação cruzada. Veja a aba Ajuda",
            )
            input_path = st.text_input(
                "input_path",
                key="input_path",
                help="Arquivo ou glob de entrada. Consulte a aba Ajuda",
            )
            modelos = st.text_input(
                "Modelos",
                key="modelos",
                help="Nomes dos modelos separados por vírgula. Veja a aba Ajuda",
            )
            preprocess = st.text_input(
                "Preprocessadores",
                key="preprocess",
                help="Pré-processadores separados por vírgula. Veja a aba Ajuda",
            )
            email = st.text_input("E-mail", help="Endereço para notificações")
            email = st.text_input(
                "E-mail", key="email", help="Endereço para notificações"
            )
            submitted = st.form_submit_button("Executar")

        cols = st.columns(2)
        if cols[0].button("Salvar"):
            _save_cfg(
                {
                    k: st.session_state.get(k, "")
                    for k in ["input_path", "modelos", "preprocess", "email"]
                }
            )
            st.success("Configurações salvas.")
        if cols[1].button("Carregar"):
            loaded = _load_cfg()
            for k in ["input_path", "modelos", "preprocess", "email"]:
                st.session_state[k] = loaded.get(k, "")
            st.experimental_rerun()

        if submitted:
            _save_cfg(
                {
                    k: st.session_state.get(k, "")
                    for k in ["input_path", "modelos", "preprocess", "email"]
                }
            )
            if not any([preparar, treinar, validar]):
                st.warning("Selecione ao menos uma análise.")
            else:
                cfg = {
                    "data": {"input_path": input_path},
                    "models": [
                        {"name": m.strip()} for m in modelos.split(",") if m.strip()
                    ],
                    "preprocess": {
                        "steps": [s.strip() for s in preprocess.split(",") if s.strip()]
                    },
                    "cv": {"strategies": [{"type": "kfold", "n_splits": 5}]},
                }
                try:
                    ConfigModel(**cfg)
                except ValidationError as exc:
                    st.error(f"Configuração inválida: {exc}")
                else:
                    run_dir = results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
                    try:
                        with st.spinner("Executando..."):
                            if treinar or validar:
                                run_experiment(cfg, run_dir)
                            elif preparar:
                                prepare_inputs(cfg["data"], use_cache=False)
                        st.success("Execução concluída.")
                        if email:
                            enviar_email(
                                "Execução concluída",
                                "A execução foi concluída.",
                                email,
                            )
                    except Exception as exc:
                        st.error(f"Erro na execução: {exc}")

    with prep_tab:
        uploaded = st.file_uploader(
            "Arquivo CSV ou XLSX",
            type=["csv", "xlsx", "xls"],
        )
        if uploaded is not None:
            df = (
                pd.read_csv(uploaded)
                if uploaded.name.lower().endswith(".csv")
                else pd.read_excel(uploaded)
            )
            is_num = pd.api.types.is_numeric_dtype
            non_num = [c for c in df.columns if not is_num(df[c])]
            target_col = st.selectbox("Coluna alvo", non_num)
            spectral_input = st.text_input("Faixa espectral (mín,max)")
            if target_col and st.button("Processar"):
                with tempfile.NamedTemporaryFile(
                    "wb", delete=False, suffix=Path(uploaded.name).suffix
                ) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = Path(tmp.name)
                cfg = {"input_path": str(tmp_path), "target_col": target_col}
                if spectral_input:
                    try:
                        lo, hi = map(float, spectral_input.split(","))
                        cfg["spectral_range"] = [lo, hi]
                    except Exception:
                        st.error("Formato inválido de faixa espectral")
                try:
                    ds_list = prepare_inputs(cfg, use_cache=False)
                    summary = summarize_dataset(ds_list)
                    st.subheader("Resumo do conjunto")
                    st.dataframe(summary)
                    for ds in ds_list:
                        st.markdown(f"### {ds['target']}")
                        head_df = ds["X"].copy()
                        head_df[ds["target"]] = ds["Y"]
                        st.dataframe(head_df.head())
                        counts = ds["Y"].value_counts().reset_index()
                        counts.columns = ["class", "count"]
                        fig = px.bar(counts, x="class", y="count")
                        st.plotly_chart(fig, use_container_width=True)
                        mean = ds["X"].mean()
                        std = ds["X"].std()
                        wl = mean.index.astype(float)
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=wl, y=mean, name="mean"))
                        fig2.add_trace(
                            go.Scatter(
                                x=wl,
                                y=mean + std,
                                name="+sd",
                                line=dict(width=0),
                                showlegend=False,
                            )
                        )
                        fig2.add_trace(
                            go.Scatter(
                                x=wl,
                                y=mean - std,
                                name="-sd",
                                fill="tonexty",
                                line=dict(width=0),
                                showlegend=False,
                            )
                        )
                        fig2.update_layout(
                            xaxis_title="Comprimento de onda",
                            yaxis_title="Reflectância",
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as exc:  # pragma: no cover - display error
                    st.error(f"Erro ao preparar dados: {exc}")
                finally:
                    tmp_path.unlink(missing_ok=True)

    with exec_tab:
        if not run_dirs:
            st.warning(f"Nenhuma execução encontrada em {results_dir}/.")
        else:
            selected = st.multiselect(
                "Selecione execuções",
                run_dirs,
                format_func=lambda p: p.name,
            )
            if not selected:
                st.info("Selecione uma ou mais execuções.")
            else:
                cols = st.columns(len(selected))
                for col, run in zip(cols, selected):
                    with col:
                        st.markdown(f"### {run.name}")
                        for metrics_path in sorted(run.rglob("metrics.json")):
                            strat_dir = metrics_path.parent
                            st.subheader(strat_dir.relative_to(run).as_posix())
                            try:
                                with metrics_path.open() as f:
                                    metrics = json.load(f)
                                if metrics:
                                    df = pd.DataFrame(metrics)
                                    if "n_components" in df.columns:
                                        df["n_components"] = df["n_components"].astype(
                                            "Int64"
                                        )
                                        cols = df.columns.tolist()
                                        if "model" in cols:
                                            cols.insert(
                                                cols.index("model") + 1,
                                                cols.pop(cols.index("n_components")),
                                            )
                                        df = df[cols]
                                    st.dataframe(df)
                            except Exception as exc:  # pragma: no cover - display error
                                st.error(f"Erro ao carregar métricas: {exc}")
                            fig_dir = strat_dir / "figures"
                            if fig_dir.exists():
                                _display_figures(fig_dir)
                            qc_path = strat_dir / "qc_report.json"
                            if qc_path.exists():
                                try:
                                    with qc_path.open() as f:
                                        qc = json.load(f)
                                    for item in qc:
                                        st.markdown(f"#### QC: {item['target']}")
                                        df_qc = pd.DataFrame.from_dict(
                                            item["pca_outliers"], orient="index"
                                        )
                                        t2_lim = df_qc["T2"].quantile(0.95)
                                        q_lim = df_qc["Q_residuals"].quantile(0.95)
                                        df_qc["outlier"] = (df_qc["T2"] > t2_lim) | (
                                            df_qc["Q_residuals"] > q_lim
                                        )
                                        fig = px.scatter(
                                            df_qc,
                                            x="PC1",
                                            y="PC2",
                                            color="outlier",
                                            color_discrete_map={
                                                True: "red",
                                                False: "blue",
                                            },
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                except (
                                    Exception
                                ) as exc:  # pragma: no cover - display error
                                    st.error(f"Erro ao carregar relatório QC: {exc}")

    with results_tab:
        if not run_dirs:
            st.warning(f"Nenhuma execução encontrada em {results_dir}/.")
        else:
            compare_runs = st.multiselect(
                "Selecione execuções para comparar",
                run_dirs,
                format_func=lambda p: p.name,
            )
            if len(compare_runs) >= 2:
                cols = st.columns(len(compare_runs))
                for col, run in zip(cols, compare_runs):
                    with col:
                        st.markdown(f"### {run.name}")
                        for metrics_path in sorted(run.rglob("metrics.json")):
                            strat_dir = metrics_path.parent
                            st.caption(strat_dir.relative_to(run).as_posix())
                            try:
                                with metrics_path.open() as f:
                                    metrics = json.load(f)
                                if metrics:
                                    df = pd.DataFrame(metrics)
                                    if "n_components" in df.columns:
                                        df["n_components"] = df["n_components"].astype(
                                            "Int64"
                                        )
                                        cols = df.columns.tolist()
                                        if "model" in cols:
                                            cols.insert(
                                                cols.index("model") + 1,
                                                cols.pop(cols.index("n_components")),
                                            )
                                        df = df[cols]
                                    st.dataframe(df)
                            except Exception as exc:  # pragma: no cover
                                st.error(f"Erro ao carregar métricas: {exc}")
                            fig_dir = strat_dir / "figures"
                            if fig_dir.exists():
                                _display_figures(fig_dir)
            elif compare_runs:
                st.warning("Selecione ao menos duas execuções para comparar.")

    with summary_tab:
        if not run_dirs:
            st.warning(f"Nenhuma execução encontrada em {results_dir}/.")
        else:
            summary = summarize_runs(results_dir)
            if summary.empty:
                st.warning("Nenhuma métrica agregada encontrada.")
            else:
                summary["run"] = summary["run"].str.split("/").str[0]
                num_cols = summary.select_dtypes("number").columns
                grouped_mean = summary.groupby("run")[num_cols].mean()
                grouped_std = summary.groupby("run")[num_cols].std().fillna(0)
                display_df = pd.DataFrame(
                    {
                        c: grouped_mean[c].map("{:.3f}".format)
                        + " ± "
                        + grouped_std[c].map("{:.3f}".format)
                        for c in grouped_mean.columns
                    }
                )

                def _highlight_best(col: pd.Series) -> list[str]:
                    cname = col.name.lower()
                    best = (
                        grouped_mean[col.name].idxmin()
                        if any(k in cname for k in ["loss", "erro", "rmse"])
                        else grouped_mean[col.name].idxmax()
                    )
                    return [
                        "background-color: #dff0d8" if idx == best else ""
                        for idx in grouped_mean.index
                    ]

                styled = display_df.style.apply(_highlight_best, axis=0)
                st.dataframe(styled, use_container_width=True)

    with help_tab:
        st.subheader("Modelos disponíveis")
        help_df = pd.DataFrame(
            [(k, v) for k, v in sorted(MODEL_DESCRIPTIONS.items())],
            columns=["Modelo", "Descrição"],
        )
        st.dataframe(help_df, hide_index=True)
        st.caption(
            "Aplicações típicas variam conforme o contexto; consulte a literatura especializada."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Diretório de saída")
    ns = parser.parse_args()
    _render_dashboard(Path(ns.results))
