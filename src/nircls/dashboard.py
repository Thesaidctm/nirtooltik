from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

import keyring
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import nircls.preprocessors as preprocessors
from nircls.config_schema import ConfigModel, ValidationError
from nircls.io import prepare_inputs, summarize_dataset
from nircls.models.factory import MODEL_DESCRIPTIONS
from nircls.pipeline import run_multiple_eval, train_and_save
from nircls.utils.email import _load_config as _load_email_config
from nircls.utils.email import _save_config as _save_email_config
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


def _get_preprocessor_names() -> list[str]:
    """Discover available preprocessing steps."""

    names: list[str] = []
    for _, mod_name, _ in pkgutil.iter_modules(preprocessors.__path__):
        module = importlib.import_module(f"{preprocessors.__name__}.{mod_name}")
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                names.append(name)
    return sorted(names)


def run_dashboard(results_dir: str | Path | None = None) -> int:
    """Launch the Streamlit dashboard.

    Parameters
    ----------
    results_dir : str | Path | None, optional
        Root directory where run artefacts will be stored. If ``None``, the
        value is read from the email configuration file or defaults to
        :data:`nircls.utils.misc.RESULTS_DIR`.
    """
    if results_dir is None:
        results_dir = _load_email_config().get("results_dir", RESULTS_DIR)
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
    return run_dashboard()


def _render_dashboard(results_dir: Path) -> None:
    """Render Streamlit app to visualize existing results."""
    import streamlit as st

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    st.set_page_config(page_title="nircls Dashboard", page_icon="🧪", layout="wide")
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

    def _run_wizard() -> None:
        st.title("nircls Dashboard")
        st.header("Configuração inicial")
        step = st.session_state.get("wizard_step", 0)
        data = st.session_state.setdefault("wizard_data", {})
        if step == 0:
            with st.form("wiz_modelos"):
                modelos = st.multiselect(
                    "Modelos",
                    options=sorted(MODEL_DESCRIPTIONS),
                    default=data.get("modelos", []),
                )
                submitted = st.form_submit_button("Próximo")
            if submitted:
                data["modelos"] = modelos
                st.session_state["wizard_step"] = 1
                st.experimental_rerun()
        elif step == 1:
            with st.form("wiz_preprocess"):
                preprocess = st.multiselect(
                    "Preprocessadores",
                    options=_get_preprocessor_names(),
                    default=data.get("preprocess", []),
                )
                submitted = st.form_submit_button("Próximo")
            if submitted:
                data["preprocess"] = preprocess
                st.session_state["wizard_step"] = 2
                st.experimental_rerun()
        else:
            with st.form("wiz_email"):
                email = st.text_input(
                    "E-mail",
                    value=data.get("email", ""),
                    help="Endereço para notificações",
                )
                submitted = st.form_submit_button("Concluir")
            if submitted:
                data["email"] = email
                _save_cfg(data)
                st.session_state.pop("wizard_step", None)
                st.session_state.pop("wizard_data", None)
                st.session_state.pop("run_wizard", None)
                st.experimental_rerun()

    if st.session_state.get("run_wizard") or not config_path.exists():
        _run_wizard()
        return

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

    (
        config_tab,
        settings_tab,
        prep_tab,
        exec_tab,
        results_tab,
        summary_tab,
        logs_tab,
        help_tab,
    ) = st.tabs(
        [
            "Configurar & Executar",
            "Configurações",
            "Preparar Dados",
            "Execução",
            "Resultados",
            "Resumo",
            "Logs",
            "Ajuda",
        ]
    )

    with config_tab:
        cfg_defaults = _load_cfg()
        email_defaults = _load_email_config()
        st.session_state.setdefault("modelos", cfg_defaults.get("modelos", []))
        st.session_state.setdefault("preprocess", cfg_defaults.get("preprocess", []))
        st.session_state.setdefault(
            "email",
            cfg_defaults.get("email", email_defaults.get("to_addr", "")),
        )

        if st.button("Reiniciar wizard"):
            if config_path.exists():
                try:
                    config_path.unlink()
                except OSError:
                    pass
            st.session_state["run_wizard"] = True
            st.experimental_rerun()

        ajuda_link = "Consulte a [aba Ajuda](#ajuda)"

        with st.form("run_form"):
            analises = st.multiselect(
                "Análises",
                options=["preparar", "treinar", "validar"],
                help=f"Escolha as análises a executar. {ajuda_link}",
            )
            uploaded_files = st.file_uploader(
                "Arquivos de entrada",
                type=["csv", "xlsx", "xls"],
                accept_multiple_files=True,
                help=f"Envie os dados de entrada. {ajuda_link}",
            )
            modelos = st.multiselect(
                "Modelos",
                options=sorted(MODEL_DESCRIPTIONS),
                key="modelos",
                help=f"Selecione os modelos. {ajuda_link}",
            )
            preprocess = st.multiselect(
                "Preprocessadores",
                options=_get_preprocessor_names(),
                key="preprocess",
                help=f"Selecione os pré-processadores. {ajuda_link}",
            )
            email = st.text_input(
                "E-mail",
                key="email",
                help=f"Endereço para notificações. {ajuda_link}",
            )
            submitted = st.form_submit_button("Executar")

        cols = st.columns(2)
        if cols[0].button("Salvar"):
            _save_cfg(
                {
                    k: st.session_state.get(k, [] if k != "email" else "")
                    for k in ["modelos", "preprocess", "email"]
                }
            )
            st.success("Configurações salvas.")
        if cols[1].button("Carregar"):
            loaded = _load_cfg()
            for k in ["modelos", "preprocess", "email"]:
                st.session_state[k] = loaded.get(k, [] if k != "email" else "")
            st.experimental_rerun()

        if submitted:
            _save_cfg(
                {
                    k: st.session_state.get(k, [] if k != "email" else "")
                    for k in ["modelos", "preprocess", "email"]
                }
            )
            if not analises:
                st.warning("Selecione ao menos uma análise.")
            elif not uploaded_files:
                st.warning("Envie ao menos um arquivo de entrada.")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    for up in uploaded_files:
                        dest = tmpdir_path / up.name
                        dest.write_bytes(up.getbuffer())
                    cfg = {
                        "data": {"input_path": str(tmpdir_path / "*")},
                        "models": [{"name": m} for m in modelos],
                        "preprocess": {"steps": list(preprocess)},
                        "cv": {"strategies": [{"type": "kfold", "n_splits": 5}]},
                        "notification_email": email or None,
                    }
                    try:
                        cfg_model = ConfigModel(**cfg)
                    except ValidationError as exc:
                        st.error(f"Configuração inválida: {exc}")
                    else:
                        cfg_dict = cfg_model.dict(exclude_none=True)
                        run_dir = results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
                        progress = st.progress(0)
                        status = st.status("Inicializando...")
                        errors: dict[str, str] = {}
                        try:
                            steps = list(analises)
                            total = len(steps)
                            completed = 0

                            if "preparar" in steps:
                                status.update(
                                    label="Preparando dados...", state="running"
                                )
                                try:
                                    prepare_inputs(cfg_dict["data"], use_cache=False)
                                except Exception as exc:
                                    errors["preparar"] = str(exc)
                                    raise
                                completed += 1
                                progress.progress(int(completed / total * 100))

                            if "treinar" in steps:
                                status.update(
                                    label="Treinando modelos...", state="running"
                                )
                                try:
                                    train_and_save(cfg_dict, run_dir / "train")
                                except Exception as exc:
                                    errors["treinar"] = str(exc)
                                    raise
                                completed += 1
                                progress.progress(int(completed / total * 100))

                            if "validar" in steps:
                                status.update(
                                    label="Validando modelos...", state="running"
                                )
                                try:
                                    run_multiple_eval(cfg_dict, run_dir / "cv")
                                except Exception as exc:
                                    errors["validar"] = str(exc)
                                    raise
                                completed += 1
                                progress.progress(int(completed / total * 100))

                            status.update(label="Execução concluída.", state="complete")
                            st.success("Execução concluída.")
                            if email:
                                enviar_email(
                                    "Execução concluída",
                                    "A execução foi concluída.",
                                    email,
                                )
                        except Exception:
                            status.update(label="Execução falhou", state="error")
                            for etapa, msg in errors.items():
                                st.error(f"Erro na etapa {etapa}: {msg}")

    with settings_tab:
        email_cfg = _load_email_config()
        host = st.text_input("Host SMTP", email_cfg.get("host", ""))
        port = st.number_input("Porta", value=int(email_cfg.get("port") or 587), step=1)
        user = st.text_input("Usuário", email_cfg.get("user", ""))
        password = st.text_input(
            "Senha", value=email_cfg.get("password", ""), type="password"
        )
        default_email = st.text_input("E-mail padrão", email_cfg.get("to_addr", ""))
        default_results = email_cfg.get("results_dir", str(results_dir))
        if hasattr(st, "directory_picker"):
            results_root = st.directory_picker(
                "Diretório de resultados", default=default_results
            )
        else:
            results_root = st.text_input(
                "Diretório de resultados", value=default_results, key="results_root"
            )
            if st.button("Selecionar pasta"):
                try:  # pragma: no cover - best effort
                    import tkinter as tk
                    from tkinter import filedialog

                    tk.Tk().withdraw()
                    selected = filedialog.askdirectory() or results_root
                    st.session_state["results_root"] = selected
                    results_root = selected
                except Exception:
                    st.warning("Seleção de pasta não suportada.")
            results_root = st.session_state.get("results_root", results_root)

        if st.button("Salvar configurações"):
            _save_email_config(
                {
                    "host": host,
                    "port": int(port),
                    "user": user,
                    "to_addr": default_email,
                    "results_dir": str(results_root),
                }
            )
            if password:
                try:  # pragma: no cover - best effort
                    keyring.set_password("nircls", user, password)
                except Exception:
                    st.warning("Falha ao salvar a senha no keyring.")
            st.success("Configurações salvas.")

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

    with logs_tab:
        log_path = Path.home() / ".nircls" / "nircls.log"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()[-200:]
            st.text_area("Log", "\n".join(lines), height=300)
            with log_path.open("rb") as f:
                st.download_button(
                    "Baixar log", f, file_name=log_path.name, mime="text/plain"
                )
        else:
            st.info(f"Arquivo de log não encontrado em {log_path}")

    with help_tab:
        st.markdown("<a id='ajuda'></a>", unsafe_allow_html=True)
        st.subheader("Modelos disponíveis")
        model_apps = {
            "PLSDA": "Análises espectrais e quimiometria",
            "SVM_RBF": "Classificação não linear",
            "SVM_LINEAR": "Classificação linear de alta dimensionalidade",
            "RF": "Dados tabulares com muitas variáveis",
            "GB": "Classificação em dados estruturados",
            "ET": "Ensembles de árvores para dados complexos",
            "ADABOOST": "Boosting adaptativo",
            "LOGREG": "Classificação binária simples",
            "QDA": "Classes com covariâncias diferentes",
            "XGBOOST": "Boosting gradiente otimizado",
            "LGBM": "Grandes volumes de dados",
            "CATBOOST": "Dados com muitas variáveis categóricas",
            "MLP": "Padrões não lineares",
            "DT": "Modelos interpretáveis",
            "KNN": "Classificação baseada em proximidade",
            "CNN1D": "Sinais espectrais unidimensionais",
            "LSTM1D": "Séries temporais sequenciais",
            "GAUSSIANNB": "Variáveis contínuas com hipótese gaussiana",
            "NB": "Classificação probabilística simples",
            "MULTINB": "Textos ou contagens discretas",
            "BERNOULLINB": "Dados binários",
        }
        help_df = pd.DataFrame(
            [
                (k, v, model_apps.get(k, "-"))
                for k, v in sorted(MODEL_DESCRIPTIONS.items())
            ],
            columns=["Modelo", "Descrição", "Aplicações Típicas"],
        )
        with st.expander("Tabela de modelos"):
            st.dataframe(help_df, hide_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Diretório de saída")
    ns = parser.parse_args()
    _render_dashboard(Path(ns.results))
