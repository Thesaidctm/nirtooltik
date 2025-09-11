from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import shutil
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
from nircls.i18n import tr
from nircls.io import prepare_inputs, summarize_dataset
from nircls.models.factory import MODEL_DESCRIPTIONS
from nircls.pipeline import run_multiple_eval, train_and_save
from nircls.utils.email import _load_config as _load_email_config
from nircls.utils.email import _save_config as _save_email_config
from nircls.utils.email import enviar_email
from nircls.utils.misc import RESULTS_DIR
from nircls.utils.palettes import Palette
from nircls.utils.plots import _export
from nircls.utils.summarize import summarize_runs
from nircls.utils.update import check_latest_version
from nircls.utils.zip import zip_dir


def _figure_caption(stem: str) -> str:
    """Map figure filename stems to human-friendly captions."""

    suffix = stem.split("__")[-1]
    return {
        "confusion": "Matriz de Confusão",
        "roc": "Curva ROC",
        "reliability": "Diagrama de Confiabilidade",
        "class_counts": "Contagem de classes",
        "mean_std": "Espectro médio ± desvio padrão",
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
    prep_fig_dir = results_dir / "prep_figures"

    st.set_page_config(page_title="nircls Dashboard", page_icon="🧪", layout="wide")
    config_path = Path.home() / ".nircls" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with config_path.open() as f:
            _initial_cfg = json.load(f)
    except Exception:
        _initial_cfg = {}
    st.session_state.setdefault("language", _initial_cfg.get("language", "pt"))
    st.session_state.setdefault(
        "palette", _initial_cfg.get("palette", Palette.VIRIDIS.value)
    )
    if _initial_cfg.get("check_updates", True):
        import nircls

        latest = check_latest_version(nircls.__version__)
        if latest:
            st.info(
                f"Nova versão disponível: {latest}. "
                f"[Baixar](https://github.com/Thesaidctm/nirtooltik/releases/tag/{latest})"
            )

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
        st.title(tr("nircls Dashboard"))
        st.header(tr("Configuração inicial"))
        step = st.session_state.get("wizard_step", 0)
        data = st.session_state.setdefault("wizard_data", {})
        if step == 0:
            with st.form("wiz_modelos"):
                modelos = st.multiselect(
                    tr("Modelos"),
                    options=sorted(MODEL_DESCRIPTIONS),
                    default=data.get("modelos", []),
                )
                submitted = st.form_submit_button(tr("Próximo"))
            if submitted:
                data["modelos"] = modelos
                st.session_state["wizard_step"] = 1
                st.rerun()
        elif step == 1:
            with st.form("wiz_preprocess"):
                preprocess = st.multiselect(
                    tr("Preprocessadores"),
                    options=_get_preprocessor_names(),
                    default=data.get("preprocess", []),
                )
                submitted = st.form_submit_button(tr("Próximo"))
            if submitted:
                data["preprocess"] = preprocess
                st.session_state["wizard_step"] = 2
                st.rerun()
        else:
            with st.form("wiz_email"):
                email = st.text_input(
                    tr("E-mail"),
                    value=data.get("email", ""),
                    help=tr("Endereço para notificações"),
                )
                submitted = st.form_submit_button(tr("Concluir"))
            if submitted:
                data["email"] = email
                _save_cfg(data)
                st.session_state.pop("wizard_step", None)
                st.session_state.pop("wizard_data", None)
                st.session_state.pop("run_wizard", None)
                st.rerun()

    if st.session_state.get("run_wizard") or not config_path.exists():
        _run_wizard()
        return

    st.title(tr("nircls Dashboard"))
    st.info(tr("Execute `nircls run-all` antes de abrir o dashboard."))

    run_dirs = [
        p
        for p in sorted(results_dir.iterdir())
        if p.is_dir() and any(p.glob("**/metrics.json"))
    ]

    def _display_figures(fig_dir: Path, prep_dir: Path | None = None) -> None:
        dirs = [fig_dir]
        if prep_dir is not None and prep_dir.exists() and prep_dir != fig_dir:
            dirs.insert(0, prep_dir)
        for current in dirs:
            bases = {
                p.with_suffix("")
                for p in current.glob("*")
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

                        components.html(
                            html_file.read_text(), height=600, scrolling=True
                        )
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

    def _render_actions(run_dirs: list[Path]) -> None:
        for run in run_dirs:
            name_col, new_col, ren_col, del_col, zip_col = st.columns([3, 3, 1, 1, 1])
            name_col.write(run.name)
            new_name = new_col.text_input(
                "Novo nome", value=run.name, key=f"rename_input_{run.name}"
            )
            if ren_col.button("Renomear", key=f"rename_btn_{run.name}"):
                new_path = run.with_name(new_name.strip())
                if new_name and new_path != run:
                    run.rename(new_path)
                    st.rerun()
            if del_col.button("Excluir", key=f"delete_{run.name}"):
                if st.confirm_dialog(f"Excluir {run.name}?"):
                    shutil.rmtree(run)
                    st.rerun()
            zip_buffer = zip_dir(run)
            zip_col.download_button(
                "ZIP",
                zip_buffer,
                file_name=f"{run.name}.zip",
                mime="application/zip",
                key=f"zip_{run.name}",
            )

    def _read_log_updates(path: Path, state_key: str = "log_pos") -> list[str]:
        """Read new log lines incrementally.

        Parameters
        ----------
        path:
            Location of the log file.
        state_key:
            ``st.session_state`` key used to track the last read position.

        Returns
        -------
        list[str]
            Newly read lines from ``path``.
        """

        pos = st.session_state.get(state_key, 0)
        try:
            with path.open("r", encoding="utf-8") as f:
                f.seek(pos)
                data = f.read()
                st.session_state[state_key] = f.tell()
        except FileNotFoundError:
            st.session_state[state_key] = 0
            data = ""
        return data.splitlines()

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
            tr("Configurar & Executar"),
            tr("Configurações"),
            tr("Preparar Dados"),
            tr("Execução"),
            tr("Resultados"),
            tr("Resumo"),
            tr("Logs"),
            tr("Ajuda"),
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
        st.session_state.setdefault("encoding", cfg_defaults.get("encoding", ""))
        st.session_state.setdefault("sep", cfg_defaults.get("sep", ","))
        st.session_state.setdefault("decimal", cfg_defaults.get("decimal", "."))
        st.session_state.setdefault(
            "palette", cfg_defaults.get("palette", Palette.VIRIDIS.value)
        )

        language = st.selectbox(
            tr("Idioma"),
            ["pt", "en"],
            index=["pt", "en"].index(st.session_state.get("language", "pt")),
        )
        if language != st.session_state.get("language"):
            st.session_state["language"] = language
            cfg = _load_cfg()
            cfg["language"] = language
            _save_cfg(cfg)
            st.rerun()

        palette = st.selectbox(
            tr("Paleta de cores"),
            [p.value for p in Palette],
            index=[p.value for p in Palette].index(
                st.session_state.get("palette", Palette.VIRIDIS.value)
            ),
        )
        if palette != st.session_state.get("palette"):
            st.session_state["palette"] = palette

        if st.button(tr("Reiniciar wizard")):
            if config_path.exists():
                try:
                    config_path.unlink()
                except OSError:
                    pass
            st.session_state["run_wizard"] = True
            st.rerun()

        ajuda_link = "Consulte a [aba Ajuda](#ajuda)"

        with st.form("run_form"):
            analises = st.multiselect(
                tr("Análises"),
                options=["preparar", "treinar", "validar"],
                help=tr("Escolha as análises a executar. {ajuda_link}").format(
                    ajuda_link=ajuda_link
                ),
            )
            uploaded_files = st.file_uploader(
                tr("Arquivos de entrada"),
                type=["csv", "xlsx", "xls"],
                accept_multiple_files=True,
                help=tr("Envie os dados de entrada. {ajuda_link}").format(
                    ajuda_link=ajuda_link
                ),
            )
            modelos = st.multiselect(
                tr("Modelos"),
                options=sorted(MODEL_DESCRIPTIONS),
                key="modelos",
                help=tr("Selecione os modelos. {ajuda_link}").format(
                    ajuda_link=ajuda_link
                ),
            )
            preprocess = st.multiselect(
                tr("Preprocessadores"),
                options=_get_preprocessor_names(),
                key="preprocess",
                help=tr("Selecione os pré-processadores. {ajuda_link}").format(
                    ajuda_link=ajuda_link
                ),
            )
            encoding = st.text_input(
                "Codificação",
                key="encoding",
                help=f"Codificação dos arquivos CSV. {ajuda_link}",
            )
            sep = st.text_input(
                "Separador",
                key="sep",
                help=f"Separador de campos do CSV. {ajuda_link}",
            )
            decimal = st.text_input(
                "Separador decimal",
                key="decimal",
                help=f"Caractere usado como separador decimal. {ajuda_link}",
            )
            email = st.text_input(
                tr("E-mail"),
                key="email",
                help=tr("Endereço para notificações. {ajuda_link}").format(
                    ajuda_link=ajuda_link
                ),
            )
            submitted = st.form_submit_button(tr("Executar"))

        cols = st.columns(2)
        if cols[0].button(tr("Salvar")):
            cfg = _load_cfg()
            cfg.update(
                {
                    "modelos": st.session_state.get("modelos", []),
                    "preprocess": st.session_state.get("preprocess", []),
                    "email": st.session_state.get("email", ""),
                    "encoding": st.session_state.get("encoding", ""),
                    "sep": st.session_state.get("sep", ","),
                    "decimal": st.session_state.get("decimal", "."),
                    "palette": st.session_state.get("palette", Palette.VIRIDIS.value),
                }
            )
            _save_cfg(cfg)
            st.success(tr("Configurações salvas."))
        if cols[1].button(tr("Carregar")):
            loaded = _load_cfg()
            for k, default in [
                ("modelos", []),
                ("preprocess", []),
                ("email", ""),
                ("encoding", ""),
                ("sep", ","),
                ("decimal", "."),
                ("palette", Palette.VIRIDIS.value),
            ]:
                st.session_state[k] = loaded.get(k, default)
            st.rerun()

        if submitted:
            cfg = _load_cfg()
            cfg.update(
                {
                    "modelos": st.session_state.get("modelos", []),
                    "preprocess": st.session_state.get("preprocess", []),
                    "email": st.session_state.get("email", ""),
                    "encoding": st.session_state.get("encoding", ""),
                    "sep": st.session_state.get("sep", ","),
                    "decimal": st.session_state.get("decimal", "."),
                    "palette": st.session_state.get("palette", Palette.VIRIDIS.value),
                }
            )
            _save_cfg(cfg)
            if not analises:
                st.warning(tr("Selecione ao menos uma análise."))
            elif not uploaded_files:
                st.warning(tr("Envie ao menos um arquivo de entrada."))
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    for up in uploaded_files:
                        dest = tmpdir_path / up.name
                        dest.write_bytes(up.getbuffer())
                    data_cfg = {"input_path": str(tmpdir_path / "*")}
                    if encoding:
                        data_cfg["encoding"] = encoding
                    if sep:
                        data_cfg["sep"] = sep
                    if decimal:
                        data_cfg["decimal"] = decimal
                    cfg = {
                        "data": data_cfg,
                        "models": [{"name": m} for m in modelos],
                        "preprocess": {"steps": list(preprocess)},
                        "cv": {"strategies": [{"type": "kfold", "n_splits": 5}]},
                        "notification_email": email or None,
                        "palette": st.session_state.get(
                            "palette", Palette.VIRIDIS.value
                        ),
                    }
                    try:
                        cfg_model = ConfigModel(**cfg)
                    except ValidationError as exc:
                        st.error(tr("Configuração inválida: {exc}").format(exc=exc))
                    else:
                        cfg_dict = cfg_model.dict(exclude_none=True)
                        run_dir = results_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
                        progress = st.progress(0)
                        status = st.status(tr("Inicializando..."))
                        errors: dict[str, str] = {}
                        try:
                            steps = list(analises)
                            total = len(steps)
                            completed = 0

                            if "preparar" in steps:
                                status.update(
                                    label=tr("Preparando dados..."), state="running"
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
                                    label=tr("Treinando modelos..."), state="running"
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
                                    label=tr("Validando modelos..."), state="running"
                                )
                                try:
                                    run_multiple_eval(cfg_dict, run_dir / "cv")
                                except Exception as exc:
                                    errors["validar"] = str(exc)
                                    raise
                                completed += 1
                                progress.progress(int(completed / total * 100))

                            status.update(
                                label=tr("Execução concluída."), state="complete"
                            )
                            st.success(tr("Execução concluída."))
                            if email:
                                enviar_email(
                                    tr("Execução concluída."),
                                    tr("A execução foi concluída."),
                                    email,
                                )
                        except Exception:
                            status.update(label=tr("Execução falhou"), state="error")
                            for etapa, msg in errors.items():
                                st.error(
                                    tr("Erro na etapa {etapa}: {msg}").format(
                                        etapa=etapa, msg=msg
                                    )
                                )

    with settings_tab:
        email_cfg = _load_email_config()
        cfg = _load_cfg()
        check_updates = st.checkbox(
            tr("Verificar atualizações automaticamente"),
            cfg.get("check_updates", True),
        )
        host = st.text_input(tr("Host SMTP"), email_cfg.get("host", ""))
        port = st.number_input(
            tr("Porta"), value=int(email_cfg.get("port") or 587), step=1
        )
        user = st.text_input(tr("Usuário"), email_cfg.get("user", ""))
        password = st.text_input(
            tr("Senha"), value=email_cfg.get("password", ""), type="password"
        )
        default_email = st.text_input(tr("E-mail padrão"), email_cfg.get("to_addr", ""))
        default_results = email_cfg.get("results_dir", str(results_dir))
        refresh_default = float(email_cfg.get("log_refresh_secs", 0))
        st.number_input(
            tr("Frequência de atualização dos logs (s)"),
            value=refresh_default,
            min_value=0.0,
            step=1.0,
            key="log_refresh_secs",
            help=tr("0 desativa atualização automática."),
        )
        if hasattr(st, "directory_picker"):
            results_root = st.directory_picker(
                tr("Diretório de resultados"), default=default_results
            )
        else:
            results_root = st.text_input(
                tr("Diretório de resultados"), value=default_results, key="results_root"
            )
            if st.button(tr("Selecionar pasta")):
                try:  # pragma: no cover - best effort
                    import tkinter as tk
                    from tkinter import filedialog

                    tk.Tk().withdraw()
                    selected = filedialog.askdirectory() or results_root
                    st.session_state["results_root"] = selected
                    results_root = selected
                except Exception:
                    st.warning(tr("Seleção de pasta não suportada."))
            results_root = st.session_state.get("results_root", results_root)

        if st.button(tr("Salvar configurações")):
            _save_email_config(
                {
                    "host": host,
                    "port": int(port),
                    "user": user,
                    "to_addr": default_email,
                    "results_dir": str(results_root),
                    "log_refresh_secs": float(
                        st.session_state.get("log_refresh_secs", 0)
                    ),
                }
            )
            cfg["check_updates"] = check_updates
            _save_cfg(cfg)
            if password:
                try:  # pragma: no cover - best effort
                    keyring.set_password("nircls", user, password)
                except Exception:
                    st.warning(tr("Falha ao salvar a senha no keyring."))
            st.success(tr("Configurações salvas."))

    with prep_tab:
        if prep_fig_dir.exists():
            _display_figures(prep_fig_dir)
        uploaded = st.file_uploader(
            tr("Arquivo CSV ou XLSX"),
            type=["csv", "xlsx", "xls"],
        )
        if uploaded is not None:
            enc = st.text_input("Codificação", st.session_state.get("encoding", ""))
            sep = st.text_input("Separador", st.session_state.get("sep", ","))
            decimal = st.text_input(
                "Separador decimal", st.session_state.get("decimal", ".")
            )
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(
                    uploaded,
                    encoding=enc or None,
                    sep=sep,
                    decimal=decimal,
                    nrows=100,
                )
                sheet = None
            else:
                xls = pd.ExcelFile(uploaded)
                sheet = st.selectbox(tr("Planilha"), xls.sheet_names)
                df = pd.read_excel(xls, sheet_name=sheet, decimal=decimal, nrows=100)
            uploaded.seek(0)
            st.subheader(tr("Pré-visualização"))
            st.dataframe(df)
            summary_cols = pd.DataFrame(
                {"coluna": df.columns, "tipo": df.dtypes.astype(str)}
            )
            st.subheader(tr("Resumo de colunas"))
            st.dataframe(summary_cols)
            is_num = pd.api.types.is_numeric_dtype
            non_num = [c for c in df.columns if not is_num(df[c])]
            target_col = st.selectbox(tr("Coluna alvo"), non_num)
            group_col = st.selectbox(tr("Coluna de grupos"), [""] + non_num)
            spectral_input = st.text_input(tr("Faixa espectral (mín,max)"))
            if target_col and st.button(tr("Processar")):
                max_size = 50 * 1024 * 1024
                if uploaded.size > max_size:
                    st.error(tr("Arquivo maior que 50 MB"))
                else:
                    progress = st.progress(0)
                    with tempfile.NamedTemporaryFile(
                        "wb", delete=False, suffix=Path(uploaded.name).suffix
                    ) as tmp:
                        uploaded.seek(0)
                        total = uploaded.size or 1
                        chunk = 1 << 20
                        written = 0
                        while True:
                            data = uploaded.read(chunk)
                            if not data:
                                break
                            tmp.write(data)
                            written += len(data)
                            progress.progress(int(written / total * 100))
                        tmp_path = Path(tmp.name)
                    cfg = {
                        "input_path": str(tmp_path),
                        "target_col": target_col,
                        "encoding": enc or None,
                        "sep": sep,
                        "decimal": decimal,
                    }
                    if sheet is not None:
                        cfg["sheet_name"] = sheet
                    if group_col:
                        cfg["group_col"] = group_col
                    if spectral_input:
                        try:
                            lo, hi = map(float, spectral_input.split(","))
                            cfg["spectral_range"] = [lo, hi]
                        except Exception:
                            st.error(tr("Formato inválido de faixa espectral"))
                    try:
                        ds_list = prepare_inputs(cfg, use_cache=False)
                        summary = summarize_dataset(ds_list)
                        st.subheader(tr("Resumo do conjunto"))
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
                            stem = ds["target"].replace(" ", "_")
                            out_base = prep_fig_dir / f"{stem}__class_counts"
                            _export(fig, out_base, interactive=True)
                            png_file = out_base.with_suffix(".png")
                            with png_file.open("rb") as f:
                                st.download_button(
                                    f"Download {png_file.name}",
                                    f,
                                    file_name=png_file.name,
                                    mime="image/png",
                                )
                            html_file = out_base.with_suffix(".html")
                            if html_file.exists():
                                with html_file.open("r") as f:
                                    st.download_button(
                                        f"Download {html_file.name}",
                                        f,
                                        file_name=html_file.name,
                                        mime="text/html",
                                    )
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
                            out_base2 = prep_fig_dir / f"{stem}__mean_std"
                            _export(fig2, out_base2, interactive=True)
                            png_file = out_base2.with_suffix(".png")
                            with png_file.open("rb") as f:
                                st.download_button(
                                    f"Download {png_file.name}",
                                    f,
                                    file_name=png_file.name,
                                    mime="image/png",
                                )
                            html_file = out_base2.with_suffix(".html")
                            if html_file.exists():
                                with html_file.open("r") as f:
                                    st.download_button(
                                        f"Download {html_file.name}",
                                        f,
                                        file_name=html_file.name,
                                        mime="text/html",
                                    )
                    except Exception as exc:  # pragma: no cover - display error
                        st.error(tr("Erro ao preparar dados: {exc}").format(exc=exc))
                    finally:
                        tmp_path.unlink(missing_ok=True)

    with exec_tab:
        if not run_dirs:
            st.warning(
                tr("Nenhuma execução encontrada em {results_dir}/.").format(
                    results_dir=results_dir
                )
            )
        else:
            selected = st.multiselect(
                tr("Selecione execuções"),
                run_dirs,
                format_func=lambda p: p.name,
            )
            if not selected:
                st.info(tr("Selecione uma ou mais execuções."))
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
                                st.error(
                                    tr("Erro ao carregar métricas: {exc}").format(
                                        exc=exc
                                    )
                                )
                            fig_dir = strat_dir / "figures"
                            if fig_dir.exists():
                                _display_figures(fig_dir, prep_fig_dir)
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
                                    st.error(
                                        tr(
                                            "Erro ao carregar relatório QC: {exc}"
                                        ).format(exc=exc)
                                    )

    with results_tab:
        if not run_dirs:
            st.warning(
                tr("Nenhuma execução encontrada em {results_dir}/.").format(
                    results_dir=results_dir
                )
            )
        else:
            _render_actions(run_dirs)
            compare_runs = st.multiselect(
                tr("Selecione execuções para comparar"),
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
                                st.error(
                                    tr("Erro ao carregar métricas: {exc}").format(
                                        exc=exc
                                    )
                                )
                            fig_dir = strat_dir / "figures"
                            if fig_dir.exists():
                                _display_figures(fig_dir, prep_fig_dir)
            elif compare_runs:
                st.warning(tr("Selecione ao menos duas execuções para comparar."))

    with summary_tab:
        if not run_dirs:
            st.warning(
                tr("Nenhuma execução encontrada em {results_dir}/.").format(
                    results_dir=results_dir
                )
            )
        else:
            _render_actions(run_dirs)
            summary = summarize_runs(results_dir)
            if summary.empty:
                st.warning(tr("Nenhuma métrica agregada encontrada."))
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
        refresh_secs = st.session_state.get("log_refresh_secs", 0.0)
        placeholder = st.empty()
        try:
            from streamlit_autorefresh import st_autorefresh

            if refresh_secs > 0:
                st_autorefresh(interval=int(refresh_secs * 1000), key="log_refresh")
            else:
                st.button(tr("Atualizar"))
        except Exception:
            st.button(tr("Atualizar"))

        if log_path.exists():
            buffer = st.session_state.setdefault("log_buffer", [])
            buffer.extend(_read_log_updates(log_path))
            st.session_state["log_buffer"] = buffer[-200:]
            placeholder.text_area(
                tr("Log"), "\n".join(st.session_state["log_buffer"]), height=300
            )
            with log_path.open("rb") as f:
                st.download_button(
                    tr("Baixar log"), f, file_name=log_path.name, mime="text/plain"
                )
        else:
            placeholder.info(
                tr("Arquivo de log não encontrado em {log_path}").format(
                    log_path=log_path
                )
            )

    with help_tab:
        st.markdown("<a id='ajuda'></a>", unsafe_allow_html=True)
        st.subheader(tr("Gerenciamento de execuções"))
        st.markdown(
            tr("Nas abas **Resultados** e **Resumo**, utilize a coluna de ações para ")
            + tr(
                "renomear, excluir (com confirmação) ou baixar um arquivo ZIP de cada execução."
            )
        )
        st.subheader(tr("Modelos disponíveis"))
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
        with st.expander(tr("Tabela de modelos")):
            st.dataframe(help_df, hide_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Diretório de saída")
    ns = parser.parse_args()
    _render_dashboard(Path(ns.results))
