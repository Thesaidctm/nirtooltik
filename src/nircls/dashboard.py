from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import argparse
import json
import webbrowser

import pandas as pd
import plotly.io as pio

from .utils.misc import RESULTS_DIR
from .utils.summarize import summarize_runs


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
    return run_dashboard(RESULTS_DIR)


def _render_dashboard(results_dir: Path) -> None:
    """Render Streamlit app to visualize existing results."""
    import streamlit as st

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    st.set_page_config(page_title="nircls Dashboard")
    st.title("nircls Dashboard")
    st.info("Execute `nircls run-all` antes de abrir o dashboard.")

    run_dirs = [
        p
        for p in sorted(results_dir.iterdir())
        if p.is_dir() and any(p.glob("**/metrics.json"))
    ]
    if not run_dirs:
        st.warning(f"Nenhuma execução encontrada em {results_dir}/.")
        return

    def _display_figures(fig_dir: Path) -> None:
        bases = {
            p.with_suffix("")
            for p in fig_dir.glob("*")
            if p.suffix in {".png", ".json"}
        }
        for base in sorted(bases):
            json_file = base.with_suffix(".json")
            png_file = base.with_suffix(".png")
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

    exec_tab, results_tab = st.tabs(["Execução", "Resultados"])

    with exec_tab:
        selected_run = st.selectbox(
            "Selecione uma execução",
            run_dirs,
            format_func=lambda p: p.name,
        )
        for metrics_path in sorted(selected_run.rglob("metrics.json")):
            strat_dir = metrics_path.parent
            st.subheader(strat_dir.relative_to(selected_run).as_posix())
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

    with results_tab:
        summary = summarize_runs(results_dir)
        if summary.empty:
            st.warning("Nenhuma métrica agregada encontrada.")
        else:
            st.dataframe(summary)

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
                                st.dataframe(pd.DataFrame(metrics))
                        except Exception as exc:  # pragma: no cover
                            st.error(f"Erro ao carregar métricas: {exc}")
                        fig_dir = strat_dir / "figures"
                        if fig_dir.exists():
                            _display_figures(fig_dir)
        elif compare_runs:
            st.warning("Selecione ao menos duas execuções para comparar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Diretório de saída")
    ns = parser.parse_args()
    _render_dashboard(Path(ns.results))
