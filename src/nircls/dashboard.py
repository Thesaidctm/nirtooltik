from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import argparse
import pandas as pd

def run_dashboard(results_dir: str | Path) -> int:
    """Launch a Streamlit dashboard for exploring experiment results.

    Parameters
    ----------
    results_dir : str | Path
        Root directory where run results are stored.
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

def _render_dashboard(results_dir: Path) -> None:
    """Render the Streamlit application given a results directory."""
    import streamlit as st

    results_dir = Path(results_dir)
    st.set_page_config(page_title="nircls Dashboard")
    st.title("nircls Dashboard")

    runs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not runs:
        st.error(f"Sem diretórios de resultados em {results_dir}")
        return
    run_names = sorted(d.name for d in runs)
    run_name = st.sidebar.selectbox("Execução", run_names)
    run_dir = results_dir / run_name

    strat_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if strat_dirs:
        strat_names = sorted(d.name for d in strat_dirs)
        strat_name = st.sidebar.selectbox("Estratégia", strat_names)
        base_dir = run_dir / strat_name
    else:
        base_dir = run_dir

    st.header("Métricas")
    ms_xlsx = base_dir / "metrics_summary.xlsx"
    ms_csv = base_dir / "metrics_summary.csv"
    if ms_xlsx.exists():
        xls = pd.ExcelFile(ms_xlsx)
        st.subheader("Validação")
        st.dataframe(pd.read_excel(xls, sheet_name="validation"))
        st.subheader("Calibração")
        st.dataframe(pd.read_excel(xls, sheet_name="calibration"))
    elif ms_csv.exists():
        st.dataframe(pd.read_csv(ms_csv))
    else:
        st.write("Nenhum arquivo de métricas encontrado.")

    report = base_dir / "report.md"
    if report.exists():
        st.header("Relatório")
        st.markdown(report.read_text())

    fig_dir = base_dir / "figures"
    if fig_dir.exists():
        st.header("Figuras")
        for img in sorted(fig_dir.glob("*.png")):
            st.image(str(img), caption=img.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Diretório com resultados")
    ns = parser.parse_args()
    _render_dashboard(Path(ns.results))
