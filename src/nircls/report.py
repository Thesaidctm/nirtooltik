from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from typing import List


def generate_report(metrics_json: Path, figures_dir: Path, out_file: Path) -> None:
    """Generate a simple evaluation report.

    Parameters
    ----------
    metrics_json : Path
        JSON file containing a list of metric dictionaries.
    figures_dir : Path
        Directory where figures (PNG) were saved.
    out_file : Path
        Output path for the generated report. Supports ``.md`` or ``.pdf``.
    """
    metrics_json = Path(metrics_json)
    figures_dir = Path(figures_dir)
    out_file = Path(out_file)

    # Load metrics and build table with mean/std columns
    with open(metrics_json) as f:
        data = json.load(f)
    if not data:
        raise ValueError("metrics_json is empty")
    df = pd.DataFrame(data)

    # create combined columns e.g., "F1-macro" = "mean ± std"
    cols: List[str] = []
    table_df = df[[c for c in ["file", "model"] if c in df.columns]].copy()
    for col in df.columns:
        if col.endswith("_mean"):
            base = col[:-5]
            mean_col = col
            std_col = f"{base}_std"
            if std_col in df.columns:
                table_df[base] = df.apply(
                    lambda r: f"{r[mean_col]:.3f} ± {r[std_col]:.3f}", axis=1
                )
            else:
                table_df[base] = df[mean_col].map(lambda v: f"{v:.3f}")
            cols.append(base)
    # Build a markdown table without requiring external dependencies
    cols = table_df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in table_df.iterrows():
        lines.append("| " + " | ".join(map(str, row.tolist())) + " |")
    table_md = "\n".join(lines)

    # collect figures
    figs = sorted(figures_dir.glob("*.png"))
    fig_lines = []
    for fig in figs:
        rel = f"{figures_dir.name}/{fig.name}"
        fig_lines.append(f"![{fig.stem}]({rel})")
    fig_md = "\n\n".join(fig_lines)

    md = f"# Evaluation Report\n\n## Metrics\n\n{table_md}\n\n## Figures\n\n{fig_md}\n"

    suffix = out_file.suffix.lower()
    if suffix in {".md", ".markdown"}:
        out_file.write_text(md)
        # also export HTML if markdown2 is available
        try:
            import markdown2  # type: ignore

            html = markdown2.markdown(md)
            out_file.with_suffix(".html").write_text(html)
        except Exception:
            pass
    elif suffix == ".pdf":
        try:
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Image as RLImage,
                Table as RLTable,
            )
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import utils
        except Exception as e:
            raise RuntimeError("reportlab is required for PDF export") from e

        doc = SimpleDocTemplate(str(out_file))
        styles = getSampleStyleSheet()
        elements: List = []
        elements.append(Paragraph("Evaluation Report", styles["Title"]))
        elements.append(Paragraph("Metrics", styles["Heading2"]))
        table_data = [table_df.columns.tolist()] + table_df.values.tolist()
        elements.append(RLTable(table_data))
        elements.append(Spacer(1, 12))
        if figs:
            elements.append(Paragraph("Figures", styles["Heading2"]))
            for fig in figs:
                try:
                    img = utils.ImageReader(str(fig))
                    iw, ih = img.getSize()
                    width = 400
                    height = ih * (width / iw)
                    elements.append(RLImage(str(fig), width=width, height=height))
                    elements.append(Spacer(1, 12))
                except Exception:
                    continue
        doc.build(elements)
    else:
        raise ValueError("Unsupported output format. Use .md or .pdf")
