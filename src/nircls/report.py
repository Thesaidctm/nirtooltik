from __future__ import annotations
from pathlib import Path
import json

import pandas as pd


def markdown_to_pdf(md_text: str, out_file: Path) -> None:
    """Convert Markdown text to a PDF file.

    The function first converts the markdown to HTML using the ``markdown``
    package and then tries to render the HTML to PDF using ``weasyprint``. If
    ``weasyprint`` is unavailable, it falls back to ``pdfkit``, which requires
    the external ``wkhtmltopdf`` binary.
    """

    try:
        from markdown import markdown
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("markdown package is required for PDF export") from e

    html = markdown(md_text)

    try:  # pragma: no cover - optional dependency
        from weasyprint import HTML  # type: ignore

        HTML(string=html).write_pdf(str(out_file))
    except Exception:
        try:
            import pdfkit  # type: ignore

            pdfkit.from_string(html, str(out_file))
        except Exception as e:
            raise RuntimeError(
                "Install weasyprint or pdfkit with wkhtmltopdf for PDF export"
            ) from e


def generate_report(
    metrics_json: Path,
    figures_dir: Path,
    out_file: Path,
) -> None:
    """Generate a simple evaluation report.

    Parameters
    ----------
    metrics_json : Path
        JSON file containing a list of metric dictionaries.
    figures_dir : Path
        Directory where figures (PNG) were saved.
    out_file : Path
        Output path for the generated report. Supports ``.md`` or ``.pdf``.

    Notes
    -----
    When ``out_file`` ends with ``.pdf`` the markdown content is converted to HTML
    and then rendered to PDF using :func:`markdown_to_pdf`. This requires the
    ``markdown`` package and either ``weasyprint`` (recommended) or ``pdfkit``
    with the ``wkhtmltopdf`` binary installed.
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
    # Build a markdown table without requiring external dependencies
    header = "| " + " | ".join(table_df.columns.tolist()) + " |"
    sep = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    lines = [header, sep]
    for _, row in table_df.iterrows():
        lines.append("| " + " | ".join(map(str, row.tolist())) + " |")
    table_md = "\n".join(lines)

    # collect figures (static images with optional interactive HTML)
    figs = sorted(figures_dir.glob("*.png"))
    fig_lines = []
    for fig in figs:
        rel = f"{figures_dir.name}/{fig.name}"
        fig_lines.append(f"![{fig.stem}]({rel})")
        html = fig.with_suffix(".html")
        if html.exists():
            rel_html = f"{figures_dir.name}/{html.name}"
            fig_lines.append(
                f'<iframe src="{rel_html}" height="500" width="700"></iframe>'
            )
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
            from markdown import markdown as md_to_html
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("markdown package is required for PDF export") from e

        html = md_to_html(md)
        out_file.with_suffix(".html").write_text(html)
        markdown_to_pdf(md, out_file)
    else:
        raise ValueError("Unsupported output format. Use .md or .pdf")
