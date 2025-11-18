from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

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
            ci_low_col = f"{base}_ci95_low"
            ci_high_col = f"{base}_ci95_high"

            def _format_row(row: pd.Series) -> str:
                mean_val = row.get(mean_col)
                if pd.isnull(mean_val):
                    return "N/A"
                text = f"{mean_val:.3f}"
                if std_col in df.columns:
                    std_val = row.get(std_col)
                    if pd.notnull(std_val):
                        text += f" ± {std_val:.3f}"
                if (
                    ci_low_col in df.columns
                    and ci_high_col in df.columns
                    and pd.notnull(row.get(ci_low_col))
                    and pd.notnull(row.get(ci_high_col))
                ):
                    text += (
                        f" (95% CI: {row[ci_low_col]:.3f}–{row[ci_high_col]:.3f})"
                    )
                return text

            table_df[base] = df.apply(_format_row, axis=1)
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

    metric_notes = textwrap.dedent(
        """
        ### Metric definitions
        - **RMSECV**: root mean squared error computed directly from all cross-validation predictions.
        - **RMSEP**: root mean squared error on prediction/hold-out samples.
        - **RPD**: residual predictive deviation (standard deviation of the reference values divided by RMSEP).
        - **MAE_percent**: mean absolute error normalised by the mean absolute value of the reference target (×100).
        """
    ).strip()

    md = (
        "# Evaluation Report\n\n"
        "## Metrics\n\n"
        f"{table_md}\n\n{metric_notes}\n\n"
        "## Figures\n\n"
        f"{fig_md}\n"
    )

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


def generate_scientific_report(metrics: dict, cfg: dict, out_dir: Path) -> Path:
    """Build a concise scientific report in Markdown.

    The function selects the best model based on F1 score, accuracy or R², then
    summarises the dataset, preprocessing steps and tested algorithms. A table
    with the main metrics is included as well as links to figures stored in the
    ``figures`` subdirectory. The report is written to ``report.md`` and, when
    optional PDF tooling (``pandoc`` or ``weasyprint``/``pdfkit``) is available,
    also exported to ``report.pdf``.

    Parameters
    ----------
    metrics:
        Dictionary or list of dictionaries with metrics as produced by the
        training/validation routines.
    cfg:
        Experiment configuration dictionary used to run the pipeline.
    out_dir:
        Directory where ``report.md`` (and optionally ``report.pdf``) will be
        written.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = metrics if isinstance(metrics, list) else [metrics]
    if not rows:
        raise ValueError("metrics is empty")
    df = pd.DataFrame(rows)

    preprocess_steps: list[str] = []
    if "preprocess_steps" in df.columns:
        step_values: set[str] = set()

        def _iter_steps(value: object) -> list[object]:
            if isinstance(value, (list, tuple, set)):
                items: list[object] = []
                for item in value:
                    items.extend(_iter_steps(item))
                return items
            return [value]

        for value in df["preprocess_steps"]:
            for step in _iter_steps(value):
                if step is None:
                    continue
                try:
                    if pd.isna(step):
                        continue
                except TypeError:
                    pass
                step_str = str(step).strip()
                if not step_str:
                    continue
                step_values.add(step_str)

        preprocess_steps = sorted(step_values)

    # Determine the key metric and best model
    key_metric = None
    for cand in ["F1-macro_mean", "Accuracy_mean", "R2_mean"]:
        if cand in df.columns:
            key_metric = cand
            break
    best_model = None
    best_score = None
    best_steps: list[str] | None = None
    if key_metric is not None:
        idx = df[key_metric].astype(float).idxmax()
        best_row = df.loc[idx]
        best_model = best_row.get("model")
        best_score = float(best_row[key_metric])
        if "preprocess_steps" in best_row:
            val = best_row["preprocess_steps"]
            if isinstance(val, (list, tuple, set)):
                best_steps = [str(item) for item in val]
            else:
                try:
                    not_null = pd.notna(val)
                except TypeError:
                    not_null = True
                if not_null:
                    best_steps = [str(val)]

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

    models = (
        [m.get("name", "") for m in cfg.get("models", [])]
        if isinstance(cfg, dict)
        else []
    )

    lines = [
        "---",
        (
            f'title: "{cfg.get("title", "Scientific Report")}"'
            if isinstance(cfg, dict)
            else 'title: "Scientific Report"'
        ),
        f"date: {dt.date.today().isoformat()}",
        "---",
        "",
        f"**Data**: `{data_cfg.get('input_path', '')}`",
        f"**Preprocessing**: {', '.join(preprocess_steps) if preprocess_steps else 'None'}",
        f"**Algorithms tested**: {', '.join(models) if models else 'N/A'}",
    ]
    if best_model is not None and key_metric is not None and best_score is not None:
        metric_name = key_metric.replace("_mean", "")
        steps_str = f" ({', '.join(best_steps)})" if best_steps else ""
        lines.append(
            f"**Best model**: {best_model}{steps_str} ({metric_name}: {best_score:.3f})"
        )
    if "overfit_warning" in df.columns and df["overfit_warning"].any():
        flagged = df.loc[df["overfit_warning"], "model"].astype(str).tolist()
        lines.append(
            "**Warning**: potential overfitting detected for models: "
            + ", ".join(flagged)
        )

    lines.append("")
    summary_md = "\n".join(lines)

    # Metrics table
    table_cols = [
        c
        for c in ["file", "model", "preprocess_steps", "overfit_warning"]
        if c in df.columns
    ]
    table_df = df[table_cols].copy()
    if "preprocess_steps" in table_df.columns:
        table_df["preprocess_steps"] = table_df["preprocess_steps"].apply(
            lambda v: ", ".join(v) if isinstance(v, (list, tuple)) else str(v)
        )
    for col in df.columns:
        if col.endswith("_mean"):
            base = col[:-5]
            mean_col = col
            std_col = f"{base}_std"
            ci_low_col = f"{base}_ci95_low"
            ci_high_col = f"{base}_ci95_high"

            def _format_row(row: pd.Series) -> str:
                mean_val = row.get(mean_col)
                if pd.isnull(mean_val):
                    return "N/A"
                text = f"{mean_val:.3f}"
                if std_col in df.columns:
                    std_val = row.get(std_col)
                    if pd.notnull(std_val):
                        text += f" ± {std_val:.3f}"
                if (
                    ci_low_col in df.columns
                    and ci_high_col in df.columns
                    and pd.notnull(row.get(ci_low_col))
                    and pd.notnull(row.get(ci_high_col))
                ):
                    text += (
                        f" (95% CI: {row[ci_low_col]:.3f}–{row[ci_high_col]:.3f})"
                    )
                return text

            table_df[base] = df.apply(_format_row, axis=1)
    header = "| " + " | ".join(table_df.columns.tolist()) + " |"
    sep = "| " + " | ".join(["---"] * len(table_df.columns)) + " |"
    lines = [header, sep]
    for _, row in table_df.iterrows():
        lines.append("| " + " | ".join(map(str, row.tolist())) + " |")
    table_md = "\n".join(lines)

    # Figures
    fig_dir = out_dir / "figures"
    fig_lines = []
    if fig_dir.exists():
        for fig in sorted(fig_dir.glob("*.png")):
            rel = f"figures/{fig.name}"
            fig_lines.append(f"![{fig.stem}]({rel})")
    fig_md = "\n\n".join(fig_lines)

    metric_notes = textwrap.dedent(
        """
        ### Metric definitions
        - **RMSECV**: root mean squared error computed directly from all cross-validation predictions.
        - **RMSEP**: root mean squared error on prediction/hold-out samples.
        - **RPD**: residual predictive deviation (standard deviation of the reference values divided by RMSEP).
        - **MAE_percent**: mean absolute error normalised by the mean absolute value of the reference target (×100).
        """
    ).strip()

    md = summary_md + f"## Metrics\n\n{table_md}\n\n{metric_notes}\n\n## Figures\n\n{fig_md}\n"

    report_md = out_dir / "report.md"
    report_md.write_text(md)

    # Optional PDF export
    pdf_path = out_dir / "report.pdf"
    try:  # pragma: no cover - optional dependency
        if shutil.which("pandoc"):
            subprocess.run(
                [
                    "pandoc",
                    str(report_md),
                    "-o",
                    str(pdf_path),
                ],
                check=True,
            )
        else:
            markdown_to_pdf(md, pdf_path)
    except Exception:
        pass

    return report_md
