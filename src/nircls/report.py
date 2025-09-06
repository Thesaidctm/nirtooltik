from __future__ import annotations
from pathlib import Path
import json
import os
from typing import List

import pandas as pd

try:  # pragma: no cover - optional dependency
    import openai
except Exception:  # pragma: no cover - import failure handled later
    openai = None


def generate_ai_summary(metrics_path: Path, model: str) -> str:
    """Use OpenAI's ChatGPT to generate a text summary of the metrics.

    Parameters
    ----------
    metrics_path : Path
        JSON file with metric dictionaries.
    model : str
        ChatGPT model identifier (e.g., ``"gpt-3.5-turbo"``).

    Returns
    -------
    str
        Summary text produced by the model.

    Notes
    -----
    The function expects an API key to be provided via the ``OPENAI_API_KEY``
    environment variable. If the key or the ``openai`` package is missing, a
    ``RuntimeError`` is raised.
    """

    metrics_path = Path(metrics_path)
    with metrics_path.open() as f:
        metrics = json.load(f)

    if openai is None:
        raise RuntimeError("openai package is required for AI summary generation")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    openai.api_key = api_key

    prompt = (
        "Summarize the following evaluation metrics in a concise paragraph:"
        f"\n{json.dumps(metrics, indent=2)}"
    )

    try:  # pragma: no cover - external API call
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant summarizing model evaluation results.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:  # pragma: no cover - propagate error
        raise RuntimeError("Failed to generate AI summary") from e

    return response["choices"][0]["message"]["content"].strip()


def generate_report(
    metrics_json: Path,
    figures_dir: Path,
    out_file: Path,
    *,
    use_chatgpt: bool = False,
    model: str = "gpt-3.5-turbo",
    api_key: str | None = None,
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
    use_chatgpt : bool, optional
        Whether to generate an AI summary using ``generate_ai_summary``.
    model : str, optional
        ChatGPT model to use when ``use_chatgpt`` is ``True``.
    api_key : str, optional
        OpenAI API key. Falls back to ``OPENAI_API_KEY`` environment variable.
    """
    metrics_json = Path(metrics_json)
    figures_dir = Path(figures_dir)
    out_file = Path(out_file)

    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)

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

    # collect figures
    figs = sorted(figures_dir.glob("*.png"))
    fig_lines = []
    for fig in figs:
        rel = f"{figures_dir.name}/{fig.name}"
        fig_lines.append(f"![{fig.stem}]({rel})")
    fig_md = "\n\n".join(fig_lines)

    ai_md = ""
    if use_chatgpt:
        try:
            ai_text = generate_ai_summary(metrics_json, model)
            ai_md = f"## Resumo AI\n\n{ai_text}\n\n"
        except Exception:
            ai_md = ""

    md = f"# Evaluation Report\n\n{ai_md}## Metrics\n\n{table_md}\n\n## Figures\n\n{fig_md}\n"

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
        if ai_md:
            elements.append(Paragraph("Resumo AI", styles["Heading2"]))
            elements.append(Paragraph(ai_text, styles["BodyText"]))
            elements.append(Spacer(1, 12))
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
