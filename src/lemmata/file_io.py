"""File reading and export for Lemmata.

Reads txt/pdf/odt/docx/epub uploads, validates sizes, generates ZIP/PDF
exports, and serialises figures to PNG/SVG bytes.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import platform
import sys
import zipfile
from datetime import datetime, timezone
from typing import Any

from lemmata.config import (
    ALLOWED_EXTENSIONS,
    APP_VERSION,
    CHART_DPI,
    MAX_FILE_SIZE_MB,
    MAX_TOTAL_SIZE_MB,
    ZIP_NAME_TEMPLATE,
)
from lemmata.preprocessing import detect_encoding

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


# ═══════════════════════════════════════════════════════════════════════════════
# FILE READING
# ═══════════════════════════════════════════════════════════════════════════════


class FileReadError(Exception):
    """Raised when a file cannot be read, with a user-friendly message."""


def read_file(file_bytes: bytes, filename: str) -> dict[str, str]:
    """Read a single uploaded file and return its text content.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded file.
    filename:
        Original filename (used for extension detection and labelling).

    Returns
    -------
    dict
        ``{"filename": str, "content": str}``.

    Raises
    ------
    FileReadError
        With a specific, user-facing message (decision 162).
    """
    ext = _get_extension(filename)

    if ext not in ALLOWED_EXTENSIONS:
        raise FileReadError(
            f"Unsupported file type '.{ext}'. "
            f"Accepted formats: {', '.join(ALLOWED_EXTENSIONS)}."
        )

    size_mb = len(file_bytes) / _MB
    if size_mb > MAX_FILE_SIZE_MB:
        raise FileReadError(
            f"File '{filename}' is {size_mb:.1f} MB, "
            f"exceeding the {MAX_FILE_SIZE_MB} MB limit."
        )

    readers = {
        "txt": _read_txt,
        "pdf": _read_pdf,
        "docx": _read_docx,
        "odt": _read_odt,
        "epub": _read_epub,
    }
    content = readers[ext](file_bytes, filename)
    return {"filename": filename, "content": content}


def read_files(
    uploaded_files: list[Any],
) -> tuple[list[dict[str, str]], list[str]]:
    """Read multiple uploaded files with size and duplicate checks.

    Parameters
    ----------
    uploaded_files:
        Streamlit ``UploadedFile`` objects (must have ``.name``, ``.read()``).

    Returns
    -------
    tuple[list[dict], list[str]]
        ``(texts, warnings)`` — successfully read files and warning messages.
        Failed files are skipped with a warning, not raised.
    """
    texts: list[dict[str, str]] = []
    warnings: list[str] = []

    # Total size check.
    total_bytes = sum(f.size for f in uploaded_files)
    if total_bytes / _MB > MAX_TOTAL_SIZE_MB:
        warnings.append(
            f"Total upload size ({total_bytes / _MB:.1f} MB) exceeds "
            f"the {MAX_TOTAL_SIZE_MB} MB limit. Please remove some files."
        )
        return [], warnings

    # Duplicate filename check (decision 163).
    names = [f.name for f in uploaded_files]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            warnings.append(f"Duplicate filename detected: '{name}'.")
        seen.add(name)

    for uf in uploaded_files:
        try:
            file_bytes = uf.read()
            result = read_file(file_bytes, uf.name)
            if not result["content"].strip():
                warnings.append(f"File '{uf.name}' is empty and was skipped.")
                continue
            texts.append(result)
        except FileReadError as exc:
            warnings.append(str(exc))
        except Exception as exc:
            logger.error("Unexpected error reading '%s'", uf.name, exc_info=True)
            warnings.append(f"Could not read '{uf.name}': {exc}")

    return texts, warnings


# ── Text Paste (decision 91) ─────────────────────────────────────────────────


def text_from_paste(pasted: str) -> dict[str, str]:
    """Wrap pasted text into the same format as an uploaded file.

    Parameters
    ----------
    pasted:
        Raw text from the paste area.

    Returns
    -------
    dict
        ``{"filename": "pasted_text.txt", "content": str}``.
    """
    return {"filename": "pasted_text.txt", "content": pasted}


# ── File Preview (decision 65) ───────────────────────────────────────────────


def get_file_preview(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """Return file metadata and a text preview.

    Parameters
    ----------
    file_bytes:
        Raw bytes.
    filename:
        Original filename.

    Returns
    -------
    dict
        ``{"filename", "size_bytes", "size_display", "word_count",
        "preview_text", "error"}``.
    """
    size = len(file_bytes)
    info: dict[str, Any] = {
        "filename": filename,
        "size_bytes": size,
        "size_display": _human_size(size),
        "word_count": 0,
        "preview_text": "",
        "error": "",
    }
    try:
        result = read_file(file_bytes, filename)
        words = result["content"].split()
        info["word_count"] = len(words)
        info["preview_text"] = " ".join(words[:200])
    except FileReadError as exc:
        info["error"] = str(exc)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT READERS
# ═══════════════════════════════════════════════════════════════════════════════


def _read_txt(file_bytes: bytes, filename: str) -> str:
    encoding = detect_encoding(file_bytes)
    try:
        return file_bytes.decode(encoding)
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="replace")


def _read_pdf(file_bytes: bytes, filename: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise FileReadError(
            "PDF support requires pdfplumber. Install it with: "
            "pip install pdfplumber"
        )

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                raise FileReadError(
                    f"'{filename}' appears to be corrupted or has no pages."
                )

            texts: list[str] = []
            image_only_pages = 0

            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        texts.append(text)
                    else:
                        image_only_pages += 1
                except Exception:
                    image_only_pages += 1

            if not texts:
                if image_only_pages > 0:
                    raise FileReadError(
                        f"'{filename}' appears to be a scanned/image-only PDF. "
                        "Lemmata requires text-based PDFs. Consider using OCR "
                        "software first."
                    )
                raise FileReadError(
                    f"'{filename}' could not be read. The file may be corrupted."
                )

            if image_only_pages > 0:
                logger.info(
                    "%s: %d image-only pages skipped", filename, image_only_pages
                )

            return "\n\n".join(texts)

    except FileReadError:
        raise
    except Exception as exc:
        msg = str(exc).lower()
        if "password" in msg or "encrypt" in msg:
            raise FileReadError(
                f"'{filename}' is password-protected. "
                "Please remove the protection before uploading."
            )
        raise FileReadError(
            f"'{filename}' could not be read. The file may be corrupted. "
            f"Detail: {exc}"
        )


def _read_docx(file_bytes: bytes, filename: str) -> str:
    try:
        from docx import Document
    except ImportError:
        raise FileReadError(
            "DOCX support requires python-docx. Install it with: "
            "pip install python-docx"
        )

    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as exc:
        raise FileReadError(
            f"Could not read DOCX file '{filename}'. "
            f"The file may be corrupted. Detail: {exc}"
        )


def _read_odt(file_bytes: bytes, filename: str) -> str:
    try:
        from odf import text as odf_text
        from odf.opendocument import load as odf_load
    except ImportError:
        raise FileReadError(
            "ODT support requires odfpy. Install it with: pip install odfpy"
        )

    try:
        doc = odf_load(io.BytesIO(file_bytes))
        paragraphs: list[str] = []
        for elem in doc.getElementsByType(odf_text.P):
            raw = _odt_extract_text(elem)
            if raw.strip():
                paragraphs.append(raw)
        return "\n\n".join(paragraphs)
    except Exception as exc:
        raise FileReadError(
            f"Could not read ODT file '{filename}'. "
            f"The file may be corrupted. Detail: {exc}"
        )


def _odt_extract_text(element: Any) -> str:
    """Recursively extract text from an ODF element."""
    parts: list[str] = []
    if hasattr(element, "data"):
        parts.append(element.data)
    for child in element.childNodes:
        parts.append(_odt_extract_text(child))
    return "".join(parts)


def _read_epub(file_bytes: bytes, filename: str) -> str:
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        raise FileReadError(
            "EPUB support requires ebooklib. Install it with: "
            "pip install ebooklib"
        )

    try:
        from html.parser import HTMLParser

        book = epub.read_epub(
            io.BytesIO(file_bytes), options={"ignore_ncx": True}
        )
        texts: list[str] = []

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            html_bytes = item.get_content()
            html_str = html_bytes.decode("utf-8", errors="replace")
            plain = _strip_html(html_str)
            if plain.strip():
                texts.append(plain)

        if not texts:
            raise FileReadError(
                f"'{filename}' contains no readable text content."
            )

        return "\n\n".join(texts)
    except FileReadError:
        raise
    except Exception as exc:
        raise FileReadError(
            f"Could not read EPUB file '{filename}'. "
            f"The file may be corrupted. Detail: {exc}"
        )


class _HTMLStripper:
    """Minimal HTML → plain-text converter."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def feed(self, data: str) -> str:
        from html.parser import HTMLParser

        class _P(HTMLParser):
            def __init__(self, outer: _HTMLStripper) -> None:
                super().__init__()
                self.outer = outer

            def handle_data(self, data: str) -> None:
                self.outer.parts.append(data)

        _P(self).feed(data)
        return "".join(self.parts)


def _strip_html(html: str) -> str:
    """Strip HTML tags, return plain text."""
    stripper = _HTMLStripper()
    return stripper.feed(html)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT — ZIP PACKAGE
# ═══════════════════════════════════════════════════════════════════════════════


def export_zip(results: dict[str, Any], params: dict[str, Any]) -> bytes:
    """Build the main ZIP export with all CSV/JSON artefacts.

    Parameters
    ----------
    results:
        Analysis results from the pipeline.  Expected keys:
        ``topics``, ``doc_topic_matrix``, ``doc_labels``,
        ``preprocessing_trace``, ``model_info``, ``dtm_info``,
        ``coherence``.
    params:
        Parameters used for the analysis (language, n_topics, etc.).

    Returns
    -------
    bytes
        In-memory ZIP file bytes, ready for ``st.download_button``.
    """
    buf = io.BytesIO()
    now = datetime.now(timezone.utc)

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # topic_words.csv
        zf.writestr("topic_words.csv", _topics_to_csv(results["topics"]))

        # doc_topic_matrix.csv
        zf.writestr(
            "doc_topic_matrix.csv",
            _doc_topic_to_csv(
                results["doc_topic_matrix"],
                results["doc_labels"],
                results["topics"],
            ),
        )

        # preprocessing_summary.csv
        zf.writestr(
            "preprocessing_summary.csv",
            _preprocessing_to_csv(results["preprocessing_trace"]),
        )

        # metrics.json
        metrics = {
            "coherence": results.get("coherence", {}),
            "perplexity": results["model_info"].get("perplexity"),
            "log_likelihood": results["model_info"].get("log_likelihood"),
            "converged": results["model_info"].get("converged"),
        }
        zf.writestr("metrics.json", _to_json(metrics))

        # analysis_results.json (decision 119)
        full_results = {
            "lemmata_version": APP_VERSION,
            "timestamp": now.isoformat(),
            "parameters": params,
            "metrics": metrics,
            "dtm_info": results.get("dtm_info", {}),
            "topics": _topics_serialisable(results["topics"]),
            "environment": get_environment_info(params),
        }
        zf.writestr("analysis_results.json", _to_json(full_results))

        # environment.json
        zf.writestr(
            "environment.json", _to_json(get_environment_info(params))
        )

    return buf.getvalue()


def get_zip_filename(language: str, n_topics: int) -> str:
    """Generate the ZIP filename from the template (decision 130).

    Returns
    -------
    str
        e.g. ``lemmata_it_5topics_20260327_143022.zip``
    """
    now = datetime.now(timezone.utc)
    return ZIP_NAME_TEMPLATE.format(
        lang=language,
        n=n_topics,
        date=now.strftime("%Y%m%d"),
        time=now.strftime("%H%M%S"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT — FIGURES
# ═══════════════════════════════════════════════════════════════════════════════


def export_figure_png(fig: Any, dpi: int = CHART_DPI) -> bytes:
    """Export a matplotlib or Altair figure to PNG bytes.

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` or an ``altair.Chart``.
    dpi:
        Resolution (default 300, from config).

    Returns
    -------
    bytes
        PNG image bytes.
    """
    # Altair chart → render via vl-convert or save method.
    if _is_altair(fig):
        return _altair_to_png(fig, dpi)

    # Matplotlib figure.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def export_figure_svg(fig: Any) -> bytes:
    """Export a matplotlib or Altair figure to SVG bytes.

    Wordclouds (matplotlib) should use :func:`export_figure_png` instead.

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` or an ``altair.Chart``.

    Returns
    -------
    bytes
        SVG image bytes.
    """
    if _is_altair(fig):
        return _altair_to_svg(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT — PDF REPORT
# ═══════════════════════════════════════════════════════════════════════════════


def export_pdf_report(results: dict[str, Any], params: dict[str, Any]) -> bytes:
    """Generate a basic PDF report.

    Implements decisions 9 and 40: cover + params + summary + topics +
    heatmap placeholder + text excerpts + environment info.

    Parameters
    ----------
    results:
        Analysis results dict.
    params:
        Parameters used for the analysis.

    Returns
    -------
    bytes
        PDF file bytes.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        raise RuntimeError(
            "PDF report requires reportlab. Install with: pip install reportlab"
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    title_style = ParagraphStyle(
        "LemmataTitle", parent=styles["Title"], fontSize=20, spaceAfter=12
    )
    h2_style = ParagraphStyle(
        "LemmataH2", parent=styles["Heading2"], spaceAfter=8
    )
    body_style = styles["BodyText"]

    # ── Cover ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("Lemmata — Analysis Report", title_style))
    story.append(
        Paragraph(
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), body_style
        )
    )
    story.append(Spacer(1, 2 * cm))

    # ── Parameters ────────────────────────────────────────────────────────
    story.append(Paragraph("Parameters", h2_style))
    param_rows = [[str(k), str(v)] for k, v in params.items()]
    if param_rows:
        t = Table(param_rows, colWidths=[6 * cm, 10 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # ── Summary Metrics ───────────────────────────────────────────────────
    story.append(Paragraph("Summary", h2_style))
    coherence = results.get("coherence", {})
    model_info = results.get("model_info", {})
    summary_lines = [
        f"Topics: {model_info.get('n_topics', '?')}",
        f"C_v coherence: {coherence.get('c_v', '?'):.4f}"
        if isinstance(coherence.get("c_v"), (int, float))
        else f"C_v coherence: {coherence.get('c_v', 'N/A')}",
        f"Perplexity: {model_info.get('perplexity', '?'):.2f}"
        if isinstance(model_info.get("perplexity"), (int, float))
        else f"Perplexity: {model_info.get('perplexity', 'N/A')}",
        f"Log-likelihood: {model_info.get('log_likelihood', '?'):.2f}"
        if isinstance(model_info.get("log_likelihood"), (int, float))
        else f"Log-likelihood: {model_info.get('log_likelihood', 'N/A')}",
    ]
    for line in summary_lines:
        story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 0.5 * cm))

    # ── Topics ────────────────────────────────────────────────────────────
    story.append(Paragraph("Topics", h2_style))
    for topic in results.get("topics", []):
        label = topic.get("label", f"Topic {topic.get('topic_id', '?')}")
        words = ", ".join(topic.get("words", []))
        story.append(Paragraph(f"<b>{label}</b>: {words}", body_style))
    story.append(Spacer(1, 0.5 * cm))

    # ── Environment ───────────────────────────────────────────────────────
    story.append(Paragraph("Environment", h2_style))
    env = get_environment_info(params)
    for k, v in env.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", body_style))

    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT INFO
# ═══════════════════════════════════════════════════════════════════════════════


def get_environment_info(params: dict[str, Any]) -> dict[str, Any]:
    """Collect full environment information (decision 46).

    Parameters
    ----------
    params:
        Analysis parameters (seed, language, n_topics, etc.).

    Returns
    -------
    dict
        Python version, key package versions, all parameters, corpus info.
    """
    info: dict[str, Any] = {
        "lemmata_version": APP_VERSION,
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    # Package versions — best effort.
    for pkg in ("scikit-learn", "spacy", "gensim", "pdfplumber", "reportlab"):
        try:
            from importlib.metadata import version

            info[f"{pkg}_version"] = version(pkg)
        except Exception:
            info[f"{pkg}_version"] = "not installed"

    info["parameters"] = params
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _get_extension(filename: str) -> str:
    """Return lowercase file extension without the dot."""
    if "." in filename:
        return filename.rsplit(".", 1)[1].lower()
    return ""


def _human_size(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} GB"


def _to_json(obj: Any) -> str:
    """Serialise to indented JSON, handling numpy types."""

    def _default(o: Any) -> Any:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return json.dumps(obj, indent=2, ensure_ascii=False, default=_default)


def _topics_to_csv(topics: list[dict[str, Any]]) -> str:
    """Serialise topics to CSV: topic_id, label, word, weight."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["topic_id", "label", "rank", "word", "weight"])
    for t in topics:
        for rank, (w, wt) in enumerate(
            zip(t.get("words", []), t.get("weights", [])), start=1
        ):
            writer.writerow([t["topic_id"], t.get("label", ""), rank, w, f"{wt:.6f}"])
    return buf.getvalue()


def _doc_topic_to_csv(
    matrix: Any, labels: list[str], topics: list[dict[str, Any]]
) -> str:
    """Serialise doc-topic matrix to CSV."""
    import numpy as np

    buf = io.StringIO()
    writer = csv.writer(buf)

    header = ["document"] + [
        t.get("label", f"Topic {t['topic_id']}") for t in topics
    ] + ["dominant_topic"]
    writer.writerow(header)

    arr = np.asarray(matrix)
    for i, label in enumerate(labels):
        row = arr[i]
        dominant = int(np.argmax(row)) + 1
        writer.writerow([label] + [f"{v:.6f}" for v in row] + [dominant])

    return buf.getvalue()


def _preprocessing_to_csv(trace: dict[str, Any]) -> str:
    """Serialise per-document preprocessing summary to CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "document",
        "original_tokens",
        "stopwords_builtin",
        "stopwords_custom",
        "pos_matched",
        "final_lemmas",
    ])
    for doc_trace in trace.get("per_document", []):
        writer.writerow([
            doc_trace["label"],
            doc_trace["original_tokens"],
            doc_trace["stopwords_builtin"],
            doc_trace["stopwords_custom"],
            doc_trace["pos_matched"],
            doc_trace["final_lemmas"],
        ])
    return buf.getvalue()


def _topics_serialisable(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a JSON-safe copy of topic summaries (no numpy)."""
    import numpy as np

    clean: list[dict[str, Any]] = []
    for t in topics:
        clean.append(
            {
                "topic_id": int(t["topic_id"]),
                "label": t.get("label", ""),
                "words": list(t.get("words", [])),
                "weights": [float(w) for w in t.get("weights", [])],
                "avg_weight": float(t.get("avg_weight", 0.0)),
            }
        )
    return clean


def _is_altair(fig: Any) -> bool:
    """Check if an object is an Altair chart."""
    try:
        import altair as alt

        return isinstance(fig, (alt.Chart, alt.LayerChart, alt.HConcatChart, alt.VConcatChart))
    except ImportError:
        return False


def _altair_to_png(chart: Any, dpi: int) -> bytes:
    """Render Altair chart to PNG bytes via vlc or selenium fallback."""
    try:
        import vlc  # vl-convert-python

        return chart.to_image(format="png", scale_factor=dpi / 72)  # type: ignore[return-value]
    except Exception:
        pass

    # Fallback: save via chart method.
    buf = io.BytesIO()
    chart.save(buf, format="png", scale_factor=dpi / 72)
    buf.seek(0)
    return buf.getvalue()


def _altair_to_svg(chart: Any) -> bytes:
    """Render Altair chart to SVG bytes."""
    buf = io.BytesIO()
    chart.save(buf, format="svg")
    buf.seek(0)
    return buf.getvalue()
